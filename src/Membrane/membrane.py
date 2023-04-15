from scipy.integrate import odeint
import dlt_dof_extraction as dlt
import dolfin as df
import numpy as np

from numbalsoda import lsoda

class MembraneModel():
    '''
    Facets where facet_f[facet] == tag are governed by this ode whose 
    source terms will be taken from V
    '''
    def __init__(self, ode, facet_f, tag, V):
        mesh = facet_f.mesh()
        assert mesh.topology().dim()-1 == facet_f.dim()
        assert isinstance(tag, int)
        # ODE will talk to PDE via a H(div)-trace function - we need to
        # know which indices of that function will be used for the communication
        assert dlt.is_dlt_scalar(V)
        self.V = V
        
        self.facets, indices = dlt.get_indices(V, facet_f, (tag, ))
        self.indices = indices.flatten()

        self.dof_locations = V.tabulate_dof_coordinates()[self.indices]
        # For every spatial point there is an ODE with states/parameters which need
        # to be tracked
        nodes = len(self.indices)        
        df.info(f'\tNumber of ODE points on the membrane {nodes}')
        self.nodes = nodes
        
        self.states = np.array([ode.init_state_values() for _ in range(nodes)])
        self.parameters = np.array([ode.init_parameter_values() for _ in range(nodes)])

        self.ode = ode
        self.time = 0
        
        self.V_index = ode.state_indices('V')

    def set_parameter_values(self, param_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        lidx = np.arange(self.nodes)        
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]
        df.info(f'\tSet parameters for {len(lidx)} ODES')            
        coords = self.dof_locations[lidx]
        
        for param in param_dict:
            col = self.ode.parameter_indices(param)
            get_param_value = param_dict[param]
            for row, x in zip(lidx, coords):  # Counts the odes
                self.parameters[row, col] = get_param_value(x)
        return self.parameters
            
    def set_state_values(self, state_dict, locator=None):
        ''' param_name -> (lambda x: value)'''
        if 'V' in state_dict:
            raise ValueError('Use `set_ODE_membrane_potential` for its update!')
        
        lidx = np.arange(self.nodes)        
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]
        df.info(f'\tSet states for {len(lidx)} ODES')
        coords = self.dof_locations[lidx]
        
        for param in state_dict:
            col = self.ode.state_indices(param)
            get_param_value = state_dict[param]
            for row, x in zip(lidx, coords):  # Counts the odes
                self.states[row, col] = get_param_value(x)
        return self.states

    def update_PDE_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        assert self.V.ufl_element() == u.function_space().ufl_element()
        
        lidx = np.arange(self.nodes)        
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        potentials = u.vector().get_local()
        potentials[self.indices[lidx]] = self.states[lidx, self.V_index]

        u.vector().set_local(potentials)
        df.as_backend_type(u.vector()).update_ghost_values()

        return u

    def set_ODE_membrane_potential(self, u, locator=None):
        '''Update PDE potentials from the ODE solver'''
        assert self.V.ufl_element() == u.function_space().ufl_element()
        
        lidx = np.arange(self.nodes)        
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        potentials = u.vector().get_local()
        self.states[lidx, self.V_index] = potentials[self.indices[lidx]]

        return self.states

    def step(self, dt, stimulus, locator=None):
        df.info(f'\tBefore ODE t = {self.time} |V| = {np.linalg.norm(self.states[:, self.V_index])}')

        ode_rhs_address = self.ode.rhs_numba.address
        
        lidx = np.arange(self.nodes)        
        if locator is not None:
            lidx = lidx[np.fromiter(map(locator, self.dof_locations), dtype=bool)]

        tsteps = np.array([self.time, self.time+dt])
        for row in lidx:  # Count ODEs
            row_parameters = self.parameters[row]
            for key, value in stimulus.items():
                row_parameters[self.ode.parameter_indices(key)] = value
            current_state = self.states[row]

            # new_state = odeint(ode.rhs, current_state, tsteps, args=(row_parameters, ))

            new_state, success = lsoda(ode_rhs_address,
                                       current_state,
                                       tsteps,
                                       data=row_parameters,
                                       rtol=1.0e-8, atol=1.0e-10)
            assert success
            self.states[row][:] = new_state[-1]
            
        self.time = tsteps[-1]

        df.info(f'\tAfter ODE t = {self.time} |V| = {np.linalg.norm(self.states[:, self.V_index])}')        

# --------------------------------------------------------------------

if __name__ == '__main__':
    import tentusscher_panfilov_2006_M_cell as ode
    from facet_plot import vtk_plot
    
    mesh = df.UnitSquareMesh(128, 128)
    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    u = df.Function(V)

    x, y = V.tabulate_dof_coordinates().T
    u.vector().set_local(np.where(x*(1-x)*y*(1-y) < 1E-10, 1, 0))
    
    facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    # df.DomainBoundary().mark(facet_f, 1)
    tag = 0
    
    membrane = MembraneModel(ode, facet_f=facet_f, tag=tag, V=V)

    membrane.set_ODE_membrane_potential(u)
    membrane.set_parameter_values({'g_Kr': lambda x: 20}, locator=lambda x: df.near(x[0], 0))

    stimulus = {'stim_amplitude': 0.1,
                'stim_period': 0.2,
                'stim_duration': 0.1,
                'stim_start': 0}

    vtk_plot(u, facet_f, (tag, ), path=f'test_ode_t{membrane.time}.vtk')    
    for _ in range(5):
        membrane.step(dt=0.01, stimulus=stimulus)

        membrane.update_PDE_membrane_potential(u)
        vtk_plot(u, facet_f, (tag, ), path=f'test_ode_t{membrane.time}.vtk')        
        print(u.vector().norm('l2'))

    # TODO:
    # - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
    # - after stepping u should be fine
    # - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
    # - things are currently quite slow -> multiprocessing?
    # - rely on cbc.beat?
