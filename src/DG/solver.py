from dolfin import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

from utils import pcws_constant_project
from utils import interface_normal, plus, minus

JUMP = lambda f, n: minus(f, n) - plus(f, n)

parameters['ghost_mode'] = 'shared_vertex'

# We here approximate the EMI system.
#
# Membrane potential is defined as phi_i - phi_e, since we have
# marked cell in ECS with 2 and cells in ICS with 1 we have an
# interface normal pointing inwards
#    ____________________
#   |                    |
#   |      ________      |
#   |     |        |     |
#   | ECS |   ICS  |     |
#   |  2  |->  1   |     |
#   | (+) |   (-)  |     |
#   |     |________|     |
#   |                    |
#   |____________________|
#
# Normal will always point from higher to lower (e.g. from 2 -> 1)

class Solver:
    def __init__(self, params, degree=1, mms=None):
        """ Initialize solver """

        self.degree = degree
        self.mms = mms
        self.params = params

        # Lagrange
        #self.lagrange = False
        self.lagrange = True

        self.sf = 10

        return

    def setup_domain(self, mesh, subdomains, surfaces):
        """ Setup mesh and associated parameters, and element spaces """

        # set mesh, subdomains, and surfaces
        self.mesh = mesh
        self.subdomains = subdomains
        self.surfaces = surfaces

        # define measures
        self.dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=surfaces)
        self.dS = Measure('dS', domain=mesh, subdomain_data=surfaces)

        # facet area and normal
        self.n = FacetNormal(mesh)
        self.hA = FacetArea(self.mesh)

        # interface normal
        self.n_g = interface_normal(subdomains, mesh)

        # DG penalty parameter
        gdim = self.mesh.geometry().dim()
        self.tau = Constant(10*gdim*self.degree)

        # DG elements for ion concentrations and the potential
        self.PK = FiniteElement('DG', mesh.ufl_cell(), self.degree)

        # For the MMS problem, we need unique tags for each of the interface walls
        if self.mms is not None: self.lm_tags = [1, 2, 3, 4]

        return


    def setup_parameters(self):
        """ setup physical parameters """

        params = self.params

        # get physical parameters
        self.C_phi = Constant(params.C_phi)
        self.C_M = Constant(params.C_M)
        self.dt = Constant(params.dt)
        self.phi_M_init = Constant(params.phi_M_init)

        self.E_Na = Constant(params.E_Na)
        self.E_K = Constant(params.E_K)
        self.g_Na_leak = params.g_Na_leak
        self.g_K_leak = Constant(params.g_K_leak)
        self.g_ch_syn = params.g_ch_syn

        # project diffusion coefficients to PK based on subdomain
        self.K = self.make_global(Constant(self.params.K1), Constant(self.params.K2))

        return

    def setup_solver_PDE(self):
        """ setup variational formulations """

        # create function space for potential (phi)
        if self.lagrange:
            R0 = FiniteElement('Real', self.mesh.ufl_cell(), 0)
            elms = MixedElement([self.PK, R0])
            self.V_pot = FunctionSpace(self.mesh, elms)
            # function for solution potential and Lagrange multiplier
            self.phi_lag = Function(self.V_pot)
        else:
            self.V_pot = FunctionSpace(self.mesh, self.PK)
            # function for solution potential
            self.phi = Function(self.V_pot)

        # define function space of piecewise constants on interface gamma for solution to ODEs
        self.Q = FunctionSpace(self.mesh, 'Discontinuous Lagrange Trace', 0)

        # set initial membrane potential
        self.phi_M_prev_PDE = pcws_constant_project(self.phi_M_init, self.Q)

        # setup variational formulation
        self.setup_varform_emi()

        return


    def setup_varform_emi(self):

        dx = self.dx; ds = self.ds; dS = self.dS

        # get facet area and normal
        hA = self.hA; n = self.n
        tau = self.tau; n_g = self.n_g

        # get physical parameters
        C_phi = self.C_phi
        C_M = self.C_M

        E_Na = self.E_Na
        E_K = self.E_K
        g_Na_leak = self.g_Na_leak
        g_K_leak = self.g_K_leak
        g_ch_syn = self.g_ch_syn

        if self.lagrange:
            # test and trial functions
            u_phi, _c = TrialFunctions(self.V_pot)
            v_phi, _d = TestFunctions(self.V_pot)
        else:
            # test and trial functions
            u_phi = TrialFunction(self.V_pot)
            v_phi = TestFunction(self.V_pot)

        # initialize form
        a = 0; L = 0

        # total channel current
        I_ch = g_Na_leak*(self.phi_M_prev_PDE - E_Na) \
             + g_ch_syn*(self.phi_M_prev_PDE - E_Na) \
             + g_K_leak*(self.phi_M_prev_PDE - E_K)

        # equation potential (drift terms)
        a += inner(self.K*grad(u_phi), grad(v_phi)) * dx \
           - inner(dot(avg(self.K*grad(u_phi)), plus(n, n_g)), jump(v_phi)) * dS(0) \
           - inner(dot(avg(self.K*grad(v_phi)), plus(n, n_g)), jump(u_phi)) * dS(0) \
           + tau/avg(hA) * inner(jump(self.K*u_phi), jump(v_phi)) * dS(0)

        if self.lagrange:
            # Lagrange multiplier to enforce mean zero of phi
            a += inner(u_phi, _d)*dx + inner(_c, v_phi)*dx

        # get and add coupling function on membrane interface for phi
        if self.mms is not None:

            lm_tags = self.lm_tags

            g_robin_phi = self.mms.rhs['bdry']['u_phi']
            g_flux_cont = self.mms.rhs['bdry']['stress']

            fphi1 = self.mms.rhs['volume_phi_1']      # potential in domain 1
            fphi2 = self.mms.rhs['volume_phi_2']      # potential in domain 2

            phi1e = self.mms.solution['phi_1']
            phi2e = self.mms.solution['phi_2']

            bdry_neumann = self.mms.rhs['bdry']['neumann']

            # add coupling term at interface (equation for potential)
            a += sum(C_phi * inner(jump(u_phi), jump(v_phi))*dS(tag) for tag in lm_tags)

            # add robin condition at interface
            L -= sum(C_phi * inner(g_robin_phi[tag], jump(v_phi)) * dS(tag) for tag in lm_tags)

            # add source terms
            L += inner(fphi1, v_phi)*dx(1) \
               + inner(fphi2, v_phi)*dx(2) \

            # we don't have normal cont. of I_M across interface
            L += sum(inner(g_flux_cont[tag], plus(v_phi, n_g)) * dS(tag) for tag in lm_tags)

            if self.lagrange:
                # terms to match the mean of exact solution (not necessary zero)
                target = assemble(inner(phi2e, _d)*dx(2) + inner(phi1e, _d)*dx(1)).sum()
                L += inner(_d, target)*dx

            # get and add coupling function on membrane interface for phi
            L += - dot(bdry_neumann, n) * v_phi * ds

        if self.mms is None:
            # coupling condition at interface
            if self.splitting_scheme:
                g_robin_phi = self.phi_M_prev_PDE
            else:
                g_robin_phi = self.phi_M_prev_PDE - self.dt/C_M * self.I_ch
                #g_robin_phi_gap = self.phi_M_prev_PDE - self.dt/C_M * self.I_gap

            L -= C_phi * inner(avg(g_robin_phi), JUMP(v_phi, n_g)) * dS(1) \
               + C_phi * inner(avg(g_robin_phi), JUMP(v_phi, n_g)) * dS(10)

            #L -= C_phi * inner(avg(g_robin_phi_gap), JUMP(v_phi, n_g)) * dS(xx) \
               #+ C_phi * inner(avg(g_robin_phi_gap), JUMP(v_phi, n_g)) * dS(xx)

            # add coupling term (equation for potential)
            a += - C_phi * inner(jump(u_phi), jump(v_phi))*dS(1) \
                 - C_phi * inner(jump(u_phi), jump(v_phi))*dS(10)

        if self.lagrange:
            self.A_emi = a
            self.L_emi = L
        else:
            self.B_pot = assemble(a + u_phi*v_phi*dx)
            self.A_pot = assemble(a)
            self.b_pot = assemble(L)

        return


    def solve_emi_lagrange(self):

        problem = LinearVariationalProblem(self.A_emi, self.L_emi, self.phi_lag)
        solver_LU = LinearVariationalSolver(problem)
        solver_LU.parameters['linear_solver'] = 'lu'
        solver_LU.solve()

        return


    def solve_emi_nullspace(self):
        ''' Solve for potential (nullspace) '''

        A = self.A_pot
        B = self.B_pot
        b = self.b_pot

        # create vector that spans the null space and normalize
        null_vec = Vector(self.phi.vector())
        self.V_pot.dofmap().set(null_vec, 1.0)
        null_vec *= 1.0/null_vec.norm("l2")

        # create null space basis object and attach to PETSc matrix
        null_space = VectorSpaceBasis([null_vec])
        as_backend_type(A).set_nullspace(null_space)

        # orthogonalize b with respect to the null space
        null_space.orthogonalize(b)

        solver = PETScKrylovSolver()
        solver.set_operators(A, B)
        solver.parameters['monitor_convergence'] = False
        solver.parameters['relative_tolerance'] = 1E-12

        ksp = solver.ksp()
        ksp.setType('gmres')
        ksp.getPC().setType('lu')
        ksp.setComputeEigenvalues(1)

        niters = solver.solve(self.phi.vector(), b)

        return


    def solve_for_time_step(self, k, t):
        """ solve system for one global time step dt"""

        # update time
        t.assign(float(t + self.dt))

        # solve for potential with previous values of concentrations
        if self.lagrange:
            self.solve_emi_lagrange()
            # update membrane potential
            phi, _ = split(self.phi_lag)
            phi_M_step_I = minus(phi, self.n_g) - plus(phi, self.n_g)
            assign(self.phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, self.Q))
        else:
            self.solve_emi_nullspace()
            # update membrane potential
            phi_M_step_I = minus(self.phi, self.n_g) - plus(self.phi, self.n_g)
            assign(self.phi_M_prev_PDE, pcws_constant_project(phi_M_step_I, self.Q))

        # PRINT FOR DEBUG
        if self.mms is None:
            dS = self.dS
            iface_size = assemble(Constant(1)*dS(10))

            if self.lagrange:
                phi, _ = self.phi_lag.split(deepcopy=True)
            else:
                phi = self.phi

            # print every sf timesteps
            if (k % self.sf) == 0:
                phi_M_ = 1.0e3*assemble(1.0/iface_size*avg(self.phi_M_prev_PDE)*dS(10))
                phi_i_trace = 1.0e3*assemble(1.0/iface_size*minus(phi, self.n_g)*dS(10))
                phi_e_trace = 1.0e3*assemble(1.0/iface_size*plus(phi, self.n_g)*dS(10))

                print("k:", k)
                print("ICS phi:", phi_i_trace)
                print("ECS phi:", phi_e_trace)
                print("phi M:", phi_M_)

                self.time_list[int(k/self.sf)] = phi_M_

        return

    def solve_system_passive(self, Tstop, t, filename=None):

        # splitting scheme
        self.splitting_scheme = False

        # setup physical parameters
        self.setup_parameters()
        # setup function spaces and numerical parameters
        self.setup_solver_PDE()

        # open files for saving bulk results
        f_pot = File('results/passive/pot.pvd')

        # print and plot membrane potential
        num_it = int(round(Tstop/float(self.dt))/self.sf)
        self.time_list = np.zeros(num_it)

        for k in range(int(round(Tstop/float(self.dt)))):

            print("k", k)
            self.solve_for_time_step(k, t)

            """
            if self.lagrange:
                phi, _ = self.phi_lag.split(deepcopy=True)
            else:
                phi = self.phi

            if (k % self.sf) == 0 and self.mms is None:
                # save potential to file
                f_pot << (phi, k)
            """

        # combine solution for the potential and concentrations
        if self.lagrange:
            uh_phi, _ = split(self.phi_lag)
        else:
            uh_phi = self.phi

        # plot phi_M over time
        plt.figure()
        plt.plot(self.time_list)
        plt.savefig("time_passive.png")
        plt.close()

        return uh_phi

    def solve_system_active(self, Tstop, t, filename=None):

        # splitting scheme
        self.splitting_scheme = True

        # setup physical parameters
        self.setup_parameters()
        # setup function spaces and numerical parameters
        self.setup_solver_PDE()

        # get HH parameters
        g_Na_bar = Constant(self.params.g_Na_bar)    # Na conductivity HH (S/m^2)
        g_K_bar = Constant(self.params.g_K_bar)      # K conductivity HH (S/m^2)

        g_Na_leak = self.g_Na_leak
        g_K_leak = self.g_K_leak
        g_ch_syn = self.g_ch_syn

        E_Na = self.E_Na
        E_K = self.E_K

        C_M = self.C_M

        # get initial values and project to pcws
        n_HH_init = Constant(self.params.n_init)     # gating variable n
        m_HH_init = Constant(self.params.m_init)     # gating variable m
        h_HH_init = Constant(self.params.h_init)     # gating variable h
        n_HH = pcws_constant_project(n_HH_init, self.Q)
        m_HH = pcws_constant_project(m_HH_init, self.Q)
        h_HH = pcws_constant_project(h_HH_init, self.Q)

        """
        # get ions
        Na = self.ion_list[0]
        K = self.ion_list[1]

        # add HH conductivities to ion conductivities
        Na['g_k'] += g_Na_bar*m_HH**3*h_HH
        K['g_k'] += g_K_bar*n_HH**4

        # total channel current
        I_ch = Na['g_k']*(self.phi_M_prev_PDE - Na['E']) + \
                K['g_k']*(self.phi_M_prev_PDE - K['E'])
        C_M = Constant(self.params.C_M)
        """

        # total channel currents conductivity
        g_Na = g_Na_leak + g_Na_bar*m_HH**3*h_HH
        g_K = g_K_leak + g_K_bar*n_HH**4

        I_ch = g_Na*(self.phi_M_prev_PDE - E_Na) + g_K*(self.phi_M_prev_PDE - E_K)

        # convert phi_M from V to mV
        V_rest = Constant(self.params.V_rest)
        V_M = 1000*(self.phi_M_prev_PDE - V_rest)

        # rate coefficients
        V_M = 1000*(self.phi_M_prev_PDE - V_rest) # convert phi_M to mV
        alpha_n = 0.01e3*(10.-V_M)/(exp((10.-V_M)/10.) - 1.)
        beta_n = 0.125e3*exp(-V_M/80.)
        alpha_m = 0.1e3*(25. - V_M)/(exp((25. - V_M)/10.) - 1)
        beta_m = 4.e3*exp(-V_M/18.)
        alpha_h = 0.07e3*exp(-V_M/20.)
        beta_h = 1.e3/(exp((30.-V_M)/10.) + 1)

        # derivatives for Hodgkin Huxley ODEs
        dphidt = - (1/C_M) * I_ch
        dndt = alpha_n*(1 - n_HH) - beta_n*n_HH
        dmdt = alpha_m*(1 - m_HH) - beta_m*m_HH
        dhdt = alpha_h*(1 - h_HH) - beta_h*h_HH

        # time step for ODEs
        n_steps_ode = 10
        dt_ode = self.dt/n_steps_ode # ODE time step (s)

        # open files for saving bulk results
        f_pot = File('results/active/pot.pvd')

        # print and plot membrane potential
        num_it = int(round(Tstop/float(self.dt))/self.sf)
        self.time_list = np.zeros(num_it)

        for k in range(int(round(Tstop/float(self.dt)))):
            # step I: Solve Hodgkin Hodgkin ODEs using forward Euler
            for i in range(n_steps_ode):
                # get new membrane and gating variables
                phi_M_ODE = pcws_constant_project(avg(self.phi_M_prev_PDE) + dt_ode*avg(dphidt), self.Q)
                n_new = pcws_constant_project(avg(n_HH) + dt_ode*avg(dndt), self.Q)
                m_new = pcws_constant_project(avg(m_HH) + dt_ode*avg(dmdt), self.Q)
                h_new = pcws_constant_project(avg(h_HH) + dt_ode*avg(dhdt), self.Q)

                # update previous membrane potential and gating variables
                assign(self.phi_M_prev_PDE, phi_M_ODE)
                assign(n_HH, n_new)
                assign(m_HH, m_new)
                assign(h_HH, h_new)

            if self.lagrange:
                phi, _ = self.phi_lag.split(deepcopy=True)
            else:
                phi = self.phi

            if (k % self.sf) == 0:
                # save potential to file
                f_pot << (phi, k)

            # solve for one time step
            #self.solve_for_time_step(k, t)

        # plot phi_M over time
        plt.figure()
        plt.plot(self.time_list)
        plt.savefig("time_active_step.png")
        plt.close()

        # combine solution for the potential and concentrations
        if self.lagrange:
            uh_phi, _ = split(self.phi_lag)
        else:
            uh_phi = self.phi

        return uh_phi


    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx = 0
        self.h5_file = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file.write(self.mesh, '/mesh')
        self.h5_file.write(self.subdomains, '/subdomains')
        self.h5_file.write(self.surfaces, '/surfaces')

        return


    def save_h5(self, uh_c, uh_phi):
        """ save results to h5 file """
        self.h5_idx += 1
        self.h5_file.write(uh_c, '/c_solution',  self.h5_idx)
        self.h5_file.write(uh_phi, '/phi_solution',  self.h5_idx)

        return


    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()

        return


    def make_global(self, f1, f2):

        mesh = self.mesh
        subdomains = self.subdomains

        # DG0 space for projecting coefficients
        if self.mms is not None:
            Q = FunctionSpace(self.mesh, self.PK)
        else:
            #Q = FunctionSpace(self.mesh, "DG", 0)
            Q = FunctionSpace(self.mesh, self.PK)

        dofmap = Q.dofmap()

        # dofs in domain 1 and domain 2
        o1_dofs = []
        o2_dofs = []

        for cell in cells(mesh): # compute dofs in the domains
            if subdomains[cell] == 1:
                o1_dofs.extend(dofmap.cell_dofs(cell.index()))
            elif subdomains[cell] == 2:
                o2_dofs.extend(dofmap.cell_dofs(cell.index()))
            else:
                print("cell not marked")
                sys.exit(0)

        o2_dofs = list(set(o2_dofs))
        o1_dofs = list(set(o1_dofs))

        F1 = interpolate(f1, Q)
        F2 = interpolate(f2, Q)
        F = Function(Q)

        F.vector()[o2_dofs] = F2.vector()[o2_dofs]
        F.vector()[o1_dofs] = F1.vector()[o1_dofs]

        return F


    def extract_values(self, u, cell_function, subdomain_id, V, set_value):
        dofmap = V.dofmap()
        mesh = V.mesh()

        for cell in cells(mesh):
            # Preserve only the dofs in cells marked as subdomain_id
            if cell_function[cell.index()] != subdomain_id:
                dofs = dofmap.cell_dofs(cell.index())
                for dof in dofs:
                    u.vector()[dof] = set_value
        return u
