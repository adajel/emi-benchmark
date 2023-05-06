import dolfin as df
import numpy as np

from petsc4py import PETSc
print = PETSc.Sys.Print

# --------------------------------------------------------------------

from facet_plot import vtk_plot, VTKSeries
from membrane import MembraneModel
import simple_ode as ode

mesh = df.UnitSquareMesh(32, 32)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
u = df.Function(V)

x, y = V.tabulate_dof_coordinates().T
u.vector().set_local(np.where(x*(1-x) < 1E-10, 2, 1))

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)

membranes = [MembraneModel(ode, facet_f=facet_f, tag=0, V=V),
             MembraneModel(ode, facet_f=facet_f, tag=1, V=V)]

[membrane.set_ODE_membrane_potential(u) for membrane in membranes]
[membrane.set_ODE_parameter_values({'A11': lambda x: -3}, locator=lambda x: df.near(x[1], 0))
 for membrane in membranes]

stimulus = {'stim_amplitude': 5,
            'stim_period': 1_000_000,
            'stim_duration': 0.1,
            'stim_start': 0}


# series = VTKSeries('test_ode2_parallel', comm=mesh.mpi_comm())
# series.add(vtk_plot(u, facet_f, (0, 1), path=next(series)), time=0)

for _ in range(100):
    # print(membrane.parameters)
    [membrane.step_lsoda(dt=0.01, stimulus=stimulus, stimulus_locator=lambda x: df.near(x[1], 1) | df.near(x[0], 0))
     for membrane in membranes]

    [membrane.get_PDE_membrane_potential(u) for membrane in membranes]
    
    # vtk_plot(u, facet_f, (tag, ), path=f'test_ode_t{membrane.time}.vtk')        
    print(u.vector().norm('l2'))

    # series.add(vtk_plot(u, facet_f, (0, 1), path=next(series)), time=membranes[0].time)
# series.write()
