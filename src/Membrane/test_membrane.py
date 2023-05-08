import dolfin as df
import numpy as np

# --------------------------------------------------------------------

from facet_plot import vtk_plot, VTKSeries
from membrane import MembraneModel
import simple_ode as ode

mesh = df.UnitSquareMesh(1, 1)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
u = df.Function(V)

x, y = V.tabulate_dof_coordinates().T
u.vector().set_local(np.where(x*(1-x) < 1E-10, 2, 1))

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
# df.DomainBoundary().mark(facet_f, 1)
tag = 0

membrane = MembraneModel(ode, facet_f=facet_f, tag=tag, V=V)

membrane.set_membrane_potential(u)
membrane.set_parameter_values({'A11': lambda x: -3}, locator=lambda x: df.near(x[1], 0))

stimulus = {'stim_amplitude': 5,
            'stim_period': 1_000_000,
            'stim_duration': 0.1,
            'stim_start': 0}

V_index = ode.state_indices('V')
potential_history = []
u_history = []

series = VTKSeries('test_ode_serial')
series.add(vtk_plot(u, facet_f, (tag, ), path=next(series)), time=0)

for _ in range(100):
    # print(membrane.parameters)
    membrane.step_lsoda(dt=0.01, stimulus=stimulus, stimulus_locator=lambda x: df.near(x[1], 1) | df.near(x[0], 0))

    membrane.get_membrane_potential(u)

    potential_history.append(1*membrane.states[:, V_index])
    u_history.append(u.vector().get_local())

    series.add(vtk_plot(u, facet_f, (tag, ), path=next(series)), time=membrane.time)    
    
    print(u.vector().norm('l2'))
series.write()

potential_history, u_history = map(np.array, (potential_history, u_history))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

for ode in range(potential_history.shape[1]):
    line, = ax[0].plot(potential_history[:, ode], label=str(ode), linestyle='none', marker='x')
    ax[1].plot(u_history[:, ode], label=f'PDE {ode}', color=line.get_color())    
plt.legend()
plt.show()

# TODO:
# - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
# - after stepping u should be fine
# - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
# - things are currently quite slow -> multiprocessing?
# - rely on cbc.beat?
