import dolfin as df
import numpy as np

# --------------------------------------------------------------------

from facet_plot import vtk_plot
from membrane import MembraneModel
import simple_ode as ode

mesh = df.UnitSquareMesh(1, 1)
V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
u = df.Function(V)

x, y = V.tabulate_dof_coordinates().T
u.vector().set_local(np.where(x*(1-x) < 1E-10, 2, 1))

facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
df.CompiledSubDomain('near(x[0], 0)').mark(facet_f, 1)
tag = 0

membranes = [MembraneModel(ode, facet_f=facet_f, tag=0, V=V),
             MembraneModel(ode, facet_f=facet_f, tag=1, V=V)]

[membrane.set_ODE_membrane_potential(u) for membrane in membranes]
[membrane.set_ODE_parameter_values({'A11': lambda x: -3}, locator=lambda x: df.near(x[1], 0))
 for membrane in membranes]

stimulus = {'stim_amplitude': 5,
            'stim_period': 1_000_000,
            'stim_duration': 0.1,
            'stim_start': 0}

potential_histories = [[], []]
u_history = []

# vtk_plot(u, facet_f, (tag, ), path=f'test_ode_t{membrane.time}.vtk')    
for _ in range(100):
    # print(membrane.parameters)
    [membrane.step_lsoda(dt=0.01, stimulus=stimulus, stimulus_locator=lambda x: df.near(x[1], 1) | df.near(x[0], 0))
     for membrane in membranes]

    [membrane.get_PDE_membrane_potential(u) for membrane in membranes]

    [potential_history.append(1*membrane.states[:, membrane.V_index])
     for potential_history, membrane in zip(potential_histories, membranes)]
    
    u_history.append(u.vector().get_local())
    
    # vtk_plot(u, facet_f, (tag, ), path=f'test_ode_t{membrane.time}.vtk')        
    print(u.vector().norm('l2'))


potential_history0, potential_history1, u_history = map(np.array, (potential_histories[0],
                                                                   potential_histories[1],
                                                                   u_history))

import matplotlib.pyplot as plt
import itertools
fix, ax = plt.subplots(1, 2)

for idx, sol in enumerate(itertools.chain(potential_history0.T, potential_history1.T)):
    line, = ax[0].plot(sol, label=str(idx), linestyle='none', marker='x')
    ax[1].plot(u_history[:, idx], label=f'PDE {idx}', color=line.get_color())    
plt.legend()
plt.show()

# TODO:
# - consider a test where we have dy/dt = A(x)y with y(t=0) = y0
# - after stepping u should be fine
# - add forcing:  dy/dt = A(x)y + f(t) with y(t=0) = y0
# - things are currently quite slow -> multiprocessing?
# - rely on cbc.beat?
