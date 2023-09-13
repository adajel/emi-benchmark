# Gotran generated code for the "hodgkin_huxley_squid_axon_model_1952_original" model

import numpy as np
import math


def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    # m=0.05, h=0.6, n=0.325, V=0.0
    init_values = np.array([0.05, 0.6, 0.325, 0.0], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

    for state_name, value in values.items():
        if state_name not in state_ind:
            raise ValueError("{0} is not a state.".format(state_name))
        ind = state_ind[state_name]

        # Assign value
        init_values[ind] = value

    return init_values

def init_parameter_values(**values):
    """
    Initialize parameter values
    """
    # Param values
    # g_Na=120.0, g_K=36.0, g_L=0.3, Cm=1.0, E_R=0.0
    init_values = np.array([120.0, 36.0, 0.3, 1.0, 0.0], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("g_Na", 0), ("g_K", 1), ("g_L", 2), ("Cm", 3), ("E_R",\
        4)])

    for param_name, value in values.items():
        if param_name not in param_ind:
            raise ValueError("{0} is not a parameter.".format(param_name))
        ind = param_ind[param_name]

        # Assign value
        init_values[ind] = value

    return init_values

def state_indices(*states):
    """
    State indices
    """
    state_inds = dict([("m", 0), ("h", 1), ("n", 2), ("V", 3)])

    indices = []
    for state in states:
        if state not in state_inds:
            raise ValueError("Unknown state: '{0}'".format(state))
        indices.append(state_inds[state])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def parameter_indices(*params):
    """
    Parameter indices
    """
    param_inds = dict([("g_Na", 0), ("g_K", 1), ("g_L", 2), ("Cm", 3),\
        ("E_R", 4)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

def monitor_indices(*monitored):
    """
    Monitor indices
    """
    monitor_inds = dict([("E_Na", 0), ("i_Na", 1), ("alpha_m", 2), ("beta_m",\
        3), ("alpha_h", 4), ("beta_h", 5), ("E_K", 6), ("i_K", 7),\
        ("alpha_n", 8), ("beta_n", 9), ("E_L", 10), ("i_L", 11), ("i_Stim",\
        12), ("dm_dt", 13), ("dh_dt", 14), ("dn_dt", 15), ("dV_dt", 16)])

    indices = []
    for monitor in monitored:
        if monitor not in monitor_inds:
            raise ValueError("Unknown monitored: '{0}'".format(monitor))
        indices.append(monitor_inds[monitor])
    if len(indices)>1:
        return indices
    else:
        return indices[0]


def rhs(states, t, parameters, values=None):
    """
    Compute the right hand side of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Assign states
    assert(len(states) == 4)

    # Assign parameters
    assert(len(parameters) == 5)

    # Init return args
    if values is None:
        values = np.zeros((4,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Expressions for the Sodium channel component
    E_Na = -115 + parameters[4]
    i_Na = parameters[0]*states[1]*(states[0]*states[0]*states[0])*(states[3]\
        - E_Na)

    # Expressions for the m gate component
    alpha_m = (2.5 + 0.1*states[3])/(-1 + math.exp(5/2 + states[3]/10))
    beta_m = 4*math.exp(states[3]/18)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07*math.exp(states[3]/20)
    beta_h = 1.0/(1 + math.exp(3 + states[3]/10))
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the Potassium channel component
    E_K = 12 + parameters[4]
    i_K = parameters[1]*math.pow(states[2], 4)*(states[3] - E_K)

    # Expressions for the n gate component
    alpha_n = (0.1 + 0.01*states[3])/(-1 + math.exp(1 + states[3]/10))
    beta_n = 0.125*math.exp(states[3]/80)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Leakage current component
    E_L = -10.613 + parameters[4]
    i_L = parameters[2]*(states[3] - E_L)

    # Expressions for the Membrane component
    i_Stim = (-20 if t <= 10.5 and t >= 10 else 0)
    values[3] = (-i_K - i_L - i_Na + i_Stim)/parameters[3]

    # Return results
    return values


from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

@cfunc(lsoda_sig, nopython=True) 
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Assign states
    #assert(len(states) == 4)

    # Assign parameters
    #assert(len(parameters) == 5)

    # # Init return args
    # if values is None:
    #     values = np.zeros((4,), dtype=np.float_)
    # else:
    #     assert isinstance(values, np.ndarray) and values.shape == (4,)

    # Expressions for the Sodium channel component
    E_Na = -115. + parameters[4]
    i_Na = parameters[0]*states[1]*(states[0]*states[0]*states[0])*(states[3] - E_Na)

    # Expressions for the m gate component
    alpha_m = (2.5 + 0.1*states[3])/(-1 + math.exp(5/2 + states[3]/10))
    beta_m = 4*math.exp(states[3]/18)
    values[0] = (1 - states[0])*alpha_m - states[0]*beta_m

    # Expressions for the h gate component
    alpha_h = 0.07*math.exp(states[3]/20)
    beta_h = 1.0/(1 + math.exp(3 + states[3]/10))
    values[1] = (1 - states[1])*alpha_h - states[1]*beta_h

    # Expressions for the Potassium channel component
    E_K = 12 + parameters[4]
    i_K = parameters[1]*math.pow(states[2], 4)*(states[3] - E_K)

    # Expressions for the n gate component
    alpha_n = (0.1 + 0.01*states[3])/(-1 + math.exp(1 + states[3]/10))
    beta_n = 0.125*math.exp(states[3]/80)
    values[2] = (1 - states[2])*alpha_n - states[2]*beta_n

    # Expressions for the Leakage current component
    E_L = -10.613 + parameters[4]
    i_L = parameters[2]*(states[3] - E_L)

    # Expressions for the Membrane component
    i_Stim = (-20 if t <= 10.5 and t >= 10 else 0)
    values[3] = (-i_K - i_L - i_Na + i_Stim)/parameters[3]


def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the\
        hodgkin_huxley_squid_axon_model_1952_original ODE
    """

    # Assign states
    assert(len(states) == 4)

    # Assign parameters
    assert(len(parameters) == 5)

    # Init return args
    if monitored is None:
        monitored = np.zeros((17,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (17,)

    # Expressions for the Sodium channel component
    monitored[0] = -115 + parameters[4]
    monitored[1] =\
        parameters[0]*states[1]*(states[0]*states[0]*states[0])*(states[3] -\
        monitored[0])

    # Expressions for the m gate component
    monitored[2] = (2.5 + 0.1*states[3])/(-1 + math.exp(5/2 + states[3]/10))
    monitored[3] = 4*math.exp(states[3]/18)
    monitored[13] = (1 - states[0])*monitored[2] - states[0]*monitored[3]

    # Expressions for the h gate component
    monitored[4] = 0.07*math.exp(states[3]/20)
    monitored[5] = 1.0/(1 + math.exp(3 + states[3]/10))
    monitored[14] = (1 - states[1])*monitored[4] - states[1]*monitored[5]

    # Expressions for the Potassium channel component
    monitored[6] = 12 + parameters[4]
    monitored[7] = parameters[1]*math.pow(states[2], 4)*(states[3] -\
        monitored[6])

    # Expressions for the n gate component
    monitored[8] = (0.1 + 0.01*states[3])/(-1 + math.exp(1 + states[3]/10))
    monitored[9] = 0.125*math.exp(states[3]/80)
    monitored[15] = (1 - states[2])*monitored[8] - states[2]*monitored[9]

    # Expressions for the Leakage current component
    monitored[10] = -10.613 + parameters[4]
    monitored[11] = parameters[2]*(states[3] - monitored[10])

    # Expressions for the Membrane component
    monitored[12] = (-20 if t <= 10.5 and t >= 10 else 0)
    monitored[16] = (-monitored[11] - monitored[1] - monitored[7] +\
        monitored[12])/parameters[3]

    # Return results
    return monitored
