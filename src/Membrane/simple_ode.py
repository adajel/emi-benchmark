# Gotran generated code for the "tentusscher_panfilov_2006_M_cell" model

import numpy as np
import math


def init_state_values(**values):
    """
    Initialize state values
    """
    # Init values
    # Xr1=0.0165, Xr2=0.473, Xs=0.0174, m=0.00165, h=0.749, j=0.6788,
    # d=3.288e-05, f=0.7026, f2=0.9526, fCass=0.9942, s=0.999998,
    # r=2.347e-08, Ca_i=0.000153, R_prime=0.8978, Ca_SR=4.272,
    # Ca_ss=0.00042, Na_i=10.132, V=-85.423, K_i=138.52
    init_values = np.array([1., 1.])
    # State indices and limit checker
    state_ind = dict([("x0", 0), ("V", 1)])

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
    init_values = np.array([-1., 0, 0, -2.,
                            0., 0, 1., 0.])

    # Parameter indices and limit checker
    param_ind = dict([("A00", 0), ("A01", 1), ("A10", 2), ("A11", 3),
                      ("stim_amplitude", 4),
                      ("stim_duration", 5),
                      ("stim_period", 6),
                      ("stim_start", 7)])
    
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
    state_inds = dict([("x0", 0), ("V", 1)])
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
    param_inds = dict([("A00", 0), ("A01", 1), ("A10", 2), ("A11", 3),
                       ("stim_amplitude", 4),
                       ("stim_duration", 5),
                       ("stim_period", 6),
                       ("stim_start", 7)])

    indices = []
    for param in params:
        if param not in param_inds:
            raise ValueError("Unknown param: '{0}'".format(param))
        indices.append(param_inds[param])
    if len(indices)>1:
        return indices
    else:
        return indices[0]

# Numba ---
from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

#@cfunc(lsoda_sig)
@cfunc(lsoda_sig, nopython=True) 
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the tentusscher_panfilov_2006_M_cell ODE
    """
    f = (-parameters[4] if t - parameters[6]*math.floor(t/parameters[6]) <=\
         parameters[5] + parameters[7] and t -\
        parameters[6]*math.floor(t/parameters[6]) >= parameters[7] else 0)
    
    # Expressions for the Reversal potentials component
    values[0] = parameters[0]*states[0] + parameters[1]*states[1] + f
    values[1] = parameters[2]*states[0] + parameters[3]*states[1] + f


def rhs(states, t, parameters, values=None):
    if values is None:
        values = np.zeros(2)

    f = (-parameters[4] if t - parameters[6]*math.floor(t/parameters[6]) <=\
         parameters[5] + parameters[7] and t -\
        parameters[6]*math.floor(t/parameters[6]) >= parameters[7] else 0)
        
    values[0] = parameters[0]*states[0] + parameters[1]*states[1] + f
    values[1] = parameters[2]*states[0] + parameters[3]*states[1] + f

    return values
    
