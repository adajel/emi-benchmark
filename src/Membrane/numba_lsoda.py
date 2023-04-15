from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc
import numpy as np
import timeit

@cfunc(lsoda_sig)
def rhs(t, u, du, p):
    k1 = p[0]
    k2 = p[1]
    k3 = p[2]
    du[0] = -k1*u[0]+k3*u[1]*u[2]
    du[1] = k1*u[0]-k2*u[1]**2-k3*u[1]*u[2]
    du[2] = k2*u[1]**2
    
u0 = np.r_[1.0, 0.0, 0.0]
t = np.r_[0.0, 1e5]
p = np.r_[0.04, 3e7, 1e4]
rhs_address = rhs.address
usol, success = lsoda(rhs_address, u0, t, data=p, rtol=1.0e-3, atol=1.0e-6)
assert success

@njit
def time_func():
    usol, success = lsoda(rhs_address, u0, t, data=p, rtol=1.0e-3, atol=1.0e-6)
    assert success

time_func()
print('NumbaLSODA', timeit.Timer(time_func).timeit(number=1000)/1000, 'seconds')
print('NumbaLSODA', timeit.Timer(time_func).timeit(number=1000)/1000, 'seconds')
