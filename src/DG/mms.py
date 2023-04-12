from dolfin import *
import ulfy  # https://github.com/MiroK/ulfy
from collections import namedtuple
import sympy as sp

MMSData = namedtuple('MMSData', ('solution', 'rhs', 'normals'))

def setup_mms(params, t):
    '''We solve EMI on

    [       ]
    [  [ ]  ]
    [       ]

    domain
    '''

    order = 2
    mesh = UnitSquareMesh(2**(2+order), 2**(2+order), 'crossed')
    x, y = SpatialCoordinate(mesh)

    # We will vary this outside
    C_phi, dt, K1, K2 = Constant(1), Constant(1), Constant(1), Constant(1)

    # define exact solutions
    phi_1 = sin(2*pi*x) + cos(2*pi*y)
    phi_2 = cos(2*pi*y)

    sigma_1 = - K1*grad(phi_1)
    sigma_2 = - K2*grad(phi_2)

    # calculate source terms
    f_phi_1 = div(sigma_1)
    f_phi_2 = div(sigma_2)

    # Normal will point from inner to outer; from 1 to 2
    normals = list(map(Constant, ((-1, 0), (0, -1), (1, 0), (0, 1))))

    # We have that f = phi_i - phi_e - dt/C_M * I_M, where I_M = F sum_c dot(z_c*J_c1, n1)
    g_phi = tuple(
              phi_1 - phi_2 - (1/C_phi)*dot(sigma_1, n1)
              for n1 in normals
    )

    # We don't have 0 on the rhs side of the interface, we have
    # F sum_k (z_k_i*J_k_i \cdot n_i) + F sum_k (z_k_e*J_k_e \cdot n_e) = g_J_phi
    g_J_phi = tuple(
        dot(sigma_1, n1) - dot(sigma_2, n1)  # sigma1.n1 + sigma2.n2 = g_sigma
         for n1 in normals
    )

    # What we want to substitute
    C_phi_, dt_, K1_, K2_ = sp.symbols('C_phi, dt, K1, K2')

    subs = {C_phi:C_phi_, dt:dt_, K1:K1_, K2:K2_}

    as_expression = lambda f, subs=subs: ulfy.Expression(f, subs=subs, degree=4,
                                                         C_phi=params.C_phi,
                                                         dt=params.dt,
                                                         K1=params.K1,
                                                         K2=params.K2,
                                                         t=t)

    phi_1_exact, phi_2_exact, sigma_2_exact = map(as_expression, (phi_1, phi_2, sigma_2))

    return MMSData(solution={'phi_1': phi_1_exact,
                             'phi_2': phi_2_exact},
                   rhs={'volume_phi_1': as_expression(f_phi_1),
                        'volume_phi_2': as_expression(f_phi_2),
                        'bdry': {'neumann': sigma_2_exact,
                                 'stress': dict(enumerate(map(as_expression, g_J_phi), 1)),
                                 'u_phi': dict(enumerate(map(as_expression, g_phi), 1))}},
                   normals=[])
