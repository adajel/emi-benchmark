from dolfin import *
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

parameters['ghost_mode'] = 'shared_vertex'

if __name__ == '__main__':
    from mms import setup_mms
    from solver import Solver
    from collections import namedtuple
    from itertools import chain

    GREEN = '\033[1;37;32m%s\033[0m'

    dt_0 = 1.0e-2
    Tstop = dt_0*2      # end time

    degree = 1

    hs, errors_ca, errors_cb, errors_phi = [], [], [], []

    i = 0
    for resolution in range(3, 7):

        t = Constant(0.0)    # time constant (s)
        #dt = dt_0/(4**i)     # time step
        dt = dt_0

        C_M = 1; phi_M_init = 1; C_phi = 0.1
        K1 = 1; K2 = 1;

        # Make some parameters up
        params = namedtuple('params', ('dt', 'C_M', 'C_phi', 'phi_M_init',
            'K1', 'K2'))(dt, C_M, C_phi, phi_M_init, K1, K2)

        t = Constant(0.0)

        mms = setup_mms(params, t)

        phi1 = mms.solution['phi_1']
        phi2 = mms.solution['phi_2']

        # Recall our geometry is
        #      ______
        #     [      ]
        #     [  ()  ]
        #     [______]
        #
        # Wher () is the surfaces['inner']. On the outer surfaces we prescribe
        # Neumann bcs on some part and Dirichlet on the rest.

        S = Solver(params=params, degree=degree, mms=mms)

        # get mesh, subdomains, surfaces path
        here = os.path.abspath(os.path.dirname(__file__))
        mesh_prefix = os.path.join(here, 'meshes/MMS/')
        mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
        subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
        surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

        # generate mesh if it does not exist
        if not os.path.isfile(mesh_path):
            script = 'make_mesh_MMS.py '                           # script path
            os.system('python3 ' + script + ' ' + str(resolution)) # run script

        mesh = Mesh(mesh_path)
        subdomains = MeshFunction('size_t', mesh, subdomains_path)
        surfaces = MeshFunction('size_t', mesh, surfaces_path)

        S.setup_domain(mesh, subdomains, surfaces)

        uh_phi = S.solve_system_passive(Tstop, t)

        # Compute L^2 error (on subdomains)
        dX = Measure('dx', domain=mesh, subdomain_data=subdomains)

        # compute error concentration a

        if S.lagrange:
            # compute error phi with Lagrange multiplier
            error_phi = inner(phi2 - uh_phi, phi2 - uh_phi)*dX(2) \
                      + inner(phi1 - uh_phi, phi1 - uh_phi)*dX(1)
            error_phi = sqrt(abs(assemble(error_phi)))
        else:
            # compute error phi with norm for null_space solver for phi
            phi1_m = Constant(assemble(phi1*dX(1)))
            phi2_m = Constant(assemble(phi2*dX(2)))
            phi_m = phi1_m + phi2_m

            print("MEAN 1", float(phi1_m))
            print("MEAN 2", float(phi2_m))

            error_phi = inner(phi2 - phi_m - uh_phi, phi2 - phi_m - uh_phi)*dX(2) \
                      + inner(phi1 - phi_m - uh_phi, phi1 - phi_m - uh_phi)*dX(1)
            error_phi = sqrt(abs(assemble(error_phi)))

        # append mesh size and errors
        hs.append(mesh.hmin())
        errors_phi.append(error_phi)

        if len(errors_phi) > 1:
            rate_phi = np.log(errors_phi[-1]/errors_phi[-2])/np.log(hs[-1]/hs[-2])
        else:
            rate_phi = np.nan

        msg = f'|phi-phih|_0 = {error_phi:.4E} [{rate_phi:.2f}]'
        mesh.mpi_comm().rank == 0 and print(GREEN % msg)

        V = FunctionSpace(mesh, "CG", 1)
        phi1_ = project(phi1, V)
        phi2_ = project(phi2, V)

        i += 1

    print("phi")
    print(rate_phi)
    for i in range(len(hs)):
        print(hs[i], errors_phi[i])

    #print("hs")
    #print(hs)

    #print("ca")
    #print(errors_ca)
    #print(rate_ca)

    #print("cb")
    #print(errors_cb)
    #print(rate_cb)

    #print("phi")
    #print(errors_phi)
    #print(rate_phi)
