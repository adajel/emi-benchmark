#!/usr/bin/python3

from dolfin import *

import os
import sys

import math as ma
import numpy as np

import matplotlib.pyplot as plt

if __name__=='__main__':

    from solver import Solver
    from collections import namedtuple

    # resolution factor of mesh
    resolution = 0

    # time variables (seconds)
    dt = 1.0e-4                      # global time step (s)
    Tstop = 1.0e-2                   # global end time (s)

    # physical parameters
    C_M = 0.02                       # capacitance
    temperature = 300                # temperature (K)
    F = 96485                        # Faraday's constant (C/mol)
    R = 8.314                        # Gas Constant (J/(K*mol))

    D_Na = Constant(1.33e-9)          # diffusion coefficients Na (m/s)
    D_K = Constant(1.96e-9)           # diffusion coefficients K (m/s)
    D_Cl = Constant(2.03e-9)          # diffusion coefficients Cl (m/s)

    g_Na_bar = 1200                   # Na max conductivity (S/m**2)
    g_K_bar = 360                     # K max conductivity (S/m**2)

    g_K_leak = Constant(8.0*0.5)      # K leak conductivity (S/m**2)
    g_Cl_leak = Constant(0.0)         # Cl leak conductivity (S/m**2)
    a_syn = 0.002                     # synaptic time constant (s)
    g_syn_bar = 40                    # synaptic conductivity (S/m**2)

    # initial values
    V_rest = -0.065                         # resting membrane potential
    phi_M_init = Constant(-0.0677379636231) # membrane potential (V)
    n_init = 0.27622914792                  # gating variable n
    m_init = 0.0379183462722                # gating variable m
    h_init = 0.688489218108                 # gating variable h

    # EMI specific parameters
    sigma_i = 2.011202                # intracellular conductivity
    sigma_e = 1.31365                 # extracellular conductivity
    V_rest = -0.065                   # resting membrane potential
    E_Na = 54.8e-3                    # Nernst potential sodium
    E_K = -88.98e-3                   # Nernst potential potassium
    g_Na_leak_emi = Constant(2.0*0.5) # Na leak conductivity (S/m**2)
    g_K_leak_emi = Constant(8.0*0.5)  # K leak conductivity (S/m**2)

    # calculate parameters
    psi = F/(R*temperature)
    C_phi = C_M/dt

    # time constant
    t = Constant(0.0)

    # Synaptic current stimuli
    g_ch_syn = Expression('g_syn_bar*exp(-fmod(t,0.02)/a_syn)*(x[0]<40e-6)', \
               g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)

    #g_ch_syn = Expression('g_syn_bar', \
               #g_syn_bar=g_syn_bar, a_syn=a_syn, t=t, degree=4)

    g_Na_leak = Constant(2.0*0.5) + g_ch_syn

    # set parameters
    params = namedtuple('params', ('dt', 'F', 'psi', 'phi_M_init', 'C_phi',
        'C_M', 'R', 'temperature', 'n_init', 'm_init', 'h_init', 'V_rest', 
        'g_Na_bar', 'g_K_bar', 'g_Na_leak', 'g_K_leak', 'E_Na', 'E_K',
        'g_ch_syn', 'K1', 'K2'))(dt, F, psi, phi_M_init, C_phi, C_M, R,
        temperature, n_init, m_init, h_init, V_rest, g_Na_bar, g_K_bar,
        g_Na_leak, g_K_leak, E_Na, E_K, g_ch_syn, sigma_i, sigma_e)

    #####################################################################

    # get mesh, subdomains, surfaces paths
    here = os.path.abspath(os.path.dirname(__file__))
    mesh_prefix = os.path.join(here, 'meshes/one_neuron/')
    mesh_path = mesh_prefix + 'mesh_' + str(resolution) + '.xml'
    subdomains_path = mesh_prefix + 'subdomains_' + str(resolution) + '.xml'
    surfaces_path = mesh_prefix + 'surfaces_' + str(resolution) + '.xml'

    # generate mesh if it does not exist
    if not os.path.isfile(mesh_path):
        script = 'make_mesh.py '                # script
        os.system('python3 ' + script + ' ' + str(resolution)) # run script

    mesh = Mesh(mesh_path)
    subdomains = MeshFunction('size_t', mesh, subdomains_path)
    surfaces = MeshFunction('size_t', mesh, surfaces_path)

    # solve system
    S = Solver(params)  # create solver
    S.setup_domain(mesh, subdomains, surfaces) # setup meshes

    filename = 'results/data/one_neuron/'
    uh = S.solve_system_active(Tstop, t)
