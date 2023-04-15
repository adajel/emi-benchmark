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
    init_values = np.array([0.0165, 0.473, 0.0174, 0.00165, 0.749, 0.6788,\
        3.288e-05, 0.7026, 0.9526, 0.9942, 0.999998, 2.347e-08, 0.000153,\
        0.8978, 4.272, 0.00042, 10.132, -85.423, 138.52], dtype=np.float_)

    # State indices and limit checker
    state_ind = dict([("Xr1", 0), ("Xr2", 1), ("Xs", 2), ("m", 3), ("h", 4),\
        ("j", 5), ("d", 6), ("f", 7), ("f2", 8), ("fCass", 9), ("s", 10),\
        ("r", 11), ("Ca_i", 12), ("R_prime", 13), ("Ca_SR", 14), ("Ca_ss",\
        15), ("Na_i", 16), ("V", 17), ("K_i", 18)])

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
    # P_kna=0.03, g_K1=5.405, g_Kr=0.153, g_Ks=0.098, g_Na=14.838,
    # g_bna=0.00029, g_CaL=3.98e-05, g_bca=0.000592, g_to=0.294,
    # K_mNa=40, K_mk=1, P_NaK=2.724, K_NaCa=1000, K_sat=0.1,
    # Km_Ca=1.38, Km_Nai=87.5, alpha=2.5, gamma=0.35, K_pCa=0.0005,
    # g_pCa=0.1238, g_pK=0.0146, Buf_c=0.2, Buf_sr=10, Buf_ss=0.4,
    # Ca_o=2, EC=1.5, K_buf_c=0.001, K_buf_sr=0.3, K_buf_ss=0.00025,
    # K_up=0.00025, V_leak=0.00036, V_rel=0.102, V_sr=0.001094,
    # V_ss=5.468e-05, V_xfer=0.0038, Vmax_up=0.006375, k1_prime=0.15,
    # k2_prime=0.045, k3=0.06, k4=0.005, max_sr=2.5, min_sr=1.0,
    # Na_o=140, Cm=0.185, F=96485.3415, R=8314.472, T=310,
    # V_c=0.016404, stim_amplitude=52, stim_duration=1, stim_period=1000,
    # stim_start=10, K_o=5.4
    init_values = np.array([0.03, 5.405, 0.153, 0.098, 14.838, 0.00029,\
        3.98e-05, 0.000592, 0.294, 40, 1, 2.724, 1000, 0.1, 1.38, 87.5, 2.5,\
        0.35, 0.0005, 0.1238, 0.0146, 0.2, 10, 0.4, 2, 1.5, 0.001, 0.3,\
        0.00025, 0.00025, 0.00036, 0.102, 0.001094, 5.468e-05, 0.0038,\
        0.006375, 0.15, 0.045, 0.06, 0.005, 2.5, 1.0, 140, 0.185, 96485.3415,\
        8314.472, 310, 0.016404, 52, 1, 1000, 10, 5.4], dtype=np.float_)

    # Parameter indices and limit checker
    param_ind = dict([("P_kna", 0), ("g_K1", 1), ("g_Kr", 2), ("g_Ks", 3),\
        ("g_Na", 4), ("g_bna", 5), ("g_CaL", 6), ("g_bca", 7), ("g_to", 8),\
        ("K_mNa", 9), ("K_mk", 10), ("P_NaK", 11), ("K_NaCa", 12), ("K_sat",\
        13), ("Km_Ca", 14), ("Km_Nai", 15), ("alpha", 16), ("gamma", 17),\
        ("K_pCa", 18), ("g_pCa", 19), ("g_pK", 20), ("Buf_c", 21), ("Buf_sr",\
        22), ("Buf_ss", 23), ("Ca_o", 24), ("EC", 25), ("K_buf_c", 26),\
        ("K_buf_sr", 27), ("K_buf_ss", 28), ("K_up", 29), ("V_leak", 30),\
        ("V_rel", 31), ("V_sr", 32), ("V_ss", 33), ("V_xfer", 34),\
        ("Vmax_up", 35), ("k1_prime", 36), ("k2_prime", 37), ("k3", 38),\
        ("k4", 39), ("max_sr", 40), ("min_sr", 41), ("Na_o", 42), ("Cm", 43),\
        ("F", 44), ("R", 45), ("T", 46), ("V_c", 47), ("stim_amplitude", 48),\
        ("stim_duration", 49), ("stim_period", 50), ("stim_start", 51),\
        ("K_o", 52)])

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
    state_inds = dict([("Xr1", 0), ("Xr2", 1), ("Xs", 2), ("m", 3), ("h", 4),\
        ("j", 5), ("d", 6), ("f", 7), ("f2", 8), ("fCass", 9), ("s", 10),\
        ("r", 11), ("Ca_i", 12), ("R_prime", 13), ("Ca_SR", 14), ("Ca_ss",\
        15), ("Na_i", 16), ("V", 17), ("K_i", 18)])

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
    param_inds = dict([("P_kna", 0), ("g_K1", 1), ("g_Kr", 2), ("g_Ks", 3),\
        ("g_Na", 4), ("g_bna", 5), ("g_CaL", 6), ("g_bca", 7), ("g_to", 8),\
        ("K_mNa", 9), ("K_mk", 10), ("P_NaK", 11), ("K_NaCa", 12), ("K_sat",\
        13), ("Km_Ca", 14), ("Km_Nai", 15), ("alpha", 16), ("gamma", 17),\
        ("K_pCa", 18), ("g_pCa", 19), ("g_pK", 20), ("Buf_c", 21), ("Buf_sr",\
        22), ("Buf_ss", 23), ("Ca_o", 24), ("EC", 25), ("K_buf_c", 26),\
        ("K_buf_sr", 27), ("K_buf_ss", 28), ("K_up", 29), ("V_leak", 30),\
        ("V_rel", 31), ("V_sr", 32), ("V_ss", 33), ("V_xfer", 34),\
        ("Vmax_up", 35), ("k1_prime", 36), ("k2_prime", 37), ("k3", 38),\
        ("k4", 39), ("max_sr", 40), ("min_sr", 41), ("Na_o", 42), ("Cm", 43),\
        ("F", 44), ("R", 45), ("T", 46), ("V_c", 47), ("stim_amplitude", 48),\
        ("stim_duration", 49), ("stim_period", 50), ("stim_start", 51),\
        ("K_o", 52)])

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
    monitor_inds = dict([("E_Na", 0), ("E_K", 1), ("E_Ks", 2), ("E_Ca", 3),\
        ("alpha_K1", 4), ("beta_K1", 5), ("xK1_inf", 6), ("i_K1", 7),\
        ("i_Kr", 8), ("xr1_inf", 9), ("alpha_xr1", 10), ("beta_xr1", 11),\
        ("tau_xr1", 12), ("xr2_inf", 13), ("alpha_xr2", 14), ("beta_xr2",\
        15), ("tau_xr2", 16), ("i_Ks", 17), ("xs_inf", 18), ("alpha_xs", 19),\
        ("beta_xs", 20), ("tau_xs", 21), ("i_Na", 22), ("m_inf", 23),\
        ("alpha_m", 24), ("beta_m", 25), ("tau_m", 26), ("h_inf", 27),\
        ("alpha_h", 28), ("beta_h", 29), ("tau_h", 30), ("j_inf", 31),\
        ("alpha_j", 32), ("beta_j", 33), ("tau_j", 34), ("i_b_Na", 35),\
        ("V_eff", 36), ("i_CaL", 37), ("d_inf", 38), ("alpha_d", 39),\
        ("beta_d", 40), ("gamma_d", 41), ("tau_d", 42), ("f_inf", 43),\
        ("tau_f", 44), ("f2_inf", 45), ("tau_f2", 46), ("fCass_inf", 47),\
        ("tau_fCass", 48), ("i_b_Ca", 49), ("i_to", 50), ("s_inf", 51),\
        ("tau_s", 52), ("r_inf", 53), ("tau_r", 54), ("i_NaK", 55),\
        ("i_NaCa", 56), ("i_p_Ca", 57), ("i_p_K", 58), ("i_up", 59),\
        ("i_leak", 60), ("i_xfer", 61), ("kcasr", 62), ("Ca_i_bufc", 63),\
        ("Ca_sr_bufsr", 64), ("Ca_ss_bufss", 65), ("k1", 66), ("k2", 67),\
        ("O", 68), ("i_rel", 69), ("i_Stim", 70), ("dXr1_dt", 71),\
        ("dXr2_dt", 72), ("dXs_dt", 73), ("dm_dt", 74), ("dh_dt", 75),\
        ("dj_dt", 76), ("dd_dt", 77), ("df_dt", 78), ("df2_dt", 79),\
        ("dfCass_dt", 80), ("ds_dt", 81), ("dr_dt", 82), ("dCa_i_dt", 83),\
        ("dR_prime_dt", 84), ("dCa_SR_dt", 85), ("dCa_ss_dt", 86),\
        ("dNa_i_dt", 87), ("dV_dt", 88), ("dK_i_dt", 89)])

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
    Compute the right hand side of the tentusscher_panfilov_2006_M_cell ODE
    """

    # Assign states
    assert(len(states) == 19)
    Xr1, Xr2, Xs, m, h, j, d, f, f2, fCass, s, r, Ca_i, R_prime, Ca_SR,\
        Ca_ss, Na_i, V, K_i = states

    # Assign parameters
    assert(len(parameters) == 53)
    P_kna, g_K1, g_Kr, g_Ks, g_Na, g_bna, g_CaL, g_bca, g_to, K_mNa, K_mk,\
        P_NaK, K_NaCa, K_sat, Km_Ca, Km_Nai, alpha, gamma, K_pCa, g_pCa,\
        g_pK, Buf_c, Buf_sr, Buf_ss, Ca_o, EC, K_buf_c, K_buf_sr, K_buf_ss,\
        K_up, V_leak, V_rel, V_sr, V_ss, V_xfer, Vmax_up, k1_prime, k2_prime,\
        k3, k4, max_sr, min_sr, Na_o, Cm, F, R, T, V_c, stim_amplitude,\
        stim_duration, stim_period, stim_start, K_o = parameters

    # Init return args
    if values is None:
        values = np.zeros((19,), dtype=np.float_)
    else:
        assert isinstance(values, np.ndarray) and values.shape == (19,)

    # Expressions for the Reversal potentials component
    E_Na = R*T*math.log(Na_o/Na_i)/F
    E_K = R*T*math.log(K_o/K_i)/F
    E_Ks = R*T*math.log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F
    E_Ca = 0.5*R*T*math.log(Ca_o/Ca_i)/F

    # Expressions for the Inward rectifier potassium current component
    alpha_K1 = 0.1/(1 + 6.14421235332821e-06*math.exp(0.06*V - 0.06*E_K))
    beta_K1 = (0.36787944117144233*math.exp(0.1*V - 0.1*E_K) +\
        3.0606040200802673*math.exp(0.0002*V - 0.0002*E_K))/(1 +\
        math.exp(0.5*E_K - 0.5*V))
    xK1_inf = alpha_K1/(alpha_K1 + beta_K1)
    i_K1 = 0.4303314829119352*g_K1*math.sqrt(K_o)*(-E_K + V)*xK1_inf

    # Expressions for the Rapid time dependent potassium current component
    i_Kr = 0.4303314829119352*g_Kr*math.sqrt(K_o)*(-E_K + V)*Xr1*Xr2

    # Expressions for the Xr1 gate component
    xr1_inf = 1.0/(1 + math.exp(-26/7 - V/7))
    alpha_xr1 = 450/(1 + math.exp(-9/2 - V/10))
    beta_xr1 = 6/(1 + 13.581324522578193*math.exp(0.08695652173913043*V))
    tau_xr1 = alpha_xr1*beta_xr1
    values[0] = (-Xr1 + xr1_inf)/tau_xr1

    # Expressions for the Xr2 gate component
    xr2_inf = 1.0/(1 + math.exp(11/3 + V/24))
    alpha_xr2 = 3/(1 + math.exp(-3 - V/20))
    beta_xr2 = 1.12/(1 + math.exp(-3 + V/20))
    tau_xr2 = alpha_xr2*beta_xr2
    values[1] = (-Xr2 + xr2_inf)/tau_xr2

    # Expressions for the Slow time dependent potassium current component
    i_Ks = g_Ks*(Xs*Xs)*(-E_Ks + V)

    # Expressions for the Xs gate component
    xs_inf = 1.0/(1 + math.exp(-5/14 - V/14))
    alpha_xs = 1400/math.sqrt(1 + math.exp(5/6 - V/6))
    beta_xs = 1.0/(1 + math.exp(-7/3 + V/15))
    tau_xs = 80 + alpha_xs*beta_xs
    values[2] = (-Xs + xs_inf)/tau_xs

    # Expressions for the Fast sodium current component
    i_Na = g_Na*(m*m*m)*(-E_Na + V)*h*j

    # Expressions for the m gate component
    m_inf = 1.0/((1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V))*(1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V)))
    alpha_m = 1.0/(1 + math.exp(-12 - V/5))
    beta_m = 0.1/(1 + math.exp(7 + V/5)) + 0.1/(1 + math.exp(-1/4 + V/200))
    tau_m = alpha_m*beta_m
    values[3] = (-m + m_inf)/tau_m

    # Expressions for the h gate component
    h_inf = 1.0/((1 + 15212.593285654404*math.exp(0.13458950201884254*V))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*V)))
    alpha_h = (4.4312679295805147e-07*math.exp(-0.14705882352941177*V) if V <\
        -40 else 0)
    beta_h = (310000*math.exp(0.3485*V) + 2.7*math.exp(0.079*V) if V < -40 else\
        0.77/(0.13 + 0.049758141083938695*math.exp(-0.0900900900900901*V)))
    tau_h = 1.0/(alpha_h + beta_h)
    values[4] = (-h + h_inf)/tau_h

    # Expressions for the j gate component
    j_inf = 1.0/((1 + 15212.593285654404*math.exp(0.13458950201884254*V))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*V)))
    alpha_j = ((37.78 + V)*(-25428*math.exp(0.2444*V) -\
        6.948e-06*math.exp(-0.04391*V))/(1 +\
        50262745825.95399*math.exp(0.311*V)) if V < -40 else 0)
    beta_j = (0.02424*math.exp(-0.01052*V)/(1 +\
        0.003960868339904256*math.exp(-0.1378*V)) if V < -40 else\
        0.6*math.exp(0.057*V)/(1 + 0.040762203978366204*math.exp(-0.1*V)))
    tau_j = 1.0/(alpha_j + beta_j)
    values[5] = (-j + j_inf)/tau_j

    # Expressions for the Sodium background current component
    i_b_Na = g_bna*(-E_Na + V)

    # Expressions for the L_type Ca current component
    V_eff = (0.01 if math.fabs(-15 + V) < 0.01 else -15 + V)
    i_CaL = 4*g_CaL*(F*F)*(-Ca_o +\
        0.25*Ca_ss*math.exp(2*F*V_eff/(R*T)))*V_eff*d*f*f2*fCass/(R*T*(-1 +\
        math.exp(2*F*V_eff/(R*T))))

    # Expressions for the d gate component
    d_inf = 1.0/(1 + 0.34415378686541237*math.exp(-0.13333333333333333*V))
    alpha_d = 0.25 + 1.4/(1 + math.exp(-35/13 - V/13))
    beta_d = 1.4/(1 + math.exp(1 + V/5))
    gamma_d = 1.0/(1 + math.exp(5/2 - V/20))
    tau_d = alpha_d*beta_d + gamma_d
    values[6] = (-d + d_inf)/tau_d

    # Expressions for the f gate component
    f_inf = 1.0/(1 + math.exp(20/7 + V/7))
    tau_f = 20 + 180/(1 + math.exp(3 + V/10)) + 200/(1 + math.exp(13/10 -\
        V/10)) + 1102.5*math.exp(-((27 + V)*(27 + V))/225)
    values[7] = (-f + f_inf)/tau_f

    # Expressions for the F2 gate component
    f2_inf = 0.33 + 0.67/(1 + math.exp(5 + V/7))
    tau_f2 = 31/(1 + math.exp(5/2 - V/10)) + 80/(1 + math.exp(3 + V/10)) +\
        562*math.exp(-((27 + V)*(27 + V))/240)
    values[8] = (-f2 + f2_inf)/tau_f2

    # Expressions for the FCass gate component
    fCass_inf = 0.4 + 0.6/(1 + 400.0*(Ca_ss*Ca_ss))
    tau_fCass = 2 + 80/(1 + 400.0*(Ca_ss*Ca_ss))
    values[9] = (-fCass + fCass_inf)/tau_fCass

    # Expressions for the Calcium background current component
    i_b_Ca = g_bca*(-E_Ca + V)

    # Expressions for the Transient outward current component
    i_to = g_to*(-E_K + V)*r*s

    # Expressions for the s gate component
    s_inf = 1.0/(1 + math.exp(4 + V/5))
    tau_s = 3 + 5/(1 + math.exp(-4 + V/5)) + 85*math.exp(-((45 + V)*(45 +\
        V))/320)
    values[10] = (-s + s_inf)/tau_s

    # Expressions for the r gate component
    r_inf = 1.0/(1 + math.exp(10/3 - V/6))
    tau_r = 0.8 + 9.5*math.exp(-((40 + V)*(40 + V))/1800)
    values[11] = (-r + r_inf)/tau_r

    # Expressions for the Sodium potassium pump current component
    i_NaK = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1 +\
        0.0353*math.exp(-F*V/(R*T)) + 0.1245*math.exp(-0.1*F*V/(R*T))))

    # Expressions for the Sodium calcium exchanger current component
    i_NaCa = K_NaCa*(Ca_o*(Na_i*Na_i*Na_i)*math.exp(F*gamma*V/(R*T)) -\
        alpha*(Na_o*Na_o*Na_o)*Ca_i*math.exp(F*(-1 + gamma)*V/(R*T)))/((1 +\
        K_sat*math.exp(F*(-1 + gamma)*V/(R*T)))*(Ca_o +\
        Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)))

    # Expressions for the Calcium pump current component
    i_p_Ca = g_pCa*Ca_i/(K_pCa + Ca_i)

    # Expressions for the Potassium pump current component
    i_p_K = g_pK*(-E_K + V)/(1 +\
        65.40521574193832*math.exp(-0.16722408026755853*V))

    # Expressions for the Calcium dynamics component
    i_up = Vmax_up/(1 + (K_up*K_up)/(Ca_i*Ca_i))
    i_leak = V_leak*(-Ca_i + Ca_SR)
    i_xfer = V_xfer*(-Ca_i + Ca_ss)
    kcasr = max_sr - (max_sr - min_sr)/(1 + (EC*EC)/(Ca_SR*Ca_SR))
    Ca_i_bufc = 1.0/(1 + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))
    Ca_sr_bufsr = 1.0/(1 + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr +\
        Ca_SR)))
    Ca_ss_bufss = 1.0/(1 + Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss +\
        Ca_ss)))
    values[12] = (V_sr*(-i_up + i_leak)/V_c - Cm*(-2*i_NaCa + i_b_Ca +\
        i_p_Ca)/(2*F*V_c) + i_xfer)*Ca_i_bufc
    k1 = k1_prime/kcasr
    k2 = k2_prime*kcasr
    O = (Ca_ss*Ca_ss)*R_prime*k1/(k3 + (Ca_ss*Ca_ss)*k1)
    values[13] = k4*(1 - R_prime) - Ca_ss*R_prime*k2
    i_rel = V_rel*(-Ca_ss + Ca_SR)*O
    values[14] = (-i_leak - i_rel + i_up)*Ca_sr_bufsr
    values[15] = (V_sr*i_rel/V_ss - V_c*i_xfer/V_ss -\
        Cm*i_CaL/(2*F*V_ss))*Ca_ss_bufss

    # Expressions for the Sodium dynamics component
    values[16] = Cm*(-i_Na - i_b_Na - 3*i_NaCa - 3*i_NaK)/(F*V_c)

    # Expressions for the Membrane component
    i_Stim = (-stim_amplitude if t - stim_period*math.floor(t/stim_period) <=\
        stim_duration + stim_start and t -\
        stim_period*math.floor(t/stim_period) >= stim_start else 0)
    values[17] = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim\
        - i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to

    # Expressions for the Potassium dynamics component
    values[18] = Cm*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to +\
        2*i_NaK)/(F*V_c)

    # Return results
    return values

def monitor(states, t, parameters, monitored=None):
    """
    Computes monitored expressions of the tentusscher_panfilov_2006_M_cell ODE
    """

    # Assign states
    assert(len(states) == 19)
    Xr1, Xr2, Xs, m, h, j, d, f, f2, fCass, s, r, Ca_i, R_prime, Ca_SR,\
        Ca_ss, Na_i, V, K_i = states

    # Assign parameters
    assert(len(parameters) == 53)
    P_kna, g_K1, g_Kr, g_Ks, g_Na, g_bna, g_CaL, g_bca, g_to, K_mNa, K_mk,\
        P_NaK, K_NaCa, K_sat, Km_Ca, Km_Nai, alpha, gamma, K_pCa, g_pCa,\
        g_pK, Buf_c, Buf_sr, Buf_ss, Ca_o, EC, K_buf_c, K_buf_sr, K_buf_ss,\
        K_up, V_leak, V_rel, V_sr, V_ss, V_xfer, Vmax_up, k1_prime, k2_prime,\
        k3, k4, max_sr, min_sr, Na_o, Cm, F, R, T, V_c, stim_amplitude,\
        stim_duration, stim_period, stim_start, K_o = parameters

    # Init return args
    if monitored is None:
        monitored = np.zeros((90,), dtype=np.float_)
    else:
        assert isinstance(monitored, np.ndarray) and monitored.shape == (90,)

    # Expressions for the Reversal potentials component
    monitored[0] = R*T*math.log(Na_o/Na_i)/F
    monitored[1] = R*T*math.log(K_o/K_i)/F
    monitored[2] = R*T*math.log((K_o + Na_o*P_kna)/(P_kna*Na_i + K_i))/F
    monitored[3] = 0.5*R*T*math.log(Ca_o/Ca_i)/F

    # Expressions for the Inward rectifier potassium current component
    monitored[4] = 0.1/(1 + 6.14421235332821e-06*math.exp(0.06*V -\
        0.06*monitored[1]))
    monitored[5] = (0.36787944117144233*math.exp(0.1*V - 0.1*monitored[1]) +\
        3.0606040200802673*math.exp(0.0002*V - 0.0002*monitored[1]))/(1 +\
        math.exp(0.5*monitored[1] - 0.5*V))
    monitored[6] = monitored[4]/(monitored[4] + monitored[5])
    monitored[7] = 0.4303314829119352*g_K1*math.sqrt(K_o)*(-monitored[1] +\
        V)*monitored[6]

    # Expressions for the Rapid time dependent potassium current component
    monitored[8] = 0.4303314829119352*g_Kr*math.sqrt(K_o)*(-monitored[1] +\
        V)*Xr1*Xr2

    # Expressions for the Xr1 gate component
    monitored[9] = 1.0/(1 + math.exp(-26/7 - V/7))
    monitored[10] = 450/(1 + math.exp(-9/2 - V/10))
    monitored[11] = 6/(1 + 13.581324522578193*math.exp(0.08695652173913043*V))
    monitored[12] = monitored[10]*monitored[11]
    monitored[71] = (-Xr1 + monitored[9])/monitored[12]

    # Expressions for the Xr2 gate component
    monitored[13] = 1.0/(1 + math.exp(11/3 + V/24))
    monitored[14] = 3/(1 + math.exp(-3 - V/20))
    monitored[15] = 1.12/(1 + math.exp(-3 + V/20))
    monitored[16] = monitored[14]*monitored[15]
    monitored[72] = (-Xr2 + monitored[13])/monitored[16]

    # Expressions for the Slow time dependent potassium current component
    monitored[17] = g_Ks*(Xs*Xs)*(-monitored[2] + V)

    # Expressions for the Xs gate component
    monitored[18] = 1.0/(1 + math.exp(-5/14 - V/14))
    monitored[19] = 1400/math.sqrt(1 + math.exp(5/6 - V/6))
    monitored[20] = 1.0/(1 + math.exp(-7/3 + V/15))
    monitored[21] = 80 + monitored[19]*monitored[20]
    monitored[73] = (-Xs + monitored[18])/monitored[21]

    # Expressions for the Fast sodium current component
    monitored[22] = g_Na*(m*m*m)*(-monitored[0] + V)*h*j

    # Expressions for the m gate component
    monitored[23] = 1.0/((1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V))*(1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*V)))
    monitored[24] = 1.0/(1 + math.exp(-12 - V/5))
    monitored[25] = 0.1/(1 + math.exp(7 + V/5)) + 0.1/(1 + math.exp(-1/4 +\
        V/200))
    monitored[26] = monitored[24]*monitored[25]
    monitored[74] = (-m + monitored[23])/monitored[26]

    # Expressions for the h gate component
    monitored[27] = 1.0/((1 +\
        15212.593285654404*math.exp(0.13458950201884254*V))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*V)))
    monitored[28] = (4.4312679295805147e-07*math.exp(-0.14705882352941177*V)\
        if V < -40 else 0)
    monitored[29] = (310000*math.exp(0.3485*V) + 2.7*math.exp(0.079*V) if V <\
        -40 else 0.77/(0.13 +\
        0.049758141083938695*math.exp(-0.0900900900900901*V)))
    monitored[30] = 1.0/(monitored[28] + monitored[29])
    monitored[75] = (-h + monitored[27])/monitored[30]

    # Expressions for the j gate component
    monitored[31] = 1.0/((1 +\
        15212.593285654404*math.exp(0.13458950201884254*V))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*V)))
    monitored[32] = ((37.78 + V)*(-25428*math.exp(0.2444*V) -\
        6.948e-06*math.exp(-0.04391*V))/(1 +\
        50262745825.95399*math.exp(0.311*V)) if V < -40 else 0)
    monitored[33] = (0.02424*math.exp(-0.01052*V)/(1 +\
        0.003960868339904256*math.exp(-0.1378*V)) if V < -40 else\
        0.6*math.exp(0.057*V)/(1 + 0.040762203978366204*math.exp(-0.1*V)))
    monitored[34] = 1.0/(monitored[32] + monitored[33])
    monitored[76] = (-j + monitored[31])/monitored[34]

    # Expressions for the Sodium background current component
    monitored[35] = g_bna*(-monitored[0] + V)

    # Expressions for the L_type Ca current component
    monitored[36] = (0.01 if math.fabs(-15 + V) < 0.01 else -15 + V)
    monitored[37] = 4*g_CaL*(F*F)*(-Ca_o +\
        0.25*Ca_ss*math.exp(2*F*monitored[36]/(R*T)))*d*f*f2*fCass*monitored[36]/(R*T*(-1 +\
        math.exp(2*F*monitored[36]/(R*T))))

    # Expressions for the d gate component
    monitored[38] = 1.0/(1 +\
        0.34415378686541237*math.exp(-0.13333333333333333*V))
    monitored[39] = 0.25 + 1.4/(1 + math.exp(-35/13 - V/13))
    monitored[40] = 1.4/(1 + math.exp(1 + V/5))
    monitored[41] = 1.0/(1 + math.exp(5/2 - V/20))
    monitored[42] = monitored[39]*monitored[40] + monitored[41]
    monitored[77] = (-d + monitored[38])/monitored[42]

    # Expressions for the f gate component
    monitored[43] = 1.0/(1 + math.exp(20/7 + V/7))
    monitored[44] = 20 + 180/(1 + math.exp(3 + V/10)) + 200/(1 +\
        math.exp(13/10 - V/10)) + 1102.5*math.exp(-((27 + V)*(27 + V))/225)
    monitored[78] = (-f + monitored[43])/monitored[44]

    # Expressions for the F2 gate component
    monitored[45] = 0.33 + 0.67/(1 + math.exp(5 + V/7))
    monitored[46] = 31/(1 + math.exp(5/2 - V/10)) + 80/(1 + math.exp(3 +\
        V/10)) + 562*math.exp(-((27 + V)*(27 + V))/240)
    monitored[79] = (-f2 + monitored[45])/monitored[46]

    # Expressions for the FCass gate component
    monitored[47] = 0.4 + 0.6/(1 + 400.0*(Ca_ss*Ca_ss))
    monitored[48] = 2 + 80/(1 + 400.0*(Ca_ss*Ca_ss))
    monitored[80] = (-fCass + monitored[47])/monitored[48]

    # Expressions for the Calcium background current component
    monitored[49] = g_bca*(-monitored[3] + V)

    # Expressions for the Transient outward current component
    monitored[50] = g_to*(-monitored[1] + V)*r*s

    # Expressions for the s gate component
    monitored[51] = 1.0/(1 + math.exp(4 + V/5))
    monitored[52] = 3 + 5/(1 + math.exp(-4 + V/5)) + 85*math.exp(-((45 +\
        V)*(45 + V))/320)
    monitored[81] = (-s + monitored[51])/monitored[52]

    # Expressions for the r gate component
    monitored[53] = 1.0/(1 + math.exp(10/3 - V/6))
    monitored[54] = 0.8 + 9.5*math.exp(-((40 + V)*(40 + V))/1800)
    monitored[82] = (-r + monitored[53])/monitored[54]

    # Expressions for the Sodium potassium pump current component
    monitored[55] = K_o*P_NaK*Na_i/((K_mNa + Na_i)*(K_mk + K_o)*(1 +\
        0.0353*math.exp(-F*V/(R*T)) + 0.1245*math.exp(-0.1*F*V/(R*T))))

    # Expressions for the Sodium calcium exchanger current component
    monitored[56] = K_NaCa*(Ca_o*(Na_i*Na_i*Na_i)*math.exp(F*gamma*V/(R*T)) -\
        alpha*(Na_o*Na_o*Na_o)*Ca_i*math.exp(F*(-1 + gamma)*V/(R*T)))/((1 +\
        K_sat*math.exp(F*(-1 + gamma)*V/(R*T)))*(Ca_o +\
        Km_Ca)*((Km_Nai*Km_Nai*Km_Nai) + (Na_o*Na_o*Na_o)))

    # Expressions for the Calcium pump current component
    monitored[57] = g_pCa*Ca_i/(K_pCa + Ca_i)

    # Expressions for the Potassium pump current component
    monitored[58] = g_pK*(-monitored[1] + V)/(1 +\
        65.40521574193832*math.exp(-0.16722408026755853*V))

    # Expressions for the Calcium dynamics component
    monitored[59] = Vmax_up/(1 + (K_up*K_up)/(Ca_i*Ca_i))
    monitored[60] = V_leak*(-Ca_i + Ca_SR)
    monitored[61] = V_xfer*(-Ca_i + Ca_ss)
    monitored[62] = max_sr - (max_sr - min_sr)/(1 + (EC*EC)/(Ca_SR*Ca_SR))
    monitored[63] = 1.0/(1 + Buf_c*K_buf_c/((K_buf_c + Ca_i)*(K_buf_c + Ca_i)))
    monitored[64] = 1.0/(1 + Buf_sr*K_buf_sr/((K_buf_sr + Ca_SR)*(K_buf_sr +\
        Ca_SR)))
    monitored[65] = 1.0/(1 + Buf_ss*K_buf_ss/((K_buf_ss + Ca_ss)*(K_buf_ss +\
        Ca_ss)))
    monitored[83] = (V_sr*(-monitored[59] + monitored[60])/V_c -\
        Cm*(-2*monitored[56] + monitored[49] + monitored[57])/(2*F*V_c) +\
        monitored[61])*monitored[63]
    monitored[66] = k1_prime/monitored[62]
    monitored[67] = k2_prime*monitored[62]
    monitored[68] = (Ca_ss*Ca_ss)*R_prime*monitored[66]/(k3 +\
        (Ca_ss*Ca_ss)*monitored[66])
    monitored[84] = k4*(1 - R_prime) - Ca_ss*R_prime*monitored[67]
    monitored[69] = V_rel*(-Ca_ss + Ca_SR)*monitored[68]
    monitored[85] = (-monitored[60] - monitored[69] +\
        monitored[59])*monitored[64]
    monitored[86] = (V_sr*monitored[69]/V_ss - V_c*monitored[61]/V_ss -\
        Cm*monitored[37]/(2*F*V_ss))*monitored[65]

    # Expressions for the Sodium dynamics component
    monitored[87] = Cm*(-monitored[22] - monitored[35] - 3*monitored[55] -\
        3*monitored[56])/(F*V_c)

    # Expressions for the Membrane component
    monitored[70] = (-stim_amplitude if t -\
        stim_period*math.floor(t/stim_period) <= stim_duration + stim_start\
        and t - stim_period*math.floor(t/stim_period) >= stim_start else 0)
    monitored[88] = -monitored[17] - monitored[22] - monitored[35] -\
        monitored[37] - monitored[49] - monitored[50] - monitored[55] -\
        monitored[56] - monitored[57] - monitored[58] - monitored[70] -\
        monitored[7] - monitored[8]

    # Expressions for the Potassium dynamics component
    monitored[89] = Cm*(-monitored[17] - monitored[50] - monitored[58] -\
        monitored[70] - monitored[7] - monitored[8] +\
        2*monitored[55])/(F*V_c)

    # Return results
    return monitored

# Numba ---
from numbalsoda import lsoda_sig
from numba import njit, cfunc, jit
import numpy as np
import timeit
import math

import tentusscher_panfilov_2006_M_cell as model


#@cfunc(lsoda_sig)
@cfunc(lsoda_sig, nopython=True) 
def rhs_numba(t, states, values, parameters):
    """
    Compute the right hand side of the tentusscher_panfilov_2006_M_cell ODE
    """
    # Expressions for the Reversal potentials component
    E_Na =\
        parameters[45]*parameters[46]*math.log(parameters[42]/states[16])/parameters[44]
    E_K =\
        parameters[45]*parameters[46]*math.log(parameters[52]/states[18])/parameters[44]
    E_Ks = parameters[45]*parameters[46]*math.log((parameters[52] +\
        parameters[0]*parameters[42])/(states[18] +\
        parameters[0]*states[16]))/parameters[44]
    E_Ca =\
        0.5*parameters[45]*parameters[46]*math.log(parameters[24]/states[12])/parameters[44]

    # Expressions for the Inward rectifier potassium current component
    alpha_K1 = 0.1/(1 + 6.14421235332821e-06*math.exp(0.06*states[17] -\
        0.06*E_K))
    beta_K1 = (0.36787944117144233*math.exp(0.1*states[17] - 0.1*E_K) +\
        3.0606040200802673*math.exp(0.0002*states[17] - 0.0002*E_K))/(1 +\
        math.exp(0.5*E_K - 0.5*states[17]))
    xK1_inf = alpha_K1/(alpha_K1 + beta_K1)
    i_K1 =\
        0.4303314829119352*parameters[1]*math.sqrt(parameters[52])*(states[17]\
        - E_K)*xK1_inf

    # Expressions for the Rapid time dependent potassium current component
    i_Kr =\
        0.4303314829119352*parameters[2]*states[0]*states[1]*math.sqrt(parameters[52])*(states[17]\
        - E_K)

    # Expressions for the Xr1 gate component
    xr1_inf = 1.0/(1 + math.exp(-26/7 - states[17]/7))
    alpha_xr1 = 450/(1 + math.exp(-9/2 - states[17]/10))
    beta_xr1 = 6/(1 +\
        13.581324522578193*math.exp(0.08695652173913043*states[17]))
    tau_xr1 = alpha_xr1*beta_xr1
    values[0] = (-states[0] + xr1_inf)/tau_xr1

    # Expressions for the Xr2 gate component
    xr2_inf = 1.0/(1 + math.exp(11/3 + states[17]/24))
    alpha_xr2 = 3/(1 + math.exp(-3 - states[17]/20))
    beta_xr2 = 1.12/(1 + math.exp(-3 + states[17]/20))
    tau_xr2 = alpha_xr2*beta_xr2
    values[1] = (-states[1] + xr2_inf)/tau_xr2

    # Expressions for the Slow time dependent potassium current component
    i_Ks = parameters[3]*(states[2]*states[2])*(states[17] - E_Ks)

    # Expressions for the Xs gate component
    xs_inf = 1.0/(1 + math.exp(-5/14 - states[17]/14))
    alpha_xs = 1400/math.sqrt(1 + math.exp(5/6 - states[17]/6))
    beta_xs = 1.0/(1 + math.exp(-7/3 + states[17]/15))
    tau_xs = 80 + alpha_xs*beta_xs
    values[2] = (-states[2] + xs_inf)/tau_xs

    # Expressions for the Fast sodium current component
    i_Na =\
        parameters[4]*states[4]*states[5]*(states[3]*states[3]*states[3])*(states[17]\
        - E_Na)

    # Expressions for the m gate component
    m_inf = 1.0/((1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*states[17]))*(1 +\
        0.0018422115811651339*math.exp(-0.1107419712070875*states[17])))
    alpha_m = 1.0/(1 + math.exp(-12 - states[17]/5))
    beta_m = 0.1/(1 + math.exp(7 + states[17]/5)) + 0.1/(1 + math.exp(-1/4 +\
        states[17]/200))
    tau_m = alpha_m*beta_m
    values[3] = (-states[3] + m_inf)/tau_m

    # Expressions for the h gate component
    h_inf = 1.0/((1 +\
        15212.593285654404*math.exp(0.13458950201884254*states[17]))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*states[17])))
    alpha_h =\
        (4.4312679295805147e-07*math.exp(-0.14705882352941177*states[17]) if\
        states[17] < -40 else 0)
    beta_h = (310000*math.exp(0.3485*states[17]) +\
        2.7*math.exp(0.079*states[17]) if states[17] < -40 else 0.77/(0.13 +\
        0.049758141083938695*math.exp(-0.0900900900900901*states[17])))
    tau_h = 1.0/(alpha_h + beta_h)
    values[4] = (-states[4] + h_inf)/tau_h

    # Expressions for the j gate component
    j_inf = 1.0/((1 +\
        15212.593285654404*math.exp(0.13458950201884254*states[17]))*(1 +\
        15212.593285654404*math.exp(0.13458950201884254*states[17])))
    alpha_j = ((37.78 + states[17])*(-25428*math.exp(0.2444*states[17]) -\
        6.948e-06*math.exp(-0.04391*states[17]))/(1 +\
        50262745825.95399*math.exp(0.311*states[17])) if states[17] < -40 else\
        0)
    beta_j = (0.02424*math.exp(-0.01052*states[17])/(1 +\
        0.003960868339904256*math.exp(-0.1378*states[17])) if states[17] <\
        -40 else 0.6*math.exp(0.057*states[17])/(1 +\
        0.040762203978366204*math.exp(-0.1*states[17])))
    tau_j = 1.0/(alpha_j + beta_j)
    values[5] = (-states[5] + j_inf)/tau_j

    # Expressions for the Sodium background current component
    i_b_Na = parameters[5]*(states[17] - E_Na)

    # Expressions for the L_type Ca current component
    V_eff = (0.01 if math.fabs(-15 + states[17]) < 0.01 else -15 + states[17])
    i_CaL =\
        4*parameters[6]*states[6]*states[7]*states[8]*states[9]*(parameters[44]*parameters[44])*(-parameters[24]\
        +\
        0.25*states[15]*math.exp(2*parameters[44]*V_eff/(parameters[45]*parameters[46])))*V_eff/(parameters[45]*parameters[46]*(-1 +\
        math.exp(2*parameters[44]*V_eff/(parameters[45]*parameters[46]))))

    # Expressions for the d gate component
    d_inf = 1.0/(1 +\
        0.34415378686541237*math.exp(-0.13333333333333333*states[17]))
    alpha_d = 0.25 + 1.4/(1 + math.exp(-35/13 - states[17]/13))
    beta_d = 1.4/(1 + math.exp(1 + states[17]/5))
    gamma_d = 1.0/(1 + math.exp(5/2 - states[17]/20))
    tau_d = alpha_d*beta_d + gamma_d
    values[6] = (-states[6] + d_inf)/tau_d

    # Expressions for the f gate component
    f_inf = 1.0/(1 + math.exp(20/7 + states[17]/7))
    tau_f = 20 + 180/(1 + math.exp(3 + states[17]/10)) + 200/(1 +\
        math.exp(13/10 - states[17]/10)) + 1102.5*math.exp(-((27 +\
        states[17])*(27 + states[17]))/225)
    values[7] = (-states[7] + f_inf)/tau_f

    # Expressions for the F2 gate component
    f2_inf = 0.33 + 0.67/(1 + math.exp(5 + states[17]/7))
    tau_f2 = 31/(1 + math.exp(5/2 - states[17]/10)) + 80/(1 + math.exp(3 +\
        states[17]/10)) + 562*math.exp(-((27 + states[17])*(27 +\
        states[17]))/240)
    values[8] = (-states[8] + f2_inf)/tau_f2

    # Expressions for the FCass gate component
    fCass_inf = 0.4 + 0.6/(1 + 400.0*(states[15]*states[15]))
    tau_fCass = 2 + 80/(1 + 400.0*(states[15]*states[15]))
    values[9] = (-states[9] + fCass_inf)/tau_fCass

    # Expressions for the Calcium background current component
    i_b_Ca = parameters[7]*(states[17] - E_Ca)

    # Expressions for the Transient outward current component
    i_to = parameters[8]*states[10]*states[11]*(states[17] - E_K)

    # Expressions for the s gate component
    s_inf = 1.0/(1 + math.exp(4 + states[17]/5))
    tau_s = 3 + 5/(1 + math.exp(-4 + states[17]/5)) + 85*math.exp(-((45 +\
        states[17])*(45 + states[17]))/320)
    values[10] = (-states[10] + s_inf)/tau_s

    # Expressions for the r gate component
    r_inf = 1.0/(1 + math.exp(10/3 - states[17]/6))
    tau_r = 0.8 + 9.5*math.exp(-((40 + states[17])*(40 + states[17]))/1800)
    values[11] = (-states[11] + r_inf)/tau_r

    # Expressions for the Sodium potassium pump current component
    i_NaK = parameters[11]*parameters[52]*states[16]/((parameters[10] +\
        parameters[52])*(parameters[9] + states[16])*(1 +\
        0.0353*math.exp(-parameters[44]*states[17]/(parameters[45]*parameters[46]))\
        +\
        0.1245*math.exp(-0.1*parameters[44]*states[17]/(parameters[45]*parameters[46]))))

    # Expressions for the Sodium calcium exchanger current component
    i_NaCa =\
        parameters[12]*(parameters[24]*(states[16]*states[16]*states[16])*math.exp(parameters[17]*parameters[44]*states[17]/(parameters[45]*parameters[46]))\
        -\
        parameters[16]*states[12]*(parameters[42]*parameters[42]*parameters[42])*math.exp(parameters[44]*states[17]*(-1 +\
        parameters[17])/(parameters[45]*parameters[46])))/((1 +\
        parameters[13]*math.exp(parameters[44]*states[17]*(-1 +\
        parameters[17])/(parameters[45]*parameters[46])))*(parameters[14] +\
        parameters[24])*((parameters[15]*parameters[15]*parameters[15]) +\
        (parameters[42]*parameters[42]*parameters[42])))

    # Expressions for the Calcium pump current component
    i_p_Ca = parameters[19]*states[12]/(parameters[18] + states[12])

    # Expressions for the Potassium pump current component
    i_p_K = parameters[20]*(states[17] - E_K)/(1 +\
        65.40521574193832*math.exp(-0.16722408026755853*states[17]))

    # Expressions for the Calcium dynamics component
    i_up = parameters[35]/(1 +\
        (parameters[29]*parameters[29])/(states[12]*states[12]))
    i_leak = parameters[30]*(states[14] - states[12])
    i_xfer = parameters[34]*(states[15] - states[12])
    kcasr = parameters[40] - (parameters[40] - parameters[41])/(1 +\
        (parameters[25]*parameters[25])/(states[14]*states[14]))
    Ca_i_bufc = 1.0/(1 + parameters[21]*parameters[26]/((parameters[26] +\
        states[12])*(parameters[26] + states[12])))
    Ca_sr_bufsr = 1.0/(1 + parameters[22]*parameters[27]/((parameters[27] +\
        states[14])*(parameters[27] + states[14])))
    Ca_ss_bufss = 1.0/(1 + parameters[23]*parameters[28]/((parameters[28] +\
        states[15])*(parameters[28] + states[15])))
    values[12] = (parameters[32]*(-i_up + i_leak)/parameters[47] -\
        parameters[43]*(-2*i_NaCa + i_b_Ca +\
        i_p_Ca)/(2*parameters[44]*parameters[47]) + i_xfer)*Ca_i_bufc
    k1 = parameters[36]/kcasr
    k2 = parameters[37]*kcasr
    O = states[13]*(states[15]*states[15])*k1/(parameters[38] +\
        (states[15]*states[15])*k1)
    values[13] = parameters[39]*(1 - states[13]) - states[13]*states[15]*k2
    i_rel = parameters[31]*(states[14] - states[15])*O
    values[14] = (-i_leak - i_rel + i_up)*Ca_sr_bufsr
    values[15] = (parameters[32]*i_rel/parameters[33] -\
        parameters[47]*i_xfer/parameters[33] -\
        parameters[43]*i_CaL/(2*parameters[33]*parameters[44]))*Ca_ss_bufss

    # Expressions for the Sodium dynamics component
    values[16] = parameters[43]*(-i_Na - i_b_Na - 3*i_NaCa -\
        3*i_NaK)/(parameters[44]*parameters[47])

    # Expressions for the Membrane component
    i_Stim = (-parameters[48] if t -\
        parameters[50]*math.floor(t/parameters[50]) <= parameters[49] +\
        parameters[51] and t - parameters[50]*math.floor(t/parameters[50]) >=\
        parameters[51] else 0)
    values[17] = -i_CaL - i_K1 - i_Kr - i_Ks - i_Na - i_NaCa - i_NaK - i_Stim\
        - i_b_Ca - i_b_Na - i_p_Ca - i_p_K - i_to

    # Expressions for the Potassium dynamics component
    values[18] = parameters[43]*(-i_K1 - i_Kr - i_Ks - i_Stim - i_p_K - i_to\
        + 2*i_NaK)/(parameters[44]*parameters[47])

