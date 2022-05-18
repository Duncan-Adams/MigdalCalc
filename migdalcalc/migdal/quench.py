#quench.py - define nuclear quenching models
import numpy as np
import scipy.interpolate as interp

####Linhard model###
def Lindhard_Factor(E_R, Z, A): #lindhard quenching model, see eq (5) of 1801.10159
    g = lambda x: 3*(x**0.15) + 0.7*(x**0.6) + x

    eps = 11.5*(Z**(-7/3))*E_R
    k = 0.133*(Z**(2/3))*(A**(-1/2))
    
    return k*g(eps)/(1 + k*g(eps))
    
def atomic_ionization_ratio(I, E_e):
    num = 1 - np.exp(-I/E_e)
    den = 3 + np.exp(-I/E_e)

    return num/den

#note this is defined only for xenon currently
def charge_yield(E_R, Z, A): #see equations (11)-(13) of 1801.10159
    Y = Lindhard_Factor((1e-3)*E_R, Z, A)

    ion_ratio = atomic_ionization_ratio(555.57, Y*E_R)

    W_i = 14.94 + 8.35*ion_ratio

    pre = Y/W_i

    t_c = 15
    t_pa = 1.5
    alpha = 3.617
    beta = 1.313

    LogE = np.log((1e-3)*E_R) #log of recoil energy in keV

    acc = (t_pa + alpha*LogE + beta*LogE**2)


    Q_y = pre*(np.exp(-np.log(2)*acc/t_c))

    return Q_y

def W_i_effective(E_R, Y, Z, A):
    W_i_0 = 14.94
    
    E_r_ee = Y(E_R)*E_R
    E_r_ee_eV = E_r_ee*1e3
    
    ion_ratio = atomic_ionization_ratio(555.57, E_r_ee_eV)
    print(ion_ratio)
    W_i_eff = W_i_0 + 8.35*ion_ratio
    
    t_c = 15
    t_pa = 1.5
    alpha = 3.617
    beta = 1.313

    LogE = np.log(E_R) #log of recoil energy in keV

    acc = (t_pa + alpha*LogE + beta*LogE**2)
    
    return W_i_eff*np.exp(np.log(2)*acc/t_c)
#############################################################
###Using some of the emprical functions from 1801.10159
#############################################################
def Y_SI_hi(ER):
    A_SI = 28
    Z_SI = 14
    #convert ER to eV from keV
    ER = 1e3*ER
    
    if(ER < 15):
        return 0
        
    if(ER < 250):
        return 0.18*(1 - np.exp(-1*(ER - 15)/71.3))
        
    return Lindhard_Factor(1e-3*ER, Z_SI, A_SI)

def Y_SI_lo(ER):
    ER = 1e3*ER # convert from keV to eV
    if (ER < 300):
        return 0
    Y = (0.20*ER - 78.37)/(ER)
    
    return (0.20*ER - 78.37)/(ER)


#takes ER in keV
#presnted in https://indico.scc.kit.edu/event/2575/contributions/9684/attachments/4817/7278/Saab_SuperCDMS_Yield_EXCESS2022.pdf
def Y_oscura(ER):
    return 0.026*ER**(0.267)
    



#Sarkis silicon stuff (2001.06503)
# ~ quenching_cdms_fit = None
# ~ quenching_sarkis_si = None

# ~ sarkis_data = np.genfromtxt('../data/Si/quench/sarkis/Si_numeric.csv', delimiter=',')
# ~ E_R_data = list(zip(*sarkis_data))[0]
# ~ qf_data = list(zip(*sarkis_data))[1]
# ~ quenching_sarkis_si = interp.interp1d(E_R_data, qf_data, bounds_error=False, fill_value = 0, kind='linear')

# ~ cdms_data = np.genfromtxt('../data/Si/quench/sarkis/cdms.csv', delimiter=',')
# ~ E_R_data = list(zip(*cdms_data))[0]
# ~ qf_data = list(zip(*cdms_data))[1]
# ~ quenching_cdms_fit = interp.interp1d(E_R_data, qf_data, bounds_error=False, fill_value = 0, kind='linear')

# ~ def Y_SI_sarkis(ER):
    # ~ if(ER > quenching_sarkis_si.x[-1]):
        # ~ return Lindhard_Factor(ER, 14, 28)
    
    # ~ return quenching_sarkis_si(ER)

# ~ def Y_SI_CDMS(ER):
    # ~ if(ER > quenching_cmds_fit.x[-1]):
        # ~ return Lindhard_Factor(ER, 14, 28)
        
    # ~ return quenching_cdms_fit(ER)

