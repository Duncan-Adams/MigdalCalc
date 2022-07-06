import numpy as np
from . import kinematics as kin

import scipy.integrate as integrate
from scipy.special import erf
from scipy.special import spherical_jn
from scipy.interpolate import interp1d as interp1d

def sigma_N(sigma_n, A, m_dm):
    m_n = 0.939*1e9
    m_T = A*m_n
    
    mu_n = kin.mu(m_n, m_dm)
    mu_T = kin.mu(m_T, m_dm)
    
    r = mu_T/mu_n
    
    return (A**2)*(r**2)*sigma_n
    
def dm_maxwell_dist(v):
    vesc = 533.0
    sigmav = 156.0
    ve = 232.0
    # Nesc - normalisation constant
    Nesc = (erf(vesc/(np.sqrt(2.0)*sigmav)) - np.sqrt(2.0/np.pi)*(vesc/sigmav)*np.exp(-vesc**2/(2.0*sigmav**2)))

    aplus = np.minimum((v+ve), v*0.0 + vesc)/(np.sqrt(2)*sigmav)
    aminus = np.minimum((v-ve), v*0.0 + vesc)/(np.sqrt(2)*sigmav)
    
    f = np.exp(-aminus**2) - np.exp(-aplus**2)


    return v*f/(np.sqrt(2*np.pi)*sigmav*ve*Nesc)

def vmin(E_R, deltaE, m_dm, A):
    m_n = 0.939*1e9
    m_T = A*m_n
    mu = kin.mu(m_T, m_dm)
    return 3e5*(m_T*E_R + mu*deltaE)/(mu*np.sqrt(2*m_T*E_R))
    
vmin_range = np.geomspace(1, 1000, 1000)
eta_arr = []
for vm in vmin_range:
    eta = integrate.quad(lambda v: 2*np.pi*v*dm_maxwell_dist(v), vm, np.inf)[0]
    eta_arr.append(eta)
    
eta_dm = interp1d(vmin_range, eta_arr, bounds_error=False, fill_value=0)
    
def helm_factor_SI(q_eV, A):
    #equations from lewin and smith
    r_n = 1.14*A**(1/3) #in fermi
    q_MeV = 1e-6*q_eV
    s = 0.9 #fm
    
    hbarc = 200 #MeV fm
    
    qr = q_MeV*r_n/(hbarc)
    qs = q_MeV*s/hbarc
    
    # ~ f1 = (spherical_jn(1, qr))/qr
    f1 = (np.sin(qr) - qr*np.cos(qr))/(qr)**3
    f2 = np.exp(-0.5*(qs)**2)
    return 3*f1*f2

def dR0_dEnr_dv(rho_dm, m_dm, sigma_n, A):
    m_n = 0.939*1e9
    m_T = A*m_n
    
    mu_T = kin.mu(m_T, m_dm)
    sigma = sigma_N(sigma_n, A, m_dm)
    
    mu_T_keV = 1e-3*mu_T
    mu_T_kg = (1.782*1e-36)*mu_T
    km_to_cm = 1e5
    inv_seconds_to_inv_days = 60*60*24
    prefactor = (1/2)*(rho_dm/m_dm)*(1/mu_T_keV)*(1/mu_T_kg)*km_to_cm*inv_seconds_to_inv_days
    def d2R(E_nr, vm):
        return prefactor*sigma*eta_dm(vm)*(helm_factor_SI(np.sqrt(2*m_T*E_nr), A))**2
        
    return d2R

