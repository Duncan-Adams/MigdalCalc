# Kinematics.py - kinematical functions used in various migdal calculations
import numpy as np

def mu(m1, m2):
    return m1*m2/(m1 + m2)

def velocity(En, m):
    v2 = 1 - 1/((En/m + 1)**2)

    return np.sqrt(v2)
    
def E0(A, En):
    return (A/(A+1)**2)*En
    
def DeltaE_Max(A, En, c_lab):
    return (A/(A+1))*En*(1 - (1 - c_lab**2)/(A**2))
    
def E_Recoil(DeltaE, c_lab, A, En):
    c = c_lab
    s = (1 - c**2)
    R = ((A+1)/(A))*DeltaE/En

    sq = 1 - s/(A**2) - R
    
    S = np.sqrt(np.abs(sq))


    E_R  = 2*E0(A,En)*(1 + s/A - c*S - 0.5*R)*np.heaviside(sq, 1)

    return E_R

#eqn 94 from ibe 
def E_Recoil_ibe(DeltaE, c, A, En):
    R = DeltaE/En
    z = np.sqrt(1 - R)

    E_R  = E0(A,En)*(2 - R - 2*c*z)

    return 

#jacobian to go from nuclear recoil energy to neutron angle in the CM frame, based on eqn 94 of ibe
def Jac_CM_ibe(DeltaE, c, A, En):
    R = DeltaE/En

    return 2*kin.E0(A, En)*np.sqrt(1 - R)

#jacobian to go from nuclear recoil energy to neutron angle in the lab frame
def Jac_lab(DeltaE, c_lab, A, En):
    c = c_lab
    P = 2*E0(A, En)
    sq = (c**2 - 1)/(A**2) + 1 - ((A+1)/(A))*(DeltaE/En) 

    S = np.sqrt(np.abs(sq))

    J = (P*(c/A + S)**2/S)*np.heaviside(sq,1)

    return J
    
def E_R_elastic(ct_cm, A, En):
    return 2*(1-ct_cm)*E0(A,En)

def q_e2(E_R, A):
    m_e2 = (511000)**2 #squared electron mass in eV
    m_n = 0.9396*1e9 #nuetron mass in eV

    return 2*(m_e2/(A*m_n))*E_R


def k_max(En, omega, angle, A):
    m_n = 0.939*1e9 #nuetron mass in eV
    m_N = A*m_n

    c = np.cos(np.pi*angle/180)

    k1 = np.sqrt(omega**2*m_N/(2*E_Recoil(omega, c, A, En)))
    k2 = np.sqrt(2*m_N*E_Recoil(omega, c, A, En))

    return min(k1, k2)

