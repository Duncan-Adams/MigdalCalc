# Kinematics.py - kinematical functions used in various migdal calculations
import numpy as np

def velocity(En, m):
    v2 = 1 - 1/((En/m + 1)**2)

    return np.sqrt(v2)
    
def E0(A, En):
    return (A/(A+1)**2)*En
    
def DeltaE_Max(A, En):
    return (A/(A+1))*En
    
def E_Recoil(DeltaE, c, A, En):
    R = DeltaE/En
    z = np.sqrt(1 - R)

    E_R  = E0(A,En)*(2 - R - 2*c*z)

    return E_R

def q_e2(E_R, A):
    m_e2 = (511000)**2 #squared electron mass in eV
    m_n = 0.9396*1e9 #nuetron mass in eV

    return 2*(m_e2/(A*m_n))*E_R
