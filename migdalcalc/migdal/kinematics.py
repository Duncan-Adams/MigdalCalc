# Kinematics.py - kinematical functions used in various migdal calculations
import numpy as np

def velocity(En, m):
    v2 = 1 - 1/((En/m + 1)**2)

    return np.sqrt(v2)
