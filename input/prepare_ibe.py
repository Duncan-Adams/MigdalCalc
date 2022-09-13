#prepare ibe.py - puts ibe data into the form used for migdal calc
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

def prepare_ibe(ibe_datafile, n_shells, energies):
    n_pts_shell = 254
    
    with open(ibe_datafile, "r") as df:
        data = df.readlines()
        
        n_lines = len(data)
        shells = []
        
        for n in range(0, n_shells):
            shell_slice = slice(n_pts_shell*n, n_pts_shell*(n+1))
            n_prin, ell = np.genfromtxt((data[shell_slice])[slice(1,2)])
            
            shell_data = np.genfromtxt(data[shell_slice], skip_header=3)
            E_nl = energies[(n_prin,ell)]
            
            dE_arr = []
            pe_arr = []
            #Add a zero point, since ibe only goes down to 1 eV
            dE_arr.append(E_nl)
            pe_arr.append((1/(2*np.pi))*shell_data[0][1])
            for data_point in shell_data:
                
                
                deltaE = data_point[0] + E_nl
                diff_P = (1/(2*np.pi))*data_point[1]
                
                dE_arr.append(deltaE)
                pe_arr.append(diff_P)
                
            shells.append(interp.interp1d(dE_arr, pe_arr, bounds_error=False, fill_value = 0, kind='linear'))
        
        deltaE_final = np.geomspace(min(energies.values()), 5e4, 1000)
        prob_final = []
        
        for dE in deltaE_final:
            prob = 0
            for shell in shells:
                prob += shell(dE)
            prob_final.append(prob)
        
    return deltaE_final, prob_final
    
################################################################################
# Xenon Binding Energies
################################################################################
Xe_Energies = dict()
Xe_Energies[(1,0)] = 3.5e4
Xe_Energies[(2,0)] = 5.4e3
Xe_Energies[(2,1)] = 4.9e3
Xe_Energies[(3,0)] = 1.1e3
Xe_Energies[(3,1)] = 9.3e2
Xe_Energies[(3,2)] = 6.6e2
Xe_Energies[(4,0)] = 2.0e2
Xe_Energies[(4,1)] = 1.4e2
Xe_Energies[(4,2)] = 61
Xe_Energies[(5,0)] = 21
Xe_Energies[(5,1)] = 9.8
################################################################################
# Silicon Binding Energies
################################################################################
Si_Energies = dict()
Si_Energies[(1,0)] = 1.872e3
Si_Energies[(2,0)] = 167
Si_Energies[(2,1)] = 115
Si_Energies[(3,0)] = 14.7
Si_Energies[(3,1)] = 8.1
################################################################################
# Argon Binding Energies
################################################################################
Ar_Energies = dict()
Ar_Energies[(1,0)] = 3.2e3
Ar_Energies[(2,0)] = 3.0e2
Ar_Energies[(2,1)] = 2.4e2
Ar_Energies[(3,0)] = 2.7e1
Ar_Energies[(3,1)] = 1.3e1
################################################################################
Xe_dE, Xe_p = prepare_ibe('./ibe/Xe.dat', n_shells = 11, energies=Xe_Energies)
Si_dE, Si_p = prepare_ibe('./ibe/Si.dat', n_shells = 3, energies=Si_Energies)
Ar_dE, Ar_p = prepare_ibe('./ibe/Ar.dat', n_shells = 5, energies=Ar_Energies)

np.savetxt('../targets/data/Xe/migdal/ibe.dat', list(zip(Xe_dE, Xe_p)), fmt='%.5e', delimiter=',')
np.savetxt('../targets/data/Si/migdal/ibe.dat', list(zip(Si_dE, Si_p)), fmt='%.5e', delimiter=',')
np.savetxt('../targets/data/Ar/migdal/ibe.dat', list(zip(Ar_dE, Ar_p)), fmt='%.5e', delimiter=',')
