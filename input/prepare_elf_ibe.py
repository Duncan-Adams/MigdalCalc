#prepare_elf_ibe.py - uses elf for outer shell, ibe for the rest
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from prepare_ibe import prepare_ibe, Si_Energies

def prepare_elf_ibe(elf_datafile, ibe_datafile, n_shells, energies):
    dE_ibe, p_ibe = prepare_ibe(ibe_datafile, n_shells, energies)
    
    elf_data = np.genfromtxt(elf_datafile, delimiter=',')
    
    dE_elf = list(zip(*elf_data))[0]
    p_elf = list(zip(*elf_data))[1]
    
    ibe_interp = interp.interp1d(dE_ibe, p_ibe, bounds_error=False, fill_value = 0, kind='linear')
    elf_interp = interp.interp1d(dE_elf, p_elf, bounds_error=False, fill_value = 0, kind='linear')
    
    dE_range = np.geomspace(1, 2.2e4, 1000)
    combined_p = []

    for dE in dE_range:
        p = ibe_interp(dE) + elf_interp(dE)
        combined_p.append(p)
        
    
    
    return dE_range, combined_p

Si_dE, Si_p = prepare_elf_ibe('./darkelf/Si_elf.dat', './ibe/Si.dat', n_shells=4, energies=Si_Energies)

np.savetxt('../targets/data/Si/migdal/elf-ibe.dat', list(zip(Si_dE, Si_p)), fmt='%.5e', delimiter=',')

