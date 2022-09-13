#this script extracts the function I(omega) from darkelf and rescales it for use in my code. 
#Note that i had to modify darkelf to return I all the way down to the bandgap instead of twice the band gap
from darkelf import darkelf
import numpy as np

Si = darkelf(mX = 1e8, mMed = 0.0, target = 'Si', filename = 'Si_mermin.dat', phonon_filename = 'Si_epsphonon_data6K.dat')

#Darkelf evaluates their unscaled migdal probability at 1 eV of NR, not q_e
#So we need to scale it to 1 ev of q_e

A = 28
m_e = 0.511*1e6
m_n = 1e9
scale_factor  = (A*m_n/2)*(1/m_e)**2

Energy_grid = Si.I_tab.x
dPdOmega_grid = np.multiply(Si.I_tab.y, scale_factor)

#write this out to a csv
np.savetxt('si_elf.dat', list(zip(Energy_grid, dPdOmega_grid)), fmt='%.5e', delimiter=',')
