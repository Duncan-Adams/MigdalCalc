#ionization.py - define charge production models
from . import kinematics as kin
import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.ndimage as ndi

import os


import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

#get the gaussian p_n defined in 2004.10709
def get_p_n_Si_gauss(n):
    F_inf = 0.119
    eps_inf = 3.8

    norm = np.sqrt(2*np.pi*n*F_inf)
    std_dev = np.sqrt(n*F_inf)*eps_inf
   
    def p_n(E):
        num = n*eps_inf - E
        return (1/norm)*np.exp(-0.5*(num/std_dev)**2)
        
    if(n == 0):
        def p_0(E):
            return 1*np.heaviside(1.2 - E, 1)
        return p_0
    if n == 1:
        def p_1(E):
            E = min(E, 2*eps_inf)
            p_2 = get_p_n_Si_gauss(2)
            p_3 = get_p_n_Si_gauss(3)
            return (1 - p_2(E) - p_3(E))*np.heaviside(E - 1.2, 1)
        return p_1
        
    return p_n

#for the low energy bins we load the calculations from 2004.10709
def get_p_n_Si(n):
    data_file = os.path.dirname(__file__) + '/../../targets/data/Si/ionization/p0K.dat'
    
    if(n == 0):
        def p_0(E):
            return 1*np.heaviside(1.2 - E, 1)
        return p_0
    if(n <= 10):
        prob_tab = np.genfromtxt(data_file)
        energies = list(zip(*prob_tab))[0]
        probs = list(zip(*prob_tab))[n]
        
        p_n = interp.interp1d(energies, probs, bounds_error=False, fill_value=0, kind='quadratic')

        return p_n
    return get_p_n_Si_gauss(n)

#Fano smearing for the crude method
def Fano_smearing(bins, hist, F, nsig = 5):
    N_e_low = bins[0]
    N_e_hi = bins[-1]
    
    N_e_low_smeared = max(np.floor(N_e_low - nsig*np.sqrt(N_e_low*F)), 0)
    N_e_hi_smeared = np.ceil(N_e_hi + nsig*np.sqrt(N_e_hi*F))
    
    
    smeared_bins = np.arange(N_e_low_smeared, N_e_hi_smeared + 1)
    smeared_hist = [0]*len(smeared_bins)
    
    
    for (i, n) in enumerate(bins):
        std_dev = np.sqrt(n*F)
        
        temp_hist = [0]*len(smeared_bins)
        #need to determine the bin index to place the value from the original histogram
        th_index = None
        for j in range(len(smeared_bins)):
            if smeared_bins[j] == n:
                th_index = j
                
        if(th_index == None):
            print("error in fano smearing")
            exit()
            
        temp_hist[th_index] = hist[i]
        
        smear = ndi.gaussian_filter(temp_hist, std_dev, truncate=nsig)
        smeared_hist = np.add(smeared_hist, smear)
        
    return smeared_bins, smeared_hist


def fixed_angle_elastic_spectrum(migdal_data, En, angle, e0, F, Y, flux=1, fano=True):
    A = migdal_data.A
    c = np.cos(angle*np.pi/180)
    E0 = (A/(A+1)**2)*En
 
    E_el_Q = Y(kin.ER_0(c, En, A))*kin.ER_0(c, En, A)
    
    elastic_ct = migdal_data.endf.getDA(En)
    elastic_xsec = elastic_ct(c)
    
    elastic_bins = [np.floor(E_el_Q/e0)]
    elastic_histo = [flux*elastic_xsec]
    
    if(fano == True):
        fano_bins, fano_hist = Fano_smearing(elastic_bins, elastic_histo, F, nsig=5)
        return fano_bins, fano_hist
        
    return elastic_bins, elastic_histo
    
def fixed_angle_elastic_spectrum_moments(migdal_data, En, angle, Y, flux=1):
    A = migdal_data.A
    c = np.cos(angle*np.pi/180)
    E0 = (A/(A+1)**2)*En
    e0 = 3.8
 
    E_el_Q = Y(kin.ER_0(c, En, A))*kin.ER_0(c, En, A)
    
    n_e_base = np.floor(max((E_el_Q - 1.2), 0)/e0)
    n_e_pm = np.floor(10*0.34*np.sqrt(n_e_base))
    
    n_e_low = max(n_e_base - n_e_pm, 0)
    n_e_hi = n_e_base + n_e_pm
    
    n_e_bins = np.arange(n_e_low, n_e_hi + 1, 1, dtype=int)
    n_e_rates = []
    
    elastic_ct = migdal_data.endf.getDA(En)
    elastic_xsec = elastic_ct(c)
    
    elastic_rate = flux*elastic_xsec
    print(elastic_rate)
    
    for n_e in n_e_bins:
        p_n = get_p_n_Si(n_e)
        rate = elastic_rate*p_n(E_el_Q)
        if(rate < 1e-12):
            rate = 0
        n_e_rates.append(rate)
    

    return n_e_bins, n_e_rates
#produces the spectrum of electron events in Silicon from a spectrum of ionization energy by chopping the spectrum into bins of 3.8eV
#and applying fano smearing
def fixed_angle_electron_spectrum_Si_crude(Eion_spectrum, Y, En, c, A, flux = 1, number_of_bins=20, fano=True):
    E0 = (A/(A+1)**2)*En
    # ~ c = np.cos(angle*np.pi/180)
    ER_el = kin.ER_0(c, E0, A)
    e0 = 3.8
    F = 0.119
    
    E_el_Q = Y(ER_el)*ER_el
    
    n_e_base = np.floor(E_el_Q/e0)
        
    first_bin = n_e_base*e0 + 1.2
    last_bin = first_bin + (number_of_bins + 1)*e0
        
    bins = np.arange(first_bin, last_bin, e0)
    n_e_bins = np.arange(1, number_of_bins + 1, 1, dtype=int)
    
    hist = []
    
    for n_e in n_e_bins:
        lower_bound = np.round(bins[n_e - 1], 6)
        upper_bound = np.round(bins[n_e], 6)
        rate = flux*integrate.quad(Eion_spectrum, lower_bound, upper_bound, limit=100, epsrel=1e-4)[0]
        # ~ E_range = np.geomspace(lower_bound, upper_bound, 101)
        # ~ samples = Eion_spectrum(E_range)
        # ~ rate = flux*integrate.trapz(samples, E_range)
        if rate < 1e-12:
            rate = 0
        hist.append(rate)
        
    if(n_e_base == 0):
        zero_bin = rate = flux*integrate.quad(Eion_spectrum, 0, 1.2, limit=100, epsrel=1e-4)[0]
        hist = [zero_bin] + hist
        n_e_bins = [0] + n_e_bins
    
    if(fano == True):
        fano_bins, fano_hist = Fano_smearing(n_e_bins, hist, F, nsig = 5)
        return fano_bins, fano_hist
        
    
    return n_e_bins, hist
    
#produces spectrum by integrating against the n electron probabilities from 2004.10709
def fixed_angle_electron_spectrum_Si_moments(Eion_spectrum, ER_Q, flux=1, number_of_bins = 20, gaussian=False):
    n_bins = np.arange(0, 1 + number_of_bins, 1)
    r_bins = []

    for n in n_bins:
        if gaussian==True:
            p_n = get_p_n_Si_gauss(n)
        else:
            p_n = get_p_n_Si(n)
            
        integrand = lambda E: flux*Eion_spectrum(E)*p_n(E)
        E_start = max(1.2, ER_Q)
        E_start = max(E_start, n*3.8 - 5*0.34*np.sqrt(n)*3.8)
        E_end = n*3.8 + 5*0.34*np.sqrt(n)*3.8
        #these endpoints are derived essentially as the 5 sigma boundaries for the n_e distributions
        if(n == 0):
            E_start = 0
            E_end = 1.2
        
        I, E = integrate.quad(integrand, E_start, E_end, epsrel=1e-3, limit=200)
        rate = I
        r_bins.append(rate)
        
    return n_bins, r_bins
