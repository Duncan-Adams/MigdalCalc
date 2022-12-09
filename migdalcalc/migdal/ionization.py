#ionization.py - define charge production models
from . import kinematics as kin
import numpy as np
 
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

#Fano smearing for the binned method
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
    

#produces the spectrum of electron events in Silicon from a spectrum of ionization energy by chopping the spectrum into bins of 3.8eV
#and applying fano smearing
def Si_electron_spectrum_binned(Eion_spectrum, Y, En, c, A, flux = 1, number_of_bins=20, fano=True):
    ER_el = kin.E_R_elastic(c, A, En)
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
    

#produces the spectrum of electron events in a noble gas
def noblegas_electron_spectrum_binned(Eion_spectrum, Y, W, F, En, c, A, flux = 1, number_of_bins=20, fano=True):
    ER_el = kin.E_R_elastic(c, A, En)
    e0 = W
    F = F
    
    E_el_Q = Y(ER_el)*ER_el
    
    n_e_base = np.floor(E_el_Q/e0)


    first_bin = n_e_base*e0
    last_bin = first_bin + (number_of_bins + 1)*e0
        
    bins = np.arange(first_bin, last_bin, e0)
    n_e_bins = np.arange(n_e_base, number_of_bins + 1, 1, dtype=int)
    print(bins)
    print(n_e_bins)
    
    hist = []

    tracker = 0
    
    for n_e in n_e_bins:
        # lower_bound = np.round(bins[n_e - 1], 6)
        # upper_bound = np.round(bins[n_e], 6)
        lower_bound = n_e*e0
        upper_bound = (n_e+1)*e0
        rate = flux*integrate.quad(Eion_spectrum, lower_bound, upper_bound, limit=100, epsrel=1e-4)[0]

        if(tracker == 0):
            print(lower_bound, upper_bound, rate)
            tracker = 1

        if rate < 1e-12:
            rate = 0
        hist.append(rate)
        
    
    if(fano == True):
        fano_bins, fano_hist = Fano_smearing(n_e_bins, hist, F, nsig = 5)
        return fano_bins, fano_hist
        

    return n_e_bins, hist

def noblegas_electron_spectrum_binned_elastic(rate, Y, W, F, En, c, A, flux = 1, number_of_bins=20, fano=True):
    ER_el = kin.E_R_elastic(c, A, En)
    e0 = W
    F = F
    
    E_el_Q = Y(ER_el)*ER_el
    
    n_e_base = np.floor(E_el_Q/e0)
        
    n_e_bins = np.arange(n_e_base, n_e_base + 2, dtype=int)
    
    hist = []
    
    hist.append(rate)        
    hist.append(0)

    if(fano == True):
        fano_bins, fano_hist = Fano_smearing(n_e_bins, hist, F, nsig = 5)
        return fano_bins, fano_hist
        
    return n_e_bins, hist

#produces spectrum by integrating against the n electron probabilities from 2004.10709
def Si_electron_spectrum(Eion_spectrum, ER_Q, flux=1, start_bin=0, number_of_bins=20, gaussian=False):        
    n_bins = np.arange(start_bin, start_bin + number_of_bins, 1, dtype=int)
    r_bins = []

    for n in n_bins:
        if gaussian==True:
            p_n = get_p_n_Si_gauss(n)
        else:
            p_n = get_p_n_Si(n)
            
        
        integrand = lambda E: flux*Eion_spectrum(E)*p_n(E)
        E_range = np.geomspace(1 + ER_Q, 1000 + ER_Q, 201)
        
        samples = integrand(E_range)

        I = integrate.trapezoid(samples, E_range)

        r_bins.append(I)
        
    return n_bins, r_bins


def Si_elastic_electron_spectrum_fixed_angle(ER_Q, Elastic_Rate, start_bin=0, number_of_bins=20, gaussian=False):
    n_bins = np.arange(start_bin, start_bin + number_of_bins, 1, dtype=int)
    r_bins = []

    for n in n_bins:
        if gaussian==True:
            p_n = get_p_n_Si_gauss(n)
        else:
            p_n = get_p_n_Si(n)
            
        rate = Elastic_Rate*p_n(ER_Q) #because the elastic spectrum is a delta function at fixed angle
        if(rate < 1e-10):
            rate = 0
        
        r_bins.append(rate)
    
    return n_bins, r_bins
