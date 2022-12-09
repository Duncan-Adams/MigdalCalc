import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate
import math

from itertools import product

from migdalcalc.migdal import kinematics as kin


#########################################################################
plt.rcParams['figure.dpi'] = 400
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15) 
plt.rcParams["axes.formatter.use_mathtext"] = True

font = {'family' : 'serif',
         'weight' : 'bold',
         'size'   : 12,
         'serif':  'cmr10'
         }

matplotlib.rc('font', **font)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)

fs = 15

#########################################################################

def load_csv(file_name, skip_header=0):
    try:
        plot_data = np.genfromtxt(file_name, delimiter=',', skip_header=skip_header)
    except FileNotFoundError:
        return None, None
    
    x = list(zip(*plot_data))[0]
    y = list(zip(*plot_data))[1]
    
    return x, y

def interp_csv(file_name, skip_header=0,fv=0):
    x, y = load_csv(file_name, skip_header)

    if(x == None or y == None):
        return None
    
    return interp.interp1d(x, y, bounds_error=False, fill_value = fv, kind='quadratic')
    
# from http://randlet.com/blog/python-significant-figures-format/
def to_precision(x,p):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x/tens)

    if n < math.pow(10, p - 1):
        e = e -1
        tens = math.pow(10, e - p+1)
        n = math.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= math.pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -3 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)
############################################

if __name__ == '__main__':
    Energies = (2*1e3, 24*1e3, 2.507*1e6)
    omegas = (10, 30, 100)

    angles = np.linspace(0, 180, 1000)

    for (En, omega) in product(Energies, omegas):

        kmax_electron_gas = 10
        kmax_gpaw = 22

        kmax_plot = []

        for angle in angles:
            kmax_plot.append(1e-3*kin.k_max(En, omega, angle, 28))

        plt.plot(angles, kmax_plot,label='Soft Limit Bound')
        plt.title(r'$\mathit{E_n}$ = ' + str(En*1e-3) + r' keV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)
        plt.xlabel(r'$\mathit{\theta_n}$ [deg]',fontsize=fs)
        plt.hlines(kmax_electron_gas, 0, 180,color='red',label="Free Electron Regime",linestyle='dashed')
        plt.hlines(kmax_gpaw, 0, 180, color='green', label="GPAW Regime",linestyle='dashed')
        plt.xlim(0, 180)
        plt.yscale('log')
        plt.ylabel(r'$\mathit{k}$ [keV]',fontsize=fs)
        plt.legend(prop={"size":15})
        # plt.show()

        if(En == 2*1e3):
            plt.title(r'$\mathit{E_n}$ = 2 keV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)

        if(En == 24*1e3):
            plt.title(r'$\mathit{E_n}$ = 24 keV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)

        if(En == 2.507*1e6):
            plt.title(r'$\mathit{E_n}$ = 2.5 MeV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)


        arrow_x_locs = np.arange(5.5, 180, 20)

        for xl in arrow_x_locs:
            arrow_rel_height = 0.3
            width = 0.003

            length_eg = arrow_rel_height*kmax_electron_gas
            length_gpaw = arrow_rel_height*kmax_gpaw

            head_width = 0.8

            plt.arrow(xl, kmax_electron_gas, 0, -1*length_eg, color='red',
                width=width, head_width=head_width, head_length=0.2*length_eg,length_includes_head=True)

            plt.arrow(xl, kmax_gpaw, 0, -1*length_gpaw, color='green',
                width=width, head_width=head_width, head_length=0.2*length_gpaw,length_includes_head=True)


        plt.savefig("./plots/Si/softlim/kmax_" + str(int(En*1e-3)) + "keV_" + str(omega) + "eV.png")      
        plt.cla()

    
    plt.close()

    hanah_neutron_enegies = (24, 2507)
    hannah_omegas = (10, 100)

    

    for (En, omega) in product(hanah_neutron_enegies, hannah_omegas):
        hannah_softlim = interp_csv("./input/hannah/soft_" + str(int(En)) + "_" + str(omega) + ".csv",fv='extrapolate')
        hannah_full = interp_csv("./input/hannah/not_" + str(int(En)) + "_" + str(omega) + ".csv",fv='extrapolate')
        hannah_half = interp_csv("./input/hannah/half_" + str(int(En)) + "_" + str(omega) + ".csv",fv='extrapolate')

        if(hannah_softlim == None or hannah_full == None):
            continue


        c_range = np.linspace(-1, 1)

        alpha=1/137
        m_n = 0.939*1e9
        v = kin.velocity(En, m_n)
        sigma_0 = 1.7071614819547574

        hannah_scalefactor = (8*np.pi**2*alpha/v)*(sigma_0/m_n**2)

        norm = integrate.quad(lambda c: hannah_full(c), -1, 1)[0]
        print(norm)


        # fig, axs = plt.subplots(2, 1,sharex=True, gridspec_kw={'height_ratios': [10, 1]})
        fig, axs = plt.subplots(1,1)

        # axs.plot(angles, (1/norm)*hannah_softlim(np.cos(np.pi*angles/180)), label="Soft")
        # axs.plot(angles, (1/norm)*hannah_full(np.cos(np.pi*angles/180)), label="Full")
        # axs.plot(angles, (1/norm)*hannah_half(np.cos(np.pi*angles/180)), label="Low-Momentum")

        axs.plot(c_range, (1/norm)*hannah_full(c_range), label="Full", color='orange',linewidth=2)
        axs.plot(c_range, (1/norm)*hannah_softlim(c_range), label="Soft-Limit", linestyle='dashed')
        axs.plot(c_range, (1/norm)*hannah_half(c_range), label="Low-Momentum",color='g',linestyle='dotted')

        # axs[1].plot(c_range, 100*abs((hannah_full(c_range) - hannah_softlim(c_range))/hannah_full(c_range)))
        # axs[1].set_ylim(0, 100)

        axs.set_xlabel(r'$\cos$ $\mathit{\theta_n}$',fontsize=fs)
        axs.set_ylabel(r'$\frac{d \sigma}{d \cos \theta} $ [Arbitrary Units]',fontsize=fs)
        axs.set_title(r'$\mathit{E_n}$ = ' + str(En) + r' keV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)
        if(En == 2507):
            axs.set_title(r'$\mathit{E_n}$ = 2.5 MeV; $\mathit{\omega}$ = ' + str(omega) + ' eV',fontsize=fs)
        plt.legend(prop={"size":15})

        plt.savefig("./plots/Si/softlim/comparison_" + str(En) + "keV_" + str(omega) + "eV.png")

        plt.close()