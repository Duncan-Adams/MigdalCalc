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

plt.rcParams['figure.dpi'] = 200

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


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CB_color_cycle)


if __name__ == '__main__':
    Energies = (2*1e3, 23*1e3, 2.507*1e6)
    omegas = (10, 30, 100)

    angles = np.linspace(0, 180, 1000)

    for (En, omega) in product(Energies, omegas):

        kmax_plot = []

        for angle in angles:
            kmax_plot.append(1e-3*kin.k_max(En, omega, angle, 28))

        plt.plot(angles, kmax_plot,label=r'k$_\mathrm{max}$')
        plt.title(r'E$_n$ = ' + str(En*1e-3) + r' keV; $\omega$ = ' + str(omega) + ' eV')
        plt.xlabel(r'$\theta_{\mathrm{Lab}}$ [deg]')
        plt.hlines(10, 0, 180,color='red',label="Free Electron Gas Bound")
        plt.hlines(30, 0, 180, color='green', label="Mermin Bound")
        plt.xlim(0, 180)
        plt.yscale('log')
        plt.ylabel(r'k [keV]')
        plt.legend()
        # plt.show()
        plt.savefig("./plots/Si/softlim/kmax_" + str(int(En*1e-3)) + "keV_" + str(omega) + "eV.png")
        plt.cla()

    
    plt.close()

    hanah_neutron_enegies = (23, 2507)
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

        axs.plot(angles, (1/norm)*hannah_softlim(np.cos(np.pi*angles/180)), label="Soft")
        axs.plot(angles, (1/norm)*hannah_full(np.cos(np.pi*angles/180)), label="Full")
        axs.plot(angles, (1/norm)*hannah_half(np.cos(np.pi*angles/180)), label="Low-Momentum")

        # axs[1].plot(c_range, 100*abs((hannah_full(c_range) - hannah_softlim(c_range))/hannah_full(c_range)))
        # axs[1].set_ylim(0, 100)

        print(hannah_scalefactor*hannah_softlim(-1))


        axs.set_xlabel(r'$\theta_{\mathrm{n}}$')
        axs.set_ylabel(r'Arbitrary Units')
        axs.set_title(r'E$_n$ = ' + str(En) + r' keV; $\omega$ = ' + str(omega) + ' eV')
        axs.legend()

        plt.savefig("./plots/Si/softlim/comparison_" + str(En) + "keV_" + str(omega) + "eV.png")

        plt.close()