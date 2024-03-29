import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate
import math

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
    plot_data = np.genfromtxt(file_name, delimiter=',', skip_header=skip_header)
    
    x = list(zip(*plot_data))[0]
    y = list(zip(*plot_data))[1]
    
    return x, y

def interp_csv(file_name, skip_header=0):
    x, y = load_csv(file_name, skip_header)
    
    return interp.interp1d(x, y, bounds_error=False, fill_value = 0, kind='quadratic')
    
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
    Energies = 2, 23, 24, 2507
    angles = [10, 72]
    
    plot_settings = {
        2: {
            10: {
                'xmin': 0,
                'xmax': 9,
                'ymin': 1e-9,
                'ymax': 1e-3
            },
            
            72: {
                'xmin': 0,
                'xmax': 9,
                'ymin': 1e-8,
                'ymax': 1e-1
            }
        },
        
        23: {
            10: {
                'xmin': 0,
                'xmax': 9,
                'ymin': 1e-9,
                'ymax': 1e-3
            },
            
            72: {
                'xmin': 20,
                'xmax': 100,
                'ymin': 1e-9,
                'ymax': 1e-3
            }
        },

        24: {
            10: {
                'xmin': 0,
                'xmax': 9,
                'ymin': 1e-9,
                'ymax': 1e-3
            },
            
            72: {
                'xmin': 20,
                'xmax': 100,
                'ymin': 1e-9,
                'ymax': 1e-3
            }
        },
        
        #impact points
        54: {
            13: {
                'xmin': 0,
                'xmax': 10,
                'ymin': 1,
                'ymax': 1e4
            },
            
            20: {
                'xmin': 0,
                'xmax': 18,
                'ymin': 1,
                'ymax': 1e4
            },
            
            29: {
                'xmin': 0,
                'xmax': 32,
                'ymin': 1e-10,
                'ymax': 1e-2
            },
            
            37: {
                'xmin': 10,
                'xmax': 50,
                'ymin': 1e-8,
                'ymax': 1e-2
            }
        },
        
        2507:{
            10: {
                'xmin': 100,
                'xmax': 220,
                'ymin': 1e-8,
                'ymax': 1e-3
            },
            
            72: {
                'xmin': 16600,
                'xmax': 17200,
                'ymin': 1e-8,
                'ymax': 1e-4
            }
        }
        
    }
        
    method = "gpaw_withLFE"
        
    for Energy in Energies:
        for angle in angles:
            
            elastic_recoil_energy = kin.E_Recoil(0, np.cos(np.pi*angle/180), 28, Energy)
            elastic_energy_str = to_precision(elastic_recoil_energy, 3)
            
            si_sarkis_prediction_migdal = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_migdal.csv')
            si_lindhard_prediction_migdal = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_lindhard_migdal.csv')
            si_sarkis22_migdal            = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_migdal22.csv')

            si_sarkis_prediction_elastic = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_elastic.csv')
            si_lindhard_prediction_elastic = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_lindhard_elastic.csv')
            si_sarkis22_elastic           = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_elastic22.csv')


            si_sarkis_prediction_migdal_atomic = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/ibe_Ne_sarkis_migdal.csv')
            si_lindhard_prediction_migdal_atomic = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/ibe_Ne_lindhard_migdal.csv')
            si_sarkis22_migdal_atomic = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/ibe_Ne_sarkis_migdal22.csv')


            n_e_baseline = 0
            for i in range(len(si_sarkis_prediction_migdal.x)):
                if si_sarkis_prediction_migdal.y[i] > 0:
                    n_e_start = si_sarkis_prediction_migdal.x[i]
                    break
                    
            n_e_end = si_lindhard_prediction_migdal.x[-1]
                    
            x_pts = np.arange(n_e_baseline, n_e_end, 1)
            
            ymin = plot_settings[Energy][angle]['ymin']
            ymax = plot_settings[Energy][angle]['ymax']
            
            xmin = plot_settings[Energy][angle]['xmin']
            xmax = plot_settings[Energy][angle]['xmax']

            ax = plt.gca()

            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            plt.yscale('log')

            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

            ax_r = ax.secondary_yaxis('right')
            ax_t = ax.secondary_xaxis('top')
            ax_r.tick_params(axis='y', direction='out')
            ax_t.tick_params(axis='x', direction='out')
            
            ax_r.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
            ax_r.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

            ax_r.yaxis.set_major_formatter(ticker.NullFormatter())
            ax_r.yaxis.set_minor_formatter(ticker.NullFormatter())

            ax_t.xaxis.set_major_formatter(ticker.NullFormatter())
            ax_t.xaxis.set_minor_formatter(ticker.NullFormatter())
            
            if(Energy != 2507 or angle !=72):
                plt.step(x_pts, si_sarkis_prediction_migdal(x_pts) + si_sarkis_prediction_migdal_atomic(x_pts), where='mid', label='Sarkis Migdal', linestyle = 'dotted', color='red')
                plt.step(x_pts, si_sarkis22_migdal(x_pts) + si_sarkis22_migdal_atomic(x_pts), where='mid', label='Sarkis 2022 Migdal', linestyle='dotted', color='orange')

            plt.step(x_pts, si_lindhard_prediction_migdal(x_pts) + si_lindhard_prediction_migdal_atomic(x_pts), where='mid', label='Lindhard Migdal',linestyle='dotted',color ='blue')
            
            if(Energy != 2507 or angle !=72):
                plt.step(si_sarkis_prediction_elastic[0], si_sarkis_prediction_elastic[1], where='mid', label='Sarkis Elastic', color='red')
                plt.step(si_sarkis22_elastic[0], si_sarkis22_elastic[1], where='mid', label='Sarkis 2022 Elastic', color='orange')

            plt.step(si_lindhard_prediction_elastic[0], si_lindhard_prediction_elastic[1], where='mid', label='Lindhard Elastic', color='blue')

            
            plt.title(r'$\mathit{E_n}$ = ' + str(Energy) + r' keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = ' + elastic_energy_str + ' keV',fontsize=fs)

            if(Energy == 2 and angle == 10):
                plt.title(r'$\mathit{E_n}$ = 2 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 2.2 eV',fontsize=fs)
                ax.set_xticks([0, 1,2,3,4,5,6,7,8,9],labels=["0", "1", "2", "3","4", "5","6", "7", "8", "9"],minor=False)    
                ax_t.set_xticks([0, 1,2,3,4,5,6,7,8,9],minor=False)

            if(Energy == 2 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 2 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 96 eV',fontsize=fs)
                ax.set_xticks([0, 1,2,3,4,5,6,7,8,9],labels=["0", "1", "2", "3","4", "5","6", "7", "8", "9"],minor=False)    
                ax_t.set_xticks([0, 1,2,3,4,5,6,7,8,9],minor=False)

            if(Energy == 23 and angle == 10):
                plt.title(r'$\mathit{E_n}$ = 23 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 25 eV',fontsize=fs)
                ax.set_xticks([0, 1,2,3,4,5,6,7,8,9],labels=["0", "1", "2", "3","4", "5","6", "7", "8", "9"],minor=False)    
                ax_t.set_xticks([0, 1,2,3,4,5,6,7,8,9],minor=False)

            if(Energy == 23 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 23 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 1.1 keV',fontsize=fs)
                ax.set_xticks([25, 35,45,55,65,75,85,95], minor=True)
                ax_t.set_xticks([25, 35,45,55,65,75,85,95], minor=True)

            if(Energy == 24 and angle == 10):
                plt.title(r'$\mathit{E_n}$ = 24 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 25 eV',fontsize=fs)
                ax.set_xticks([0, 1,2,3,4,5,6,7,8,9],labels=["0", "1", "2", "3","4", "5","6", "7", "8", "9"],minor=False)    
                ax_t.set_xticks([0, 1,2,3,4,5,6,7,8,9],minor=False)

            if(Energy == 24 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 24 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 1.1 keV',fontsize=fs)
                ax.set_xticks([25, 35,45,55,65,75,85,95], minor=True)
                ax_t.set_xticks([25, 35,45,55,65,75,85,95], minor=True)

            if(Energy == 2507 and angle == 10):
                plt.title(r'$\mathit{E_n}$ = 2.5 MeV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 2.7 keV',fontsize=fs)
                ax.set_xticks([110, 130, 150,170,190,210], minor=True)
                ax_t.set_xticks([110, 130, 150,170,190,210], minor=True)

            if(Energy == 2507 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 2.5 MeV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 120 keV',fontsize=fs)
                ax.set_xticks([16650, 16750, 16850,16950,17050,17150], minor=True)
                ax_t.set_xticks([16650, 16750, 16850,16950,17050,17150], minor=True)

            plt.xlabel('Electron-Hole Pairs',fontsize=fs)
            plt.ylabel(r"$dP/d\theta_n$ [events/neutron/$\Delta\theta_n$]",fontsize=fs)
            plt.legend(prop={"size":15})
            plt.savefig('./plots/Si/N_eh_' + str(Energy) + 'keV_' + str(angle) + 'deg.png')
            plt.cla()


    # exit()
    for Energy in Energies:
        for angle in angles:
            
            elastic_recoil_energy = kin.E_Recoil(0, np.cos(np.pi*angle/180), 28, Energy)
            elastic_energy_str = to_precision(elastic_recoil_energy, 3)
            
            migdal_ibe = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/ibe_Ne_sarkis_migdal22.csv')
            migdal_elf = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_migdal22.csv')
            
            si_sarkis_prediction_elastic = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_elastic22.csv')
            elastic_interp = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/' + method + '_Ne_sarkis_elastic22.csv')

            
            n_e_baseline = 0
                    
            n_e_end = max(migdal_ibe.x[-1], migdal_elf.x[-1]) 

            n_e_total = np.arange(0, n_e_end, dtype=int)

            rate_total = []

            for n_e in n_e_total:
                rate_total.append(migdal_ibe(n_e) + migdal_elf(n_e) + elastic_interp(n_e))
                    
            x_pts = np.arange(n_e_baseline, n_e_end, 1)
            
            ymin = plot_settings[Energy][angle]['ymin']
            ymax = plot_settings[Energy][angle]['ymax']
            
            xmin = plot_settings[Energy][angle]['xmin']
            xmax = plot_settings[Energy][angle]['xmax']


            plt.fill_between(si_sarkis_prediction_elastic[0], si_sarkis_prediction_elastic[1], step='mid', label='Quenched Elastic', alpha=0.65)
            plt.fill_between(x_pts, migdal_elf(x_pts), step='mid', label='Valence Band ME', alpha=0.65)
            plt.fill_between(x_pts, migdal_ibe(x_pts), step='mid', label="Inner Shells ME", alpha=0.65)

            plt.step(n_e_total, rate_total, where='mid',label="Total Rate", color="black")


            
            # plt.step(si_sarkis_prediction_elastic[0], si_sarkis_prediction_elastic[1], where='mid', label='Elastic'

            ax = plt.gca()


            
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)

            plt.yscale('log')


            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

            ax_r = ax.secondary_yaxis('right')
            ax_t = ax.secondary_xaxis('top')
            ax_r.tick_params(axis='y', direction='out')
            ax_t.tick_params(axis='x', direction='out')
            
            ax_r.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12))
            ax_r.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

            ax_r.yaxis.set_major_formatter(ticker.NullFormatter())
            ax_r.yaxis.set_minor_formatter(ticker.NullFormatter())

            ax_t.xaxis.set_major_formatter(ticker.NullFormatter())
            ax_t.xaxis.set_minor_formatter(ticker.NullFormatter())

            plt.title(r'$\mathit{E_n}$ = ' + str(Energy) + r' keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = ' + elastic_energy_str + ' keV',fontsize=fs)

            if(Energy == 23 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 23 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 1.1 keV',fontsize=fs)
                ax.set_xticks([25, 35,45,55,65,75,85,95], minor=True)
                ax_t.set_xticks([25, 35,45,55,65,75,85,95], minor=True)

            if(Energy == 24 and angle == 72):
                plt.title(r'$\mathit{E_n}$ = 24 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 1.1 keV',fontsize=fs)
                ax.set_xticks([25, 35,45,55,65,75,85,95], minor=True)
                ax_t.set_xticks([25, 35,45,55,65,75,85,95], minor=True)

            if(Energy == 2 and angle == 10):
                plt.title(r'$\mathit{E_n}$ = 2 keV; $\mathit{\theta_n}$ = ' + str(angle) + r'$^\circ$; $\mathit{E_{nr,el}}$ = 2.2 eV',fontsize=fs)
                ax.set_xticks([0, 1,2,3,4,5,6,7,8,9],labels=["0", "1", "2", "3","4", "5","6", "7", "8", "9"],minor=False)    
                ax_t.set_xticks([0, 1,2,3,4,5,6,7,8,9],minor=False)      



            plt.xlabel('Electron-Hole Pairs',fontsize=fs)
            plt.ylabel(r"$dP/d\theta_n$ [events/neutron/$\Delta\theta_n$]",fontsize=fs)
            plt.legend(prop={"size":15})
            plt.tight_layout(pad=2.02)
            plt.savefig('./plots/Si/fig3/N_eh_' + str(Energy) + 'keV_' + str(angle) + 'deg_sarkis.png')
            plt.cla()
    
