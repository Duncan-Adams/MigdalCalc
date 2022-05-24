import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import numpy as np
import scipy.interpolate as interp
import scipy.integrate as integrate

def load_csv(file_name, skip_header=0):
    plot_data = np.genfromtxt(file_name, delimiter=',', skip_header=skip_header)
    
    x = list(zip(*plot_data))[0]
    y = list(zip(*plot_data))[1]
    
    return x, y

def interp_csv(file_name, skip_header=0):
    x, y = load_csv(file_name, skip_header)
    
    return interp.interp1d(x, y, bounds_error=False, fill_value = 0, kind='quadratic')


if __name__ == '__main__':
    Energies = 2, 23, 2507
    angles = [10, 72]
    # ~ Energies = [54]
    # ~ angles_impact = [13, 20, 29, 37]
    
    plot_settings = {
        2: {
            10: {
                'xmin': 0,
                'xmax': 9,
                'ymin': 1e-8,
                'ymax': 1e-1
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
                'ymin': 1e-8,
                'ymax': 1e-1
            },
            
            72: {
                'xmin': 20,
                'xmax': 100,
                'ymin': 1e-8,
                'ymax': 1e-1
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
                'ymax': 1e-1
            },
            
            72: {
                'xmin': 15700,
                'xmax': 16300,
                'ymin': 1e-10,
                'ymax': 1e-2
            }
        }
        
    }
        

    # ~ for angle in angles_impact:
        # ~ migdal_x, migdal_y = load_csv('../output/electron_yield/Si/' + str(E_impact) +  'keV/elf-ibe/' + 'yield_sarkis_' + str(angle) + 'deg.csv')
        # ~ elastic_x, elastic_y = load_csv('../output/electron_yield/Si/' + str(E_impact) +  'keV/elf-ibe/' + 'elastic_sarkis_' + str(angle) + 'deg.csv')
        
        
        # ~ print("Total Migdal Events: " + str(np.sum(migdal_y[1:])))
        # ~ plt.step(migdal_x, migdal_y, where='mid', label='Migdal')
        # ~ plt.step(elastic_x, elastic_y, where='mid', label="Elastic")
        
        # ~ plt.title(r"$\theta_{CM}$ = " + str(angle) + "deg")
        # ~ plt.xlabel('electron-hole pairs')
        # ~ plt.ylabel('Rate [g$^{-1}$ day$^{-1}$]')
        
        # ~ xmin = plot_settings[E_impact][angle]['xmin']
        # ~ xmax = plot_settings[E_impact][angle]['xmax']
        # ~ ymin = plot_settings[E_impact][angle]['ymin']
        # ~ ymax = plot_settings[E_impact][angle]['ymax']
        # ~ plt.xlim(xmin, xmax)
        # ~ plt.ylim(ymin, ymax)
        # ~ plt.yscale('log')
        # ~ plt.legend()
        # ~ plt.savefig(str(angle) + 'deg.png', dpi=900)
        # ~ plt.cla()
        
    
    
    
        
    for Energy in Energies:
        for angle in angles:
            
            si_sarkis_prediction_migdal = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/Ne_sarkis_migdal.csv')
            si_lindhard_prediction_migdal = interp_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/Ne_lindhard_migdal.csv')
            
            si_sarkis_prediction_elastic = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/Ne_sarkis_elastic.csv')
            si_lindhard_prediction_elastic = load_csv('./output/' + str(Energy) + 'keV/' + str(angle) + 'deg/Ne_lindhard_elastic.csv')
            
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
            
            plt.step(x_pts, si_sarkis_prediction_migdal(x_pts), where='mid', label='Sarkis Migdal', linestyle = 'dashed', color='red')
            plt.step(x_pts, si_lindhard_prediction_migdal(x_pts), where='mid', label='Lindhard Migdal',linestyle='dashed',color ='blue')
            
            plt.step(si_sarkis_prediction_elastic[0], si_sarkis_prediction_elastic[1], where='mid', label='Sarkis Elastic', color='red')
            plt.step(si_lindhard_prediction_elastic[0], si_lindhard_prediction_elastic[1], where='mid', label='Lindhard Elastic', color='blue')
            
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)
            plt.title(r'E$_N$ = ' + str(Energy) + 'keV; $\Theta_{CM}$ = ' + str(angle) + 'deg')
            plt.xlabel('Electron-Hole Pairs')
            plt.ylabel('Events/neutron')
            plt.legend()
            plt.yscale('log')
            plt.savefig('./plots/Si/N_eh_' + str(Energy) + 'keV_' + str(angle) + 'deg.png')
            plt.cla()
    
