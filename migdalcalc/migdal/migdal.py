from scipy.interpolate import interp1d as interp1d
import numpy as np
import os

class migdal():
    #This function takes a data file with deltaE in the first column, and migdal
    #probability in the second. It assumes energy units in eV and migdal probability normalized at q_e = 1 eV
    def __add_method(self, name, datafile):
        migdal_prob_data = np.genfromtxt(datafile, delimiter=',')
        
        x = list(zip(*migdal_prob_data))[0]
        y = list(zip(*migdal_prob_data))[1]
        
        prob_interp = interp1d(x, y, bounds_error=False, fill_value=0, kind='linear')
        
        self.method_dict[name] = prob_interp
        
        return
        
    #methods is an array of tuples containing the name of the method and a datafile with the migdal probabilities
    def __init__(self, methods=None, directory='./'):
        self.method_dict = dict()
        for method in methods:
            name = method[0]
            datafile = method[1]
            
            self.__add_method(name, directory + datafile)
        
    def dP_dDeltaE(self, DeltaE, method='ibe'):
        return self.method_dict[method](DeltaE)
    
        
