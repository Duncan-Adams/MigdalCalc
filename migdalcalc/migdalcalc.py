import yaml
import os

import numpy as np
from scipy.interpolate import interp1d as interp1d

from .migdal import migdal 
from .migdal import nuclear
from .migdal import kinematics as kin

class migdalcalc():
    def __init__(self, target='Xe-131', nuclear_data_name='jendl40'):
        target_directory = os.path.dirname(__file__) + '/../targets/'
        yaml_file = target_directory + target + '.yaml'
        data = None
        
        with open(yaml_file, 'r') as yf:
            data = yaml.safe_load(yf)
        
        migdal_methods = data['methods']
        
        self.mig = migdal.migdal(migdal_methods, target_directory)
        
        nuclear_dataset = data['nuclear_datasets'][0][nuclear_data_name]
        self.A = data['A']
        
        self.nuc = nuclear.nuclear(self.A, target_directory + nuclear_dataset[0],  target_directory + nuclear_dataset[1])
        #useful conventions
        
    #differential migdal rate in Enr and DeltaE
    def d2R_dEnr_dDeltaE(self, En, method='ibe'):
        A = self.A
        
        E0 = kin.E0(A, En)
        elastic_spectrum = self.nuc.dSig_dEnr(En)
        
        def migdal_spectrum(Enr, DeltaE):
            return kin.q_e2(Enr, A)*elastic_spectrum(Enr)*self.mig.dP_dDeltaE(DeltaE, method)
            
        return migdal_spectrum
    
    #differential migdal rate in cos theta and DeltaE
    def d2R_dc_dDeltaE(self, En, method='ibe'):
        A = self.A
        E0 = kin.E0(A, En)
        
        spectrum_Enr = self.d2R_dEnr_dDeltaE(En, method)
        #jacobian of Enr -> cos theta
        J = lambda x: 2*E0*np.sqrt(1 - x)
        
        def migdal_spectrum(deltaE, c):
            R = deltaE/En
            
            return spectrum_Enr(kin.E_Recoil(deltaE, c, A, En), deltaE)*J(R)
            
        return migdal_spectrum
    
    # Helper function that converts Eion to deltaE for a given quenching model Y(Enr)
    def _get_DeltaE_Eion(self, c, En, Y):
        A = self.A
        E0 = kin.E0(A, En)
        
        dE_max = kin.DeltaE_Max(A, En)
        dE_range = np.geomspace(1e-3, dE_max, 1000)
        
        Eion_arr = []
        
        for dE in dE_range:
            ER = kin.E_Recoil(dE, c, A, En)
            Eion = Y(ER)*ER + dE
            Eion_arr.append(Eion)
        
        return interp1d(Eion_arr, dE_range, bounds_error=False, fill_value = 0)
    
    #differential migdal rate in ionization energy at fixed angle
    def dR_dEion(self, c, En, Y, method='ibe'):
        dE_c_spec = self.d2R_dc_dDeltaE(En, method)
        DeltaE_Eion = self._get_DeltaE_Eion(c, En, Y)
        
        def Eion_spectrum(Eion):
            return dE_c_spec(DeltaE_Eion(Eion), c)
            
        return Eion_spectrum
    
