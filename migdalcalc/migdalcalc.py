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
        
        nuclear_dict = dict()
        for ds in data['nuclear_datasets']:
            nuclear_dict = nuclear_dict | ds
            
        nuclear_dataset = nuclear_dict[nuclear_data_name]
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
        #jacobian of Enr -> cos theta_n
        J = lambda dE, c: kin.Jac_lab(dE, c, A, En) 
        
        def migdal_spectrum(deltaE, c):
            return spectrum_Enr(kin.E_Recoil(deltaE, c, A, En), deltaE)*J(deltaE, c)
            
        return migdal_spectrum

    #differential rate in cos theta_CM and Delta E, based on eqn 94 of ibe
    def d2R_dc_dDeltaE_CM_ibe(self, En, method='ibe'):
        A = self.A
        E0 = kin.E0(A, En)
        
        spectrum_Enr = self.d2R_dEnr_dDeltaE(En, method)
        #jacobian of Enr -> cos theta_n
        J = lambda dE, c: kin.Jac_lab_CM(dE, c, A, En) 
        
        def migdal_spectrum(deltaE, c):
            return spectrum_Enr(kin.E_Recoil_CM(deltaE, c, A, En), deltaE)*J(deltaE, c)
            
        return migdal_spectrum
    
    # Helper function that converts Eion to deltaE for a given quenching model Y(Enr)
    def _get_DeltaE_Eion(self, c, En, Y):
        A = self.A
        E0 = kin.E0(A, En)
        
        dE_max = kin.DeltaE_Max(A, En, c)
        dE_range = np.geomspace(1e-3, dE_max, 1000)
        
        Eion_arr = []
        
        for dE in dE_range:
            ER = kin.E_Recoil(dE, c, A, En)
            Eion = Y(ER)*ER + dE
            Eion_arr.append(Eion)
        
        return interp1d(Eion_arr, dE_range, bounds_error=False, fill_value = 0)
    
    #differential migdal rate in ionization energy at fixed (Lab) angle
    def dR_dEion(self, c, En, Y, method='ibe'):
        dE_c_spec = self.d2R_dc_dDeltaE(En, method)
        DeltaE_Eion = self._get_DeltaE_Eion(c, En, Y)
        
        def Eion_spectrum(Eion):
            return dE_c_spec(DeltaE_Eion(Eion), c)
            
        return Eion_spectrum

    #differential migdal rate in ionization energy at fixed (CM) angle, based on equation 94 of ibe
    def dR_dEion_CM_ibe(self, c, En, Y, method='ibe'):
        dE_c_spec = self.d2R_dc_dDeltaE_CM_ibe(En, method)
        DeltaE_Eion = self._get_DeltaE_Eion(c, En, Y)
        
        def Eion_spectrum(Eion):
            return dE_c_spec(DeltaE_Eion(Eion), c)
            
        return Eion_spectrum
        
    #Get the differenial migdal cross section, integrated over nuclear recoil energies, assuming Hard Sphere scattering
    def dR_dDeltaE_HS(self, En, method='ibe', flux=1):
        A = self.A
        m_e = 511000
        one_eV = 1
        m_n = 0.939*1e9
        M_A = A*m_n
        
        sigma = self.nuc.SIG(En)
        
        pref = 2*((m_e/one_eV)**2)*((1/(A+1))**2)
        
        dE_max = kin.DeltaE_Max(A, En)
        
        def migdal_spectrum_HS(dE):
            return flux*pref*En*(sigma/m_n)*self.mig.dP_dDeltaE(dE, method)*(2 - ((A+1)/A)*(dE/En))*np.sqrt(np.heaviside(dE_max - dE, 1)*(1 - ((A+1)/A)*(dE/En)))
        
        return migdal_spectrum_HS
    
    def dR_dEnr_HS_elastic(self, En, flux=1):
        A = self.A
        sigma = self.nuc.SIG(En)
        E0 = kin.E0(A, En)
        
        def elastic_spectrun_HS(E_nr):
            return np.heaviside(4*E0 - E_nr,1)*flux*sigma/(4*E0)
            
        return elastic_spectrun_HS
        
