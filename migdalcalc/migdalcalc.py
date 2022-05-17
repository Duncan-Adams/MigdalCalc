import numpy as np
import yaml
import os

from .migdal import migdal 
from .migdal import nuclear

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
        A = data['A']
        
        self.nuc = nuclear.nuclear(A, target_directory + nuclear_dataset[0],  target_directory + nuclear_dataset[1])
        #useful conventions
        
    #differential migdal rate in Enr and omega
    def d2R_dEnr_domega
    
    #differential migdal rate in cos theta and omega
    def d2R_dc_domega
    
    
    
    #differential migdal rate in ionization energy at fixed angle
    def dR_dEion
    
