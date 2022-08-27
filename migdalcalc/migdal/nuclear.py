import json
import scipy.integrate as integrate
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as leg
import numpy.polynomial as poly
from math import comb

class nuclear():
    ##Helper functions
    def __load_DA(self, DA_file):
        with open(DA_file, "r") as f:    
            data_arr = []
            
            #skip the header
            f.readline()
            f.readline()
            
            stop = 0
            data = []
            while True:
                line = f.readline()
                if(len(line) == 0):
                    break
                
                endf_row = self.__read_line(line)
                
                if (endf_row[0] != 'empty'):
                    stop += 1
                    
                if(stop == 2):
                    data_arr.append(data)
                    stop = 1
                    data = []
                
                for column in endf_row:
                    if column != 'empty':
                        data.append(column)
            
            energies = []
            a_l_arr = []
            for data in data_arr:
                energies.append(data[0]) #the key is the numeric value of the energy in eV
                a_l = data[1:]
                a_l.insert(0, 1.0)
                a_l_arr.append(a_l)
                
            
            max_len_a_l = 0
            for a_l in a_l_arr:    
                max_len_a_l = max(max_len_a_l, len(a_l))
            
            a_l_points = []
            a_l_interps = []
            for l in range(0, max_len_a_l):
                a_l_points.append([])
                
                for a_l in a_l_arr:
                    if(l < len(a_l)):
                        a_l_points[l].append(a_l[l])
                    else:
                        a_l_points[l].append(0)
                        
                a_l_interps.append(interp.interp1d(energies, a_l_points[l], bounds_error=True))
                            
            return a_l_interps
            
    def __read_float(self, v):
        if (v.strip() == ''):
            return 'empty'
        try:
            return float(v)
        except ValueError:
            return float(v[0] + v[1:].replace('+', 'e+').replace('-', 'e-'))
    
    #you can thank the endf standard for this cursed shit...
    def __read_line(self, l):
        slices = {'data': (slice(0, 12), slice(12, 23), slice(23, 34), slice(34, 45), slice(45, 56), slice(56, 67), (slice(67, 78)))}
        return [self.__read_float(l[s]) for s in slices['data']]
    
    def __load_SIG(self, SIG_file):
        #Construct Interpreter for total xsec
        with open(SIG_file, "r") as f:
            json_data = json.load(f)
            data_points = json_data['datasets'][0]['pts']
            
            E_arr = []
            sig_arr = []
            
            for point in data_points:
                E_arr.append(point["E"])
                sig_arr.append(point["Sig"])
            return interp.interp1d(E_arr, sig_arr, bounds_error=False, fill_value=0)  
       
    def __init__(self, A, SIG_file, DA_file):
        self.A = A
        self.DA = self.__load_DA(DA_file)
        self.SIG = self.__load_SIG(SIG_file)
    
    #Gives the double differential cross section in Enr, phi
    def dSig_dEnr(self, En):
        sigma = self.SIG(En)
        A = self.A
        
        a_l = []
        
        for a_l_i in self.DA:
            if(abs(a_l_i(En)) > 0):
                a_l.append(a_l_i(En))

        highest_power = len(a_l) #highest power in energy the expanstion will have
        energy_coeeficients = [0]*highest_power
    
        E0 = (A/(A+1)**2)*En
        jac = 1/(2*E0) #jacobian from cos theta to ER
    
        for ell in range(highest_power):
    
            pref = 0.5*a_l[ell]*(2*ell + 1)*sigma/(2*np.pi)
    
            for k in range(ell+1):
    
                sign = (-1)**k
                comb_factor = (1/4**k)*comb(ell, k)*comb(ell + k, k)
    
                acc = sign*pref*comb_factor*(1/E0**k)
    
                energy_coeeficients[k] += jac*acc
    
        return poly.Polynomial(energy_coeeficients)

    def dSig_dEnr_test(self, En):
        sigma = self.SIG(En)
        A = self.A
        
        a_l = []
        
        for a_l_i in self.DA:
            if(abs(a_l_i(En)) > 0):
                a_l.append(a_l_i(En))

        highest_power = len(a_l) #highest power in energy the expanstion will have
        energy_coeeficients = [0]*highest_power
    
        E0 = (A/(A+1)**2)*En
        jac = 1/(2*E0) #jacobian from cos theta to ER
    
        for k in range(highest_power):
    
            pref = 0.5*sigma/(2*np.pi)
            sign = (-1)**k
            E_factor = (1/(4*E0)**k)

            for ell in range(k, highest_power):


                comb_factor = comb(ell, k)*comb(ell + k, k)
    
                acc = E_factor*a_l[ell]*(2*ell + 1)*sign*pref*comb_factor
    
                energy_coeeficients[k] += jac*acc
    
        return poly.Polynomial(energy_coeeficients)
    
    #Gives the double differential cross section in cos theta, phi
    def dSig_dOmega(self, En):
        normalized_coeff = []
        sigma = self.SIG(En)
                
        a_l = []
        
        for a_l_i in self.DA:
            if(abs(a_l_i(En)) > 0):
                a_l.append(a_l_i(En))

        for l in range(len(a_l)):
            C = (2*l + 1)/2
            N = ((sigma)/(2*np.pi))
            normalized_coeff.append(C*N*a_l[l])

        return leg.Legendre(normalized_coeff)

