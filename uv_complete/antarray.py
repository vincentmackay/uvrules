#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:45:26 2024

@author: vincent
"""

import numpy as np
from astropy import constants
import os
import uv_complete.utils as utils
import uv_complete.algos.greedy
from collections import namedtuple
import pickle


# Define a structured result format
CandidateResult = namedtuple("CandidateResult", [
    "success", "n_new_fulfilled", "min_distance", "array_size", "favored_i", "favored_j", "favored_k", "rejected_combinations"
])



class AntArray(object):
    
    
    def __init__(self, diameter=None, diameter_lambda = None, max_array_size = None, ref_freq = None, ref_wl = None, packing_density = 2, min_bl_lambda = None, max_bl_lambda = None, min_bl = None, max_bl = None, fulfill_tolerance = None, p_norm = np.inf, verbose = False):
        '''
            Initialize the AntArray object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                diameter: float
                    Diameter of the array dishes, in meters. Required, with a default value of 10.
                max_array_size: float or None
                    Maximum size of the array, in meters. Optional.
                    since `commanded` is optional.
                ref_freq: float
                    Reference frequency, in Hz.
                p_norm: float or np.inf
                    Norm to use to check fulfillment. Should remain np.inf.
        '''
        
        self.verbose = verbose
        self.antpos = np.array([[0.,0.]])
        
        self.ref_wl, self.ref_freq = self._initialize_ref_wl(ref_wl, ref_freq, verbose)
        self.max_bl, self.max_bl_lambda = self._resolve_units(max_bl, max_bl_lambda, self.ref_wl, name = 'max_bl', verbose=self.verbose)
        self.min_bl, self.min_bl_lambda = self._resolve_units(min_bl, min_bl_lambda, self.ref_wl, name = 'min_bl', verbose=self.verbose)
        self.diameter, self.diameter_lambda = self._resolve_units(diameter, diameter_lambda, self.ref_wl, name = 'diameter', verbose=self.verbose)
        if self.min_bl < self.diameter:
            if verbose:
                print('Minimum baseline shortest than diameter; impossible, setting min_bl to the value of diameter.')
            self.min_bl = diameter
        self.packing_density = packing_density
        self.fulfilled_idx = None
        self.not_fulfilled_idx = None
        self.p_norm = p_norm
        self.uv_cell_size = constants.c.value / self.ref_freq / self.packing_density
        if fulfill_tolerance is None:
            self.fulfill_tolerance = self.uv_cell_size/2
        else:
            self.fulfill_tolerance = fulfill_tolerance
        self.max_array_size = max_array_size
        self.commanded = utils.generate_commanded_points(uv_cell_size=self.uv_cell_size, min_bl=self.min_bl, max_bl=self.max_bl,show_plot=False)
        self.array_config = utils.get_array_config(self.antpos)
        
        
        
        
    def _resolve_units(self, value, value_lambda, ref_wl, name, default_lambda = 2, verbose=False):
        """Generalized function to resolve values in meters and wavelengths.
    
        Parameters:
            value (float or None): The value in meters.
            value_lambda (float or None): The value in wavelengths.
            ref_wl (float): The reference wavelength.
            name (str): Name of the parameter (for verbosity messages).
            default_lambda (float): The default value in wavelengths.
            verbose (bool): If True, prints info.
    
        Returns:
            (value, value_lambda): Tuple of resolved values in meters and wavelengths.
        """
        if value is None and value_lambda is None:
            self._verbose_print(f"Using default {name}_lambda of {default_lambda}.")
            value_lambda = default_lambda
            value = value_lambda * ref_wl
        elif value is None:
            value = value_lambda * ref_wl
        elif value_lambda is None:
            value_lambda = value / ref_wl
        return value, value_lambda


    def _initialize_ref_wl(self,ref_wl=None, ref_freq=None, verbose=False):
        """Initialize the reference wavelength and frequency."""
        if ref_wl is not None and ref_freq is not None:
            self._verbose_print("Both ref_freq and ref_wl were given. Using ref_freq.")
    
        if ref_freq is None and ref_wl is None:
            self._verbose_print("Using default frequency of 150 MHz.")
            ref_freq = 150e6
            ref_wl = constants.c.value / ref_freq
        elif ref_freq is None:
            ref_freq = constants.c.value / ref_wl
        else:
            ref_wl = constants.c.value / ref_freq
    
        return ref_wl, ref_freq
    

    def _verbose_print(self, message):
        """Helper function to print messages when verbose=True."""
        if self.verbose:
            print(message)
    
    def set_array_config(self):
        self.array_config = utils.get_array_config(self.antpos)    
        
    def get_array_config(self):
        return utils.get_array_config(self.antpos)

    def select_baselines(self):
        return utils.select_baselines(self.commanded, self.antpos, self.fulfill_tolerance)
    
    def get_redundancy(self,red_tol_lambda = None):
        self.redundancy = utils.get_redundancy_lattice(self.antpos,self.ref_wl,red_tol_lambda)
        return self.redundancy
          
    
    def save(self, path_to_file):
        """Save the AntArray instance to disk."""
        with open(path_to_file, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path_to_file, inplace=True):
        """Load an AntArray instance from disk.

        Parameters:
        - path_to_file (str): Path to the saved AntArray instance.
        - inplace (bool, default=True): 
            - If True, update the current instance.
            - If False, return a new AntArray object.

        Returns:
        - If inplace=False, returns a loaded AntArray object.
        """
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"Checkpoint file {path_to_file} not found.")

        with open(path_to_file, "rb") as f:
            loaded_antarray = pickle.load(f)

        if inplace:
            self.__dict__.update(loaded_antarray.__dict__)  # Overwrite current instance
            return None  # Nothing to return
        else:
            return loaded_antarray  # Return a new instance
    
    
   
    def add_ant_greedy(self, **kwargs):
        return uv_complete.algos.greedy.add_ant_greedy(self, **kwargs)

   
    
    
        
        
        





        
