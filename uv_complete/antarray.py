#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:45:26 2024

@author: vincent
"""

import numpy as np
import uv_complete.utils, uv_complete.rules
from astropy import constants

class AntArray(object):
    
    def __init__(self, diameter=10, max_array_size = None, mid_freq = 140e6, bandwidth = 20e6,  packing_density = 2, min_bl_lambda = 10, max_bl_lambda = 100, fulfill_tolerance = None, use_midband = True, p_norm = np.inf):
        '''
            Initialize the AntArray object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                diameter: float
                    Diameter of the array dishes, in meters. Required, with a default value of 10.
                max_array_size: float or None
                    Maximum size of the array, in meters. Optional.
                    since `commanded` is optional.
                mid_freq: float
                    Midband frequency, in Hz.
                p_norm: float or np.inf
                    Norm to use to check fulfillment. Should remain np.inf.
        '''
        self.use_midband=use_midband
        self.antpos = np.array([0.,0.])
        self.diameter = diameter
        self.mid_freq = mid_freq
        self.bandwidth = bandwidth
        self.freqs = np.array([self.mid_freq - self.bandwidth/2, self.mid_freq + bandwidth/2])
        self.wavelengths = constants.c.value / self.freqs
        self.min_bl_lambda = min_bl_lambda
        self.max_bl_lambda = max_bl_lambda
        if use_midband:
            self.min_bl = min_bl_lambda * constants.c.value / mid_freq
            self.max_bl = max_bl_lambda * constants.c.value / mid_freq
            #self.min_bl = min_bl_lambda * (self.wavelengths[0] + self.wavelengths[1])/2
            #self.max_bl = max_bl_lambda * (self.wavelengths[0] + self.wavelengths[1])/2
        else:
            self.min_bl = min_bl_lambda * np.min(self.wavelengths)
            self.max_bl = max_bl_lambda * np.max(self.wavelengths)
        if self.min_bl < self.diameter:
            print('Minimum baseline shortest than diameter; impossible, setting min_bl to the value of diameter.')
            self.min_bl = diameter
        self.packing_density = packing_density
        self.fulfilled_idx = None
        self.not_fulfilled_idx = None
        self.p_norm = p_norm
        

        if use_midband:
            self.uv_cell_size = constants.c.value / mid_freq / self.packing_density
        else:
            self.uv_cell_size = np.min(self.wavelengths) / self.packing_density

        
        if fulfill_tolerance is None:
            self.fulfill_tolerance = self.uv_cell_size/2
        else:
            self.fulfill_tolerance = fulfill_tolerance
        self.max_array_size = max_array_size
        
        self.generate_commanded_points(show_plot=False)
            
        
        self.array_config = uv_complete.utils.get_array_config(self.antpos)
        
    
    def check_commanded(method):
        def wrapper(self, *args, **kwargs):
            if self.commanded is None:
                raise ValueError("self.commanded is None. Cannot perform the operation.")
            return method(self, *args, **kwargs)
        return wrapper
    
        
    def generate_commanded_points(self, show_plot = True, ax = None):
        self.commanded = uv_complete.utils.generate_commanded_points(self.uv_cell_size, self.min_bl, self.max_bl, show_plot, ax)
      
    def generate_commanded_grid(self, show_plot = True, ax = None):
        self.commanded_grid = uv_complete.utils.generate_commanded_grid(uv_cell_size = self.uv_cell_size, min_bl_lambda= self.min_bl_lambda, max_bl_lambda=self.max_bl_lambda, show_plot = show_plot, ax = ax)
        
    
    @check_commanded
    def check_fulfillment_idx(self, flip_tolerance = 0.0, return_arrays = False):
        self.fulfilled_idx, self.not_fulfilled_idx = uv_complete.utils.check_fulfillment_idx(self.commanded,self.antpos,self.fulfill_tolerance,self.p_norm,flip_tolerance)
        if return_arrays:
            return self.fulfilled_idx, self.not_fulfilled_idx
        
    @check_commanded
    def get_new_fulfilled(self, new_antpos):
        self.check_fulfillment()
        return uv_complete.utils.get_new_fulfilled(new_antpos, self.antpos, self.not_fulfilled, self.fulfill_tolerance, self.p_norm)
    
    def get_array_size(self):
        return uv_complete.utils.get_array_size(self.antpos)
    
    def get_min_distance_from_new_antpos(self, new_antpos):
        return uv_complete.utils.get_min_distance_from_new_antpos(self.antpos, new_antpos)
    
    def get_min_distance(self):
        return uv_complete.utils.get_min_distance(self.antpos)
    
    def collision_check(self):
        return uv_complete.utils.collision_check(self.antpos, self.diameter)
    
    def plot_array(self, just_plot_array=False,plot_new_fulfilled=False,fig=None,ax=None,n_new_fulfilled_list=None,n_not_fulfilled_list=None,new_fulfilled_list=None):
        uv_complete.utils.plot_array(self.antpos,self.commanded,self.fulfill_tolerance,just_plot_array,plot_new_fulfilled,fig,ax,n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list)
        
    def get_antpos_history(self):
        return uv_complete.utils.get_antpos_history(self.commanded,self.antpos,self.fulfill_tolerance)
        
    def set_array_config(self):
        self.array_config = uv_complete.utils.get_array_config(self.antpos)    
        
    def get_array_config(self):
        return uv_complete.utils.get_array_config(self.antpos)
        
    @check_commanded
    def pick_baselines(self):
        return uv_complete.utils.pick_baselines(self.commanded, self.antpos, self.fulfill_tolerance)
    
    @check_commanded
    def add_ant_rules(self,center_at_origin = True, order = -1, n_to_add = np.inf, n_max_antennas = np.inf, compare_all_commanded = False, compare_all_antpos = True, save_file = True, save_name = None, verbose = True, show_plot = True, try_continue = True, num_cores = None):
        self.antpos = uv_complete.rules.add_ant_rules(commanded = self.commanded, antpos = self.antpos, diameter = self.diameter, fulfill_tolerance = self.fulfill_tolerance, max_array_size = self.max_array_size, center_at_origin = center_at_origin, order = order, n_to_add = n_to_add, n_max_antennas = n_max_antennas, compare_all_commanded = compare_all_commanded, compare_all_antpos = compare_all_antpos, save_file=save_file, save_name = save_name, verbose=verbose, show_plot=show_plot, try_continue = try_continue, num_cores = num_cores)
          
          
          
          
          
          
        
        
        
        
        
        
        
        
        
        
        
        
        
