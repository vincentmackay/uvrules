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
    
    def __init__(self, antpos=np.array([0, 0]), commanded=None, diameter=10, max_array_size = None, mid_freq = 140e6, bandwidth = 20e6, freq_step = 0.1e6, packing_density = 2, min_bl_lambda = 10, max_bl_lambda = 100, uv_cell_size = None, fulfill_tolerance = None, fulfilled_idx=None, not_fulfilled_idx=None,  use_midband = True, p_norm = np.inf, array_config = None ):
        '''
            Initialize the AntArray object
            
            Parameters:
            ----------------------------------------------------------------------------------------
                antpos: numpy array of floats
                    Array of shape (n,2) that contains the position of the antennas. Default is np.array([0,0]),
                    i.e. one antenna at the center of the coordinate system.
                commanded: numpy array of floats
                    Array of shape (n,2) that contains the commanded uv points. Optional, can be generated with
                    .generate_commanded(). Required to use a "rules"-type algorithm.
                diameter: float
                    Diameter of the array dishes, in meters. Required, with a default value of 10.
                max_array_size: float or None
                    Maximum size of the array, in meters. Optional.
                fulfilled: numpy array of floats
                    Array of shape (n,2) that contains the fulfilled uv points. Optional, since `commanded` is
                    optional.
                not_fulfilled: numpy array of floats
                    Array of shape (n,2) that contains the uv points that are yet to be fulfilled. Optioinal,
                    since `commanded` is optional.
                p_norm: float or np.inf
                    Norm to use to check fulfillment. Should remain np.inf.
        '''
        self.use_midband=use_midband
        self.antpos = antpos
        self.commanded = commanded
        self.diameter = diameter
        self.mid_freq = mid_freq
        self.bandwidth = bandwidth
        self.freq_step = freq_step
        self.freqs = np.arange(self.mid_freq - self.bandwidth/2, self.mid_freq + self.bandwidth/2+self.freq_step,self.freq_step)
        self.wavelengths = constants.c.value / self.freqs
        self.min_bl_lambda = min_bl_lambda
        self.max_bl_lambda = max_bl_lambda
        if use_midband:
            self.min_bl = min_bl_lambda * constants.c.value / mid_freq
            self.max_bl = max_bl_lambda * constants.c.value / mid_freq
        else:
            self.min_bl = min_bl_lambda * np.min(self.wavelengths)
            self.max_bl = max_bl_lambda * np.max(self.wavelengths)
        self.packing_density = packing_density
        self.fulfilled_idx = fulfilled_idx
        self.not_fulfilled_idx = not_fulfilled_idx
        self.p_norm = p_norm
        
        if uv_cell_size is None:
            if use_midband:
                self.uv_cell_size = constants.c.value / mid_freq / self.packing_density
            else:
                self.uv_cell_size = np.min(self.wavelengths) / self.packing_density
        else:
            self.uv_cell_size = uv_cell_size
        if fulfill_tolerance is None:
            self.fulfill_tolerance = self.uv_cell_size/2
        self.max_array_size = max_array_size
        
        if commanded is None:
            self.generate_commanded(show_plot=False)
        
        if array_config is None:
            array_config = uv_complete.utils.get_array_config(self.antpos)
        
    
    def check_commanded(method):
        def wrapper(self, *args, **kwargs):
            if self.commanded is None:
                raise ValueError("self.commanded is None. Cannot perform the operation.")
            return method(self, *args, **kwargs)
        return wrapper
    
        
    def generate_commanded(self, show_plot = True, ax = None):
        self.commanded = uv_complete.utils.generate_uv_grid(self.uv_cell_size, self.min_bl, self.max_bl, show_plot, ax)
        
    
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
    def add_ant_rules(self, order = -1, show_plot = True, save_file = False,save_name='rules', verbose = True, n_max_antennas = np.inf, n_to_add = np.inf, center_at_origin = True, check_first = 'antpos', check_all_antpos = True, check_all_not_fulfilled = False, show_plot_skip = 10, second_priority = 'min_distance_from_new_antpos'):
        self.antpos = uv_complete.rules.add_ant_rules(self.commanded,self.antpos,self.diameter,self.max_array_size,self.fulfill_tolerance,order,show_plot,save_file,save_name,verbose,n_max_antennas,n_to_add,center_at_origin,check_first,check_all_antpos,check_all_not_fulfilled,show_plot_skip,second_priority)
    
    @check_commanded
    def add_ant_rules_parallelized(self,center_at_origin = True, n_to_add = np.inf, n_max_antennas = np.inf, save_file = True, save_name = 'para_default_name', verbose = True, show_plot = False, num_cores = 64):
        self.antpos = uv_complete.rules.add_ant_rules_parallelized(self.commanded, self.antpos, self.diameter, self.max_array_size, self.fulfill_tolerance, center_at_origin = center_at_origin, n_to_add = n_to_add, n_max_antennas = n_max_antennas, save_file=save_file, save_name = save_name, verbose=verbose, show_plot=show_plot, num_cores = num_cores)
    
    @check_commanded
    def add_ant_rules_parallelized_2(self,center_at_origin = True, n_to_add = np.inf, n_max_antennas = np.inf, save_file = True, save_name = 'para_default_name', verbose = True, show_plot = False, try_continue = True, num_cores = None):
        self.antpos = uv_complete.rules.add_ant_rules_parallelized_2(self.commanded, self.antpos, self.diameter, self.max_array_size, self.fulfill_tolerance, center_at_origin = center_at_origin, n_to_add = n_to_add, n_max_antennas = n_max_antennas, save_file=save_file, save_name = save_name, verbose=verbose, show_plot=show_plot, try_continue = try_continue, num_cores = num_cores)  
    
    @check_commanded
    def add_ant_rules_2(self,center_at_origin = True, order = -1, n_to_add = np.inf, n_max_antennas = np.inf, check_all_commanded = False, check_all_antpos = True, save_file = True, save_name = 'para_default_name', verbose = True, show_plot = False, try_continue = True, num_cores = None):
        self.antpos = uv_complete.rules.add_ant_rules_2(self.commanded, self.antpos, self.diameter, self.max_array_size, self.fulfill_tolerance, center_at_origin = center_at_origin, order = order, n_to_add = n_to_add, n_max_antennas = n_max_antennas, check_all_commanded = check_all_commanded, check_all_antpos = check_all_antpos, save_file=save_file, save_name = save_name, verbose=verbose, show_plot=show_plot, try_continue = try_continue, num_cores = num_cores)
          
          
          
          
          
          
        
        
        
        
        
        
        
        
        
        
        
        
        
