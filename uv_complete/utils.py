#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
import uv_complete.geometry as geometry
import matplotlib.pyplot as plt
import copy
from datetime import timedelta
random_seed = 11141



def get_uvs_from_baseline_select(array_config,baseline_select,flip_tolerance = 0.0):
    uvs = []
    for i_bl,bl in enumerate(baseline_select):
        u = array_config[bl[1]][0] - array_config[bl[0]][0]
        v = array_config[bl[1]][1] - array_config[bl[0]][1]
        if v==0:
            u = np.abs(u)
        elif v<0:
            u *= -1
            v *= -1
        uv = np.array([u,v])
        
        uvs.append(uv)
    return np.asarray(uvs)

def select_baselines(commanded, antpos, fulfill_tolerance):
    not_fulfilled = np.copy(commanded)
    antpairs = []
    for i,pos1 in enumerate(antpos):
        for j,pos2 in enumerate(antpos):
            if j<=i:
                continue
            else:
                # Compute the difference between positions to get the uv point
                u = pos2[0] - pos1[0]
                v = pos2[1] - pos1[1]
                if v==0:
                    u = np.abs(u)
                elif v<0:
                    u *= -1
                    v *= -1
                uv = np.array([u,v])
                
                idx_new_fulfilled = np.where(np.linalg.norm(not_fulfilled - uv,axis=1,ord=np.inf) < fulfill_tolerance)
                #if len(not_fulfilled)%10==0 or len(not_fulfilled)<10:
                #    print(len(not_fulfilled))
                if len(idx_new_fulfilled[0])<1:
                    continue
                else:
                    not_fulfilled = np.delete(not_fulfilled,idx_new_fulfilled,axis=0)
                    antpairs.append((i,j))
                if len(not_fulfilled)%1000==0:
                    print(f'{len(not_fulfilled)}/{len(commanded)} remaining, antpairs now has length {len(antpairs)}.')
                if len(not_fulfilled)<1:
                    print('Done')
                    return antpairs

def get_array_config(antpos):
    if len(antpos.shape)==1:
        return {0: [antpos[0], antpos[1]]}
    else:
        return {i: [antpos[i,0], antpos[i,1]] for i in range(len(antpos))}




def generate_commanded_points(uv_cell_size=1., min_bl=10, max_bl=100, show_plot = True, ax = None):
    # Returns a grid of uv points in a half annulus, without the (u<0,v=0) segment
    
    uv_points = []
    
    max_u = max_bl
    min_u= -max_bl
    max_v = max_bl
    min_v = 0
    
    # make right side
    for u in np.arange(0, max_u + uv_cell_size, uv_cell_size):
        for v in np.arange(min_v, max_v + uv_cell_size, uv_cell_size):
            distance = np.sqrt(u**2 + v**2)
            if min_bl < distance <= max_bl:
                if not (u<0 and v==0):
                    #print(u,v)
                    uv_points.append((u, v))
    
    # make left side
    for u in np.arange(-uv_cell_size, min_u - uv_cell_size, -uv_cell_size):
        for v in np.arange(min_v, max_v + uv_cell_size, uv_cell_size):
            distance = np.sqrt(u**2 + v**2)
            if min_bl < distance <= max_bl:
                if not (u<0 and v==0):
                    #print(u,v)
                    uv_points.append((u, v))
    uv_points = np.asarray(uv_points)
    if show_plot:
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=[10,5])
        ax.plot(uv_points[:,0],uv_points[:,1],'.',markersize=1,color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Commanded points')
        ax.set_xlim(min_u-1, max_u+1)
        ax.set_ylim(min_v-1, max_v+1)
        ax.set_xlabel(r'$u$ [m]')
        ax.set_ylabel(r'$v$ [m]')
        ax.grid()
        
        
    uv_points = uv_points[np.argsort(np.linalg.norm(uv_points, axis=1))]
    
    #uv_dict = {i:uv_points[i] for i in range(len(uv_points))}
    
    return uv_points#uv_dict

def get_efficiency(n_fulfilled, antpos):
    if len(antpos) == 1:
        return 1.
    else:
        n_baselines = len(antpos) * (len(antpos) - 1) / 2
        return (n_fulfilled / n_baselines) ** .5

def get_efficiency_array(commanded, antpos, n_not_fulfilled_list):
    efficiency_array = []
    for i in range(len(antpos)):
        n_fulfilled = len(commanded) - n_not_fulfilled_list[i]
        n_baselines = i * (i - 1) / 2
        if n_baselines == 0:
            efficiency_array.append(1)
        else:
            efficiency_array.append( (n_fulfilled / n_baselines) ** .5 )
        
        
    return efficiency_array
    

def shuffle_antpos(antpos,diameter,tightness = 8, max_n_attempts = 1e6, verbose = True):
    
    shuffled_antpos = np.asarray([[0,0]])
    array_size = geometry.get_array_size(antpos)
    n_attempts = 0
    while len(shuffled_antpos) < len(antpos) and n_attempts<max_n_attempts:
        if verbose:
            if n_attempts > 0.9 * max_n_attempts and n_attempts%10==0:
                print('\n'+f'At attempt {n_attempts}/{max_n_attempts}... ', end = '')
        
        theta = 2 * np.pi * np.random.random()
        r = np.abs(np.random.normal())
        r *= array_size / tightness
        u = r * np.cos(theta)
        v = r * np.sin(theta)
        new_antpos = [u,v]
        array_size_pass = np.linalg.norm(new_antpos) < array_size/2
        #array_size_pass = True
        if array_size_pass:
            temp_shuffled_antpos = np.vstack([shuffled_antpos,new_antpos])
            if not geometry.collision_check(temp_shuffled_antpos,diameter):
                shuffled_antpos = temp_shuffled_antpos
                if verbose:
                    if len(shuffled_antpos)%10==0:
                        print(f'Antpos {len(shuffled_antpos)}/{len(antpos)}... ', end='')
                n_attempts = 0
            else:
                n_attempts += 1
                continue
        else:
            n_attempts += 1
            continue
    if verbose:
        print('\nDone.')
    
    return shuffled_antpos

    
def nudge_antpos(antpos,diameter,fulfill_tolerance):
    for i in range(len(antpos)):
        success = False
        while not success:
            nudge_u = 2 * (np.random.random() - 0.5) * fulfill_tolerance
            nudge_v = 2 * (np.random.random() - 0.5) * fulfill_tolerance
            temp_antpos = copy.deepcopy(antpos)
            temp_antpos[i,0]+=nudge_u
            temp_antpos[i,1]+=nudge_v
            if not geometry.collision_check(temp_antpos,diameter):
                antpos = temp_antpos
                success = True
    return antpos


def get_n_baselines_involved(antpos,commanded, fulfill_tolerance):
    n_baselines_involved = np.zeros(len(antpos),dtype = int)
    for i in range(len(antpos)):
        n_baselines_involved[i] = len(geometry.get_new_fulfilled(antpos[i], antpos, commanded,fulfill_tolerance))
    return n_baselines_involved

def get_n_baselines_involved_unique(antpos,commanded,fulfill_tolerance):
    n_baselines_involved = np.zeros(len(antpos),dtype = int)
    for i in range(len(antpos)):
        temp_antpos = np.delete(antpos,i,axis=0)
        _, not_fulfilled_idx = geometry.check_fulfillment(commanded,temp_antpos,fulfill_tolerance)
        n_baselines_involved[i] = len(not_fulfilled_idx)
    return n_baselines_involved


def format_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"


