#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
import itertools
import copy
from astropy import constants
from collections import Counter
random_seed = 11141

def nonzero_sign(x):
    return np.where (x>=0,1,-1)

def get_redundancy_blind(antpos,ref_wl,min_bl_lambda = 0,max_bl_lambda = np.inf,red_tol_lambda = None):
    if red_tol_lambda is None:
        print('Using the default redundancy tolerance of 0.1 lambda.')
        red_tol = 0.1 * ref_wl
    else:
        red_tol = red_tol_lambda * ref_wl
    
    uvs = antpos_to_uv(antpos)
    
    bl_lengths = np.linalg.norm(uvs, axis=1)
    
    min_bl = min_bl_lambda * ref_wl
    max_bl = max_bl_lambda * ref_wl
    
    mask_range = (bl_lengths >= min_bl) & (bl_lengths <= max_bl)
    uvs_in_range = uvs[mask_range]
    
    tree = cKDTree(uvs_in_range)
    redundancy = tree.query_ball_point(uvs_in_range, red_tol, return_length=True)
    
    # Remove one for the redundancy, because a point with only 1 point in its 
    # neighborhood (itself) should count as having no redundancy
    redundancy -= 1
    
    return redundancy


def get_redundancy_lattice(antpos, ref_wl, red_tol_lambda = None):
    if red_tol_lambda is None:
        print('Using the default redundancy tolerance of 0.1 lambda.')
        red_tol = 0.1 * ref_wl
    else:
        red_tol = red_tol_lambda * ref_wl
    
    uvs = antpos_to_uv(antpos)
    # Determine the bounds of the rectangle that contains all points
    min_coords = np.floor(uvs.min(axis=0))
    
    # Map points to lattice squares using integer division
    square_indices = ((uvs - min_coords) // red_tol).astype(int)
    
    # Count the occurrences of each square index
    counts = Counter(map(tuple, square_indices))
    
    # Return the non-zero counts as a list of integers
    return list(counts.values())


def get_redundancy_commanded(commanded,antpos,ref_wl,red_tol_lambda = None):
    if red_tol_lambda is None:
        print('Using the default redundancy tolerance of 0.1 lambda.')
        red_tol = 0.1 * ref_wl
    else:
        red_tol = red_tol_lambda * ref_wl
    
    uvs = antpos_to_uv(antpos)
    tree = cKDTree(uvs)
    redundancy = tree.query_ball_point(commanded, r=red_tol, return_length=True)
    return redundancy

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

def get_antpos_history(commanded, antpos, fulfill_tolerance):
    n_new_fulfilled_list = []
    n_not_fulfilled_list = []
    new_fulfilled_list = []
    if antpos.shape != (2,):
        if antpos.shape==2:
            new_fulfilled_temp = get_new_fulfilled(antpos[1],antpos[0],commanded,fulfill_tolerance)
            n_new_fulfilled_list.append(len(new_fulfilled_temp))
            new_fulfilled_list.append(new_fulfilled_temp)
        elif len(antpos)>2:
            not_fulfilled_temp = np.copy(commanded)
            for i in range(len(antpos)):
                antpos_temp = antpos[:i]
                new_fulfilled_temp = get_new_fulfilled(antpos[i],antpos_temp,not_fulfilled_temp,fulfill_tolerance)
                n_new_fulfilled_list.append(len(new_fulfilled_temp))
                new_fulfilled_list.append(new_fulfilled_temp)
                fulfilled_temp, not_fulfilled_temp = check_fulfillment(not_fulfilled_temp,antpos_temp, fulfill_tolerance)
                n_not_fulfilled_list.append(len(not_fulfilled_temp))
    return n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list

def plot_array(antpos, commanded = None, diameter = None, fulfill_tolerance = 0.5, just_plot_array = False, plot_new_fulfilled = False, fig=None,ax=None,n_new_fulfilled_list = None,n_not_fulfilled_list = None,new_fulfilled_list = None, fulfilled = None, not_fulfilled = None,step_time_array = None, efficiency_array = None):
    
    if commanded is None:
        just_plot_array=True
        print('No commanded array passed, will just plot the array.')
    
    if just_plot_array:
        if fig is None and ax is None:
            fig,ax = plt.subplots(1,1,figsize = [5,5])
        ax.plot(antpos[:,0],antpos[:,1],'.')
        ax.set_xlabel(r'EW [m]')
        ax.set_ylabel(r'NS [m]')
        ax.set_aspect('equal')
    elif not plot_new_fulfilled:
        if fig is None and ax is None:
            fig,ax = plt.subplots(1,2,figsize=[12,6])
            ax[0].plot(commanded[:,0],commanded[:,1],'.',markersize=2,color='k',alpha=1,label='Commanded points',zorder=0)
            ax[0].set_title('uv plane')
            ax[0].set_xlabel(r'$u$')
            ax[0].set_ylabel(r'$v$')
            ax[1].plot(antpos[:,0],antpos[:,1],'.',color='k')
            ax[1].set_title(f'Array ({len(antpos)} antennas)')
            ax[1].set_xlabel(r'EW [m]')
            ax[1].set_ylabel(r'NS [m]')
            for i in [0,1]:
                ax[i].set_aspect('equal')
        
    else:
        if fig is None and ax is None:
            fig,ax = plt.subplots(2,2,figsize=(12,10))
        if n_new_fulfilled_list is None or n_not_fulfilled_list is None or new_fulfilled_list is None:
            n_new_fulfilled_list,n_not_fulfilled_list, new_fulfilled_list = get_antpos_history(commanded, antpos,fulfill_tolerance)
        if fulfilled is None and not_fulfilled is None:
            fulfilled,not_fulfilled = check_fulfillment(commanded,antpos,fulfill_tolerance)
        
        colormap = cm.viridis
        ax[0,0].plot(commanded[:,0],commanded[:,1],'.',markersize=2,color='k',alpha=1,label='Commanded points',zorder=0)
        for i,new_fulfilled in enumerate(new_fulfilled_list):
            if new_fulfilled is not None:
                if len(new_fulfilled.shape)>1:
                    ax[0,0].plot(new_fulfilled[:,0],new_fulfilled[:,1],'.',markersize=2,color=colormap(i/len(new_fulfilled_list)),zorder=1)
        #ax[0].scatter(fulfilled[:,0],fulfilled[:,1],c=colors_fulfilled,s=1,zorder=1)#,label='Fulfilled points')
        ax[0,0].set_title('uv plane')
        ax[0,0].set_xlabel(r'$u$')
        ax[0,0].set_ylabel(r'$v$')
        
        plt.subplots_adjust(bottom=0.2)
        v_min = 0
        v_max = len(antpos)
        color_scale_antpos = np.linspace(0,1,len(antpos))
        
        if diameter is None:
            radius = 5
        else:
            radius = diameter / 2
        
        ax[1,0].set_xlim([1.1 * np.min(antpos[:,0]), 1.1 * np.max(antpos[:,0])])
        ax[1,0].set_ylim([1.1 * np.min(antpos[:,1]), 1.1 * np.max(antpos[:,1])])
        marker_size = (radius * 72 / fig.dpi) ** 2
        array_scatter_plot = ax[1,0].scatter(antpos[:,0],antpos[:,1],s=marker_size, c=colormap(color_scale_antpos))
        ax[1,0].set_title(f'Array ({len(antpos)} antennas)')
        ax[1,0].set_xlabel(r'EW [m]')
        ax[1,0].set_ylabel(r'NS [m]')
        
        
        
        cbar = fig.colorbar(array_scatter_plot, ax=ax[0,0],orientation='horizontal', pad=0.2)
        cbar.set_label('Antenna rank')  # Set label for the color bar
        cbar.set_ticks([0, 1])  # Optionally, set custom ticks
        cbar.set_ticklabels([v_min, v_max])
        
        ax[0,1].plot(np.arange(len(n_new_fulfilled_list)) + 1, n_new_fulfilled_list,color='b')
        ax[0,1].set_ylabel('Number of newly fulfilled points',color='b')
        ax[0,1].set_xlabel('Antenna rank')
        ax[0,1].set_xlim([1,len(n_new_fulfilled_list)])
        ax[0,1].grid()
        
        ax_remaining = ax[0,1].twinx()
        ax_remaining.plot(np.arange(len(n_not_fulfilled_list)) + 1,n_not_fulfilled_list,color='r')
        ax_remaining.set_ylabel('Number of commanded points that\nremain to be fulfilled',color='r')
        for i in range(2):
            ax[i,0].set_aspect('equal', adjustable='box')
            
        if step_time_array is not None:
            ax[1,1].plot(np.arange(len(step_time_array))[2:]+1,step_time_array[2:], color = 'b')
            ax[1,1].set_xlabel('Antenna rank')
            ax[1,1].set_ylabel('Time [sec]', color = 'b')
            ax[1,1].set_xlim([1,len(step_time_array)])
            ax[1,1].grid()
        if efficiency_array is not None:
            ax_efficiency = ax[1,1].twinx()
            ax_efficiency.plot(np.arange(len(efficiency_array))[1:] + 1, efficiency_array[1:], color='r')
            ax_efficiency.set_ylabel('Antenna count efficiency', color ='r')
        
    return fig,ax
    
def collision_check(antpos,diameter):
    # Returns true if there is a collision
    if diameter is None:
        return False
    else:
        return get_min_distance(antpos)<=diameter
check_collision = collision_check # just an alias because sometimes I mix them up


def get_min_distance(antpos):
    # Returns the minimum distance between any two points
    return np.min(pdist(antpos, 'euclidean'))

def get_min_distance_from_new_antpos(antpos, new_antpos):
    # Returns the minimum distance between an array that is currently antpos and a new point
    return np.min(np.linalg.norm(antpos-new_antpos,axis=1))
    
def get_array_size(antpos):
    if len(antpos.shape)==1:
        return 0
    # Returns the physical size of the array
    return np.max(pdist(antpos, 'euclidean'))
get_max_distance = get_array_size # just an alias because "get_max_distance" follows the "get_min_distance" convention


# RETIRE THIS, TOO CONFUSING
def get_n_new_fulfilled(new_antpos,antpos,not_fulfilled,fulfill_tolerance, p_norm =  np.inf):
# RETIRE THIS, TOO CONFUSING
    # Returns how many new commanded uvs are fulfilled when adding new_ant to antpos
    
    # Compute new uvs
    new_uvs = antpos - new_antpos
    new_uvs *= nonzero_sign(new_uvs[:,1].reshape(-1,1))
    idx_v0 = new_uvs[:, 1] == 0
    new_uvs[idx_v0, 0] = np.abs(new_uvs[idx_v0, 0])
    
    
    # Create a KD-tree
    tree = cKDTree(new_uvs)
    
    # Count how many are close
    counts = tree.query_ball_point(not_fulfilled, r=fulfill_tolerance, p = p_norm)
    n_new_fulfilled = sum(len(count) > 0 for count in counts)

    return n_new_fulfilled

def get_new_fulfilled_old(new_antpos,antpos,not_fulfilled,fulfill_tolerance,p_norm = np.inf):
    # Returns how many new commanded uvs are fulfilled when adding new_ant to antpos
    
    # Compute new uvs
    new_uvs = antpos - new_antpos
    new_uvs *= nonzero_sign(new_uvs[:,1].reshape(-1,1))
    idx_v0 = new_uvs[:, 1] == 0
    new_uvs[idx_v0, 0] = np.abs(new_uvs[idx_v0, 0])
    
    
    # Create a KD-tree
    tree = cKDTree(new_uvs)
    
    # Count how many are close
    counts = tree.query_ball_point(not_fulfilled, r=fulfill_tolerance, p = p_norm)
    n_new_fulfilled = sum(len(count) > 0 for count in counts)

    new_fulfilled = np.array([np.squeeze(new_uvs[count[0]]) for count in counts if len(count)>0])
    
    return n_new_fulfilled,new_fulfilled

def get_new_fulfilled(new_antpos,antpos,not_fulfilled,fulfill_tolerance,p_norm = np.inf):
    # Returns how many new commanded uvs are fulfilled when adding new_ant to antpos
    
    # Compute new uvs
    new_uvs = antpos - new_antpos
    new_uvs *= nonzero_sign(new_uvs[:,1].reshape(-1,1))
    idx_v0 = new_uvs[:, 1] == 0
    new_uvs[idx_v0, 0] = np.abs(new_uvs[idx_v0, 0])
    
    
    # Create a KD-tree
    tree = cKDTree(new_uvs)
    
    # Count how many are close
    counts = tree.query_ball_point(not_fulfilled, r=fulfill_tolerance, p = p_norm)
    new_fulfilled = np.array([np.squeeze(new_uvs[count[0]]) for count in counts if len(count)>0])
    
    return new_fulfilled


def get_new_fulfilled_grid(new_antpos_grid,antpos_grid,not_fulfilled_grid):
    # Returns how many new commanded uvs are fulfilled when adding new_ant to antpos
    
    # Compute new uvs
    new_uvs_grid = antpos_grid - new_antpos_grid
    new_uvs_grid *= nonzero_sign(new_uvs_grid[:,1].reshape(-1,1))
    idx_v0_grid = new_uvs_grid[:, 1] == 0
    new_uvs_grid[idx_v0_grid, 0] = np.abs(new_uvs_grid[idx_v0_grid, 0])
    
    new_fulfilled_grid = np.intersect1d(new_uvs_grid('i,i'),not_fulfilled_grid('i,i'))
    new_fulfilled_grid = new_fulfilled_grid.view(new_uvs_grid.dtype).reshape(-1,2)
    
    return new_fulfilled_grid

# RETIRED, TRY TO AVOID USING
def check_fulfillment_old(commanded, antpos, fulfill_tolerance, p_norm = np.inf, flip_tolerance = 0.0):
    # Returns the number of fulfilled and unfulfilled points, along with the corresponding arrays

    if len(antpos.shape)<2 or len(antpos)<2:
        return 0, len(commanded), np.array([]), commanded
    
    else:
        antpos_uvs = antpos_to_uv(antpos,flip_tolerance=flip_tolerance)
        
        # Build a KD-tree for antpos_uvs
        tree = cKDTree(antpos_uvs)
        
        # Find indices of commanded points that are within the threshold distance of points in antpos_uvs
        idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p = p_norm)
        
        # Determine which points in commanded are close to any in antpos_uvs
        fulfilled_mask = np.asarray([bool(idx) for idx in idx_fulfilled])
    
        # Determine far antpos as those not close to any point in antpos_uvs
        fulfilled = commanded[fulfilled_mask]
        not_fulfilled = commanded[~fulfilled_mask]
        
        return len(fulfilled), len(not_fulfilled), fulfilled, not_fulfilled
    
def check_fulfillment(commanded, antpos, fulfill_tolerance, p_norm = np.inf, flip_tolerance = 0.0):
    # Returns the number of fulfilled and unfulfilled points, along with the corresponding arrays

    if len(antpos.shape)<2 or len(antpos)<2:
        return np.array([]), commanded
    
    else:
        antpos_uvs = antpos_to_uv(antpos,flip_tolerance=flip_tolerance)
        
        # Build a KD-tree for antpos_uvs
        tree = cKDTree(antpos_uvs)
        
        # Find indices of commanded points that are within the threshold distance of points in antpos_uvs
        idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p = p_norm)
        
        # Determine which points in commanded are close to any in antpos_uvs
        fulfilled_mask = np.asarray([bool(idx) for idx in idx_fulfilled])
    
        # Determine far antpos as those not close to any point in antpos_uvs
        fulfilled = commanded[fulfilled_mask]
        not_fulfilled = commanded[~fulfilled_mask]
        
        return fulfilled, not_fulfilled
    
    
def check_fulfillment_idx(commanded, antpos, fulfill_tolerance, p_norm=np.inf, flip_tolerance=0.0, verbose = False):
    # Returns the indices of fulfilled and unfulfilled points

    if len(antpos.shape) < 2 or len(antpos) < 2:
        return np.array([]), np.arange(len(commanded))
    else:
        antpos_uvs = antpos_to_uv(antpos, flip_tolerance=flip_tolerance)

        # Build a KD-tree for antpos_uvs
        tree = cKDTree(antpos_uvs)

        # Find indices of commanded points that are within the threshold distance of points in antpos_uvs
        idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p=p_norm)

        # Determine which points in commanded are close to any in antpos_uvs
        fulfilled_mask = np.asarray([bool(idx) for idx in idx_fulfilled])

        # Get the indices of fulfilled and not_fulfilled points
        fulfilled_indices = np.where(fulfilled_mask)[0]
        not_fulfilled_indices = np.where(~fulfilled_mask)[0]
        
        if verbose:
            print(f'{len(fulfilled_indices)}/{len(commanded)} fulfilled, {len(not_fulfilled_indices)}/{len(commanded)} remaining.')
        
        return fulfilled_indices, not_fulfilled_indices


def antpos_to_uv(antpos, flip_tolerance = 1e-5, unique_only = False, fill_all_plane = False):
    # Returns the uv points from a given array of antenna positions
    
    n_ants = len(antpos)
    n_bls = int(n_ants*(n_ants-1)/2)
    uv_points = [None] * ((1 + fill_all_plane) * n_bls)
    # Generate all pairwise combinations of antenna positions
    i_pair = 0
    for pos1, pos2 in itertools.combinations(antpos, 2):
        u = pos2[0] - pos1[0]
        v = pos2[1] - pos1[1]
        
        if v>flip_tolerance or (np.abs(v)<=flip_tolerance and u>=0):
            uv_point = [u, v]
        else:
            uv_point = [-u, -v]
    
        uv_points[i_pair] = uv_point
        if fill_all_plane:
            uv_points[i_pair + n_bls] = [-uv_point[0],-uv_point[1]]
        i_pair+=1
        
    if unique_only:
        return np.unique(np.asarray(uv_points),axis=0)
    else:
        return np.asarray(uv_points)


def initialize_ref_wl(ref_wl = None, ref_freq = None, verbose = False):
    # Initialize the reference wavelength and frequency    

    if ref_freq is None and ref_wl is None:
        if verbose:
            print('Using default frequency of 150 MHz.')
        ref_freq = 150e6
        ref_wl = constants.c.value / ref_freq
    elif ref_freq is None and ref_wl is not None:
        ref_freq = constants.c.value / ref_wl
    elif ref_freq is not None and ref_wl is None:
        ref_wl = constants.c.value / ref_freq
    else:
        if verbose:
            print('Cannot set both ref_freq and ref_wl. Using ref_freq.')
        ref_wl = constants.c.value / ref_freq
    return ref_wl, ref_freq

def initialize_bl_range(min_bl = None, max_bl = None, min_bl_lambda = None, max_bl_lambda = None, ref_wl = 1, verbose = False):
    # Initialize range of baselines
    
    if min_bl_lambda is None and min_bl is None:
        if verbose:
            print('Using default min_bl_lambda of 10.')
        min_bl_lambda = 10
        min_bl = min_bl_lambda * ref_wl
    elif min_bl_lambda is None and min_bl is not None:
        min_bl_lambda = min_bl/ref_wl
    elif min_bl_lambda is not None and min_bl is None:
        min_bl = min_bl_lambda * ref_wl
    else:
        if verbose:
            print('Cannot set both min_bl_lambda and min_bl. Using min_bl_lambda.')
        min_bl = min_bl_lambda * ref_wl
        
        
    if max_bl_lambda is None and max_bl is None:
        if verbose:
            print('Using default max_bl_lambda of 100.')
        max_bl_lambda = 100
        max_bl = max_bl_lambda * ref_wl
    elif max_bl_lambda is None and max_bl is not None:
        max_bl_lambda = max_bl/ref_wl
    elif max_bl_lambda is not None and max_bl is None:
        max_bl = max_bl_lambda * ref_wl
    else:
        if verbose:
            print('Cannot set both max_bl_lambda and max_bl. Using max_bl_lambda.')
        max_bl_lambda = max_bl_lambda
        max_bl = max_bl_lambda * ref_wl
        
    return min_bl,max_bl,min_bl_lambda,max_bl_lambda    


def initialize_diameter(diameter = None, diameter_lambda = None, ref_wl = 1, verbose = False):
    if diameter_lambda is None and diameter is None:
        if verbose:
            print('Using default diameter of 10 m.')
        diameter = 10
        diameter_lambda = diameter / ref_wl
    elif diameter_lambda is None and diameter is not None:
        diameter_lambda = diameter/ref_wl
    elif diameter_lambda is not None and diameter is None:
        diameter_lambda = diameter_lambda
        diameter = diameter_lambda * ref_wl
    else:
        if verbose:
            print('Cannot set both diameter_lambda and diameter. Using diameter.')
        diameter_lambda = diameter / ref_wl
    
    return diameter, diameter_lambda

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

def generate_commanded_grid(uv_cell_size = 0.5, min_bl_lambda=10, max_bl_lambda=100, show_plot = True, ax = None):
    # This function isn't complete yet...
    print('WARNING! You probably mean to use generate_commanded_points()')
    uv_points = []
    
    max_bl_grid = max_bl_lambda / uv_cell_size
    min_bl_grid = min_bl_lambda / uv_cell_size
    
    max_u_grid = max_bl_grid
    min_u_grid= -max_bl_grid
    max_v_grid = max_bl_grid
    min_v_grid = 0
    
    for u in np.arange(min_u_grid, max_u_grid + 1, 1):
        for v in np.arange(min_v_grid, max_v_grid + 1, 1):
            distance = np.sqrt(u**2 + v**2)
            if min_bl_grid < distance <= max_bl_grid:
                if not (u<0 and v==0):
                    #print(u,v)
                    uv_points.append((u, v))
    uv_points = np.asarray(uv_points)
    if show_plot:
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=[10,5])
        ax.plot(uv_points[:,0],uv_points[:,1],'.',markersize=1,color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Commanded grid points')
        ax.set_xlim(min_u_grid-1, max_u_grid+1)
        ax.set_ylim(min_v_grid-1, max_v_grid+1)
        ax.set_xlabel(r'$u$ [uv_cell_size]')
        ax.set_ylabel(r'$v$ [uv_cell_size]')
        ax.grid()
        
        
    uv_points = uv_points[np.argsort(np.linalg.norm(uv_points, axis=1))]
    
    #uv_dict = {i:uv_points[i] for i in range(len(uv_points))}
    
    return uv_points#uv_dict

def generate_uv_random(uv_cell_size, r_min, r_max, fulfill_tolerance):
    # Returns a grid of uv points in a half annulus, randomly distributed.
    # Calculate the area of the half-ring
    area_half_ring = 0.5 * np.pi * (r_max**2 - r_min**2)
    area_per_point = np.pi * (uv_cell_size/2)**2
    n_points = area_half_ring / area_per_point
    density = n_points / area_half_ring

    # Calculate the number of points based on the density and area
    n = int(area_half_ring * density)

    points = []
    while len(points) < n:
        # Generate a new point uniformly distributed by area
        u = np.random.uniform(0, 1)
        r = np.sqrt(r_min**2 + (r_max**2 - r_min**2) * u)
        theta = np.random.uniform(0, np.pi)

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Check if the new point satisfies the distance constraint
        if len(points) == 0 or all(np.linalg.norm(np.array(points) - [x, y], axis=1) >= fulfill_tolerance):
            points.append([x, y])

    return np.array(points)


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
    array_size = get_array_size(antpos)
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
            if not collision_check(temp_shuffled_antpos,diameter):
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

def make_fake_commanded(antpos,commanded,verbose = False):
    # This function takes an antpos and generated a set of points that would
    # have commanded that antpos. Maybe a better name is "reverse-engineer"
    # commanded.
    if verbose:
        print('Getting the uvs... ',end='')
    uvs = antpos_to_uv(antpos,unique_only = False)
    if verbose:
        print('done. Getting the min_bl... ',end='')
    min_bl = np.min(np.linalg.norm(commanded,axis=1))
    if verbose:
        print('done. Getting the max_bl... ',end='')
    max_bl = np.max(np.linalg.norm(commanded,axis=1))
    if verbose:
        print('done. Getting the norms... ',end='')
    uvs_norms = np.linalg.norm(uvs, axis = 1)
    if verbose:
        print('done. Getting the mask... ',end='')
    mask = (uvs_norms >= min_bl) & (uvs_norms <= max_bl)
    if verbose:
        print('done. Returning.')
    return uvs[mask]
    
def nudge_antpos(antpos,diameter,fulfill_tolerance):
    for i in range(len(antpos)):
        success = False
        while not success:
            nudge_u = 2 * (np.random.random() - 0.5) * fulfill_tolerance
            nudge_v = 2 * (np.random.random() - 0.5) * fulfill_tolerance
            temp_antpos = copy.deepcopy(antpos)
            temp_antpos[i,0]+=nudge_u
            temp_antpos[i,1]+=nudge_v
            if not collision_check(temp_antpos,diameter):
                antpos = temp_antpos
                success = True
    return antpos


def get_n_baselines_involved(antpos,commanded, fulfill_tolerance):
    n_baselines_involved = np.zeros(len(antpos),dtype = int)
    for i in range(len(antpos)):
        n_baselines_involved[i] = len(get_new_fulfilled(antpos[i], antpos, commanded,fulfill_tolerance))
    return n_baselines_involved
