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
random_seed = 11141

def nonzero_sign(x):
    return np.where (x>=0,1,-1)


def pick_baselines(commanded, antpos, fulfill_tolerance):
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
                if len(idx_new_fulfilled[0])<1:
                    continue
                else:
                    not_fulfilled = np.delete(not_fulfilled,idx_new_fulfilled,axis=0)
                    antpairs.append((i,j))
                if len(not_fulfilled)%1000==0:
                    print(f'{len(not_fulfilled)/len(commanded)} remaining, antpairs now has length {len(antpairs)}.')
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
            for i in range(len(antpos))[3:]:
                antpos_temp = antpos[:i-1]
                new_fulfilled_temp = get_new_fulfilled(antpos[i],antpos_temp,not_fulfilled_temp,fulfill_tolerance)
                n_new_fulfilled_list.append(len(new_fulfilled_temp))
                new_fulfilled_list.append(new_fulfilled_temp)
                fulfilled_temp, not_fulfilled_temp = check_fulfillment(not_fulfilled_temp,antpos_temp, fulfill_tolerance)
                n_not_fulfilled_list.append(len(not_fulfilled_temp))
    return n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list

def plot_array(antpos, commanded = None, fulfill_tolerance = 0.5, just_plot_array = False, plot_new_fulfilled = False, fig=None,ax=None,n_new_fulfilled_list = None,n_not_fulfilled_list = None,new_fulfilled_list = None, fulfilled = None, not_fulfilled = None):
    
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
            fig,ax = plt.subplots(1,3,figsize=(18,6))
        if n_new_fulfilled_list is None or n_not_fulfilled_list is None or new_fulfilled_list is None:
            n_new_fulfilled_list,n_not_fulfilled_list, new_fulfilled_list = get_antpos_history(commanded, antpos,fulfill_tolerance)
        if fulfilled is None and not_fulfilled is None:
            fulfilled,not_fulfilled = check_fulfillment(commanded,antpos,fulfill_tolerance)
        
        colormap = cm.viridis
        ax[0].plot(commanded[:,0],commanded[:,1],'.',markersize=2,color='k',alpha=1,label='Commanded points',zorder=0)
        for i,new_fulfilled in enumerate(new_fulfilled_list):
            if len(new_fulfilled.shape)>1:
                ax[0].plot(new_fulfilled[:,0],new_fulfilled[:,1],'.',markersize=2,color=colormap(i/len(new_fulfilled_list)),zorder=1)
        #ax[0].scatter(fulfilled[:,0],fulfilled[:,1],c=colors_fulfilled,s=1,zorder=1)#,label='Fulfilled points')
        ax[0].set_title('uv plane')
        ax[0].set_xlabel(r'$u$')
        ax[0].set_ylabel(r'$v$')
        
        plt.subplots_adjust(bottom=0.2)
        v_min = 0
        v_max = len(antpos)
        color_scale_antpos = np.linspace(0,1,len(antpos))
        array_scatter_plot = ax[1].scatter(antpos[:,0],antpos[:,1],c=colormap(color_scale_antpos))
        ax[1].set_title(f'Array ({len(antpos)} antennas)')
        ax[1].set_xlabel(r'EW [m]')
        ax[1].set_ylabel(r'NS [m]')
        
        cbar_ax = fig.add_axes([ax[0].get_position().x0, 0.05, ax[1].get_position().x1 - ax[0].get_position().x0, 0.05])
        cbar = fig.colorbar(array_scatter_plot, cax=cbar_ax,orientation='horizontal', pad=0.2)
        cbar.set_label('Antenna rank')  # Set label for the color bar
        cbar.set_ticks([0, 1])  # Optionally, set custom ticks
        cbar.set_ticklabels([v_min, v_max])
        
        ax[2].plot(n_new_fulfilled_list,color='b')
        ax[2].set_ylabel('Number of newly fulfilled points',color='b')
        ax[2].set_xlabel('New antenna rank')
        ax[2].grid()
        
        ax_remaining = ax[2].twinx()
        ax_remaining.plot(n_not_fulfilled_list,color='r')
        ax_remaining.set_ylabel('Number of commanded points that\nremain to be fulfilled',color='r')
        for i in range(2):
            ax[i].set_aspect('equal', adjustable='box')
        
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
    
    
def check_fulfillment_idx(commanded, antpos, fulfill_tolerance, p_norm=np.inf, flip_tolerance=0.0):
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

        return fulfilled_indices, not_fulfilled_indices


def antpos_to_uv(antpos, flip_tolerance = 0.0, unique_only = False):
    # Returns the uv points from a given array of antenna positions
    
    n_ants = len(antpos)
    n_bls = int(n_ants*(n_ants-1)/2)
    if unique_only:
        uv_points = []
    else:
        uv_points = [None] * n_bls
    # Generate all pairwise combinations of antenna positions
    i_pair = 0
    for pos1, pos2 in itertools.combinations(antpos, 2):
        # Compute the difference between positions to get the uv point
        u = pos2[0] - pos1[0]
        v = pos2[1] - pos1[1]
        if v>flip_tolerance or (np.abs(v)<=flip_tolerance and u>=0):
            uv_point = [u, v]
        else:
            uv_point = [-u, -v]
        if unique_only:
            if not uv_point in uv_points:
                uv_points.append(uv_point)
        else:
            uv_points[i_pair] = uv_point
        i_pair+=1
        
        
    return np.asarray(uv_points)


def generate_uv_grid(uv_cell_size=1., min_bl=10, max_bl=100, show_plot = True, ax = None):
    # Returns a grid of uv points in a half annulus, without the (u<0,v=0) segment
    
    uv_points = []
    
    max_u = max_bl
    min_u= -max_bl
    max_v = max_bl
    min_v = 0
    
    for u in np.arange(min_u, max_u + uv_cell_size, uv_cell_size):
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

