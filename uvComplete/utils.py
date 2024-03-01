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




def get_new_fulfilled_list(commanded, built, fulfill_tolerance):
    n_new_fulfilled_list = []
    n_not_fulfilled_list = []
    new_fulfilled_list = []
    if built.shape != (2,):
        if built.shape==2:
            n_new_fulfilled_temp, new_fulfilled_temp = get_new_fulfilled(built[1],built[0],commanded,fulfill_tolerance)
            n_new_fulfilled_list.append(n_new_fulfilled_temp)
            new_fulfilled_list.append(new_fulfilled_temp)
        elif built.shape[0]>2:
            not_fulfilled_temp = np.copy(commanded)
            for i in range(built.shape[0])[3:]:
                built_temp = built[:i-1]
                n_new_fulfilled_temp, new_fulfilled_temp = get_new_fulfilled(built[i],built_temp,not_fulfilled_temp,fulfill_tolerance)
                n_new_fulfilled_list.append(n_new_fulfilled_temp)
                new_fulfilled_list.append(new_fulfilled_temp)
                _, n_not_fulfilled_temp, _, not_fulfilled_temp = check_fulfillment(not_fulfilled_temp,built_temp, fulfill_tolerance)
                n_not_fulfilled_list.append(n_not_fulfilled_temp)
            print('Done, adding the remaining antennas.')
    return n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list

def plot_array(built, commanded = None, fulfill_tolerance = 0.5, just_plot_array = True,fig=None,ax=None,n_new_fulfilled_list = None,n_not_fulfilled_list = None,new_fulfilled_list = None, fulfilled = None, not_fulfilled = None):
    
    if commanded is None:
        just_plot_array=True
        print('No commanded array passed, will just plot the array.')
    
    if just_plot_array:
        if fig is None and ax is None:
            fig,ax = plt.subplots(1,1,figsize = [5,5])
        ax.plot(built[:,0],built[:,1],'.')
        ax.set_xlabel(r'EW [$\lambda$]')
        ax.set_ylabel(r'NS [$\lambda$]')
        ax.set_aspect('equal')
        
    else:
        if fig is None and ax is None:
            fig,ax = plt.subplots(1,3,figsize=(18,6))
        if n_new_fulfilled_list is None or n_not_fulfilled_list is None or new_fulfilled_list is None:
            n_new_fulfilled_list,n_not_fulfilled_list, new_fulfilled_list = get_new_fulfilled_list(commanded, built,fulfill_tolerance)
        if fulfilled is None and not_fulfilled is None:
            _,_,fulfilled,not_fulfilled = check_fulfillment(commanded,built,fulfill_tolerance)
        
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
        v_max = built.shape[0]
        color_scale_built = np.linspace(0,1,built.shape[0])
        array_scatter_plot = ax[1].scatter(built[:,0],built[:,1],c=colormap(color_scale_built))
        ax[1].set_title('Array')
        ax[1].set_xlabel(r'EW [$\lambda$]')
        ax[1].set_ylabel(r'NS [$\lambda$]')
        
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
    
def collision_check(built,diameter):
    # Returns true if there is a collision
    return get_min_distance(built)<=diameter
check_collision = collision_check # just an alias because sometimes I mix them up



def get_min_distance(built):
    # Returns the minimum distance between any two points
    return np.min(pdist(built, 'euclidean'))

def get_min_distance_from_new_antpos(built, new_antpos):
    # Returns the minimum distance between an array that is currently built and a new point
    return np.min(np.linalg.norm(built-new_antpos,axis=1))
    
def get_array_size(built):
    # Returns the physical size of the array
    return np.max(pdist(built, 'euclidean'))
get_max_distance = get_array_size # just an alias because "get_max_distance" follows the "get_min_distance" convention

def get_n_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance, p_norm =  np.inf):
    # Returns how many new commanded uvs are fulfilled when adding new_ant to built
    
    # Compute new uvs
    new_uvs = built - new_antpos
    new_uvs *= nonzero_sign(new_uvs[:,1].reshape(-1,1))
    idx_v0 = new_uvs[:, 1] == 0
    new_uvs[idx_v0, 0] = np.abs(new_uvs[idx_v0, 0])
    
    
    # Create a KD-tree
    tree = cKDTree(new_uvs)
    
    # Count how many are close
    counts = tree.query_ball_point(not_fulfilled, r=fulfill_tolerance, p = p_norm)
    n_new_fulfilled = sum(len(count) > 0 for count in counts)

    return n_new_fulfilled

def get_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance,p_norm = np.inf):
    # Returns how many new commanded uvs are fulfilled when adding new_ant to built
    
    # Compute new uvs
    new_uvs = built - new_antpos
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

def check_fulfillment(commanded, built, fulfill_tolerance, p_norm = np.inf):
    # Returns the number of fulfilled and unfulfilled points, along with the corresponding arrays

    built_uvs = antpos_to_uv(built)
    
    # Build a KD-tree for built_uvs
    tree = cKDTree(built_uvs)
    
    # Find indices of commanded points that are within the threshold distance of points in built_uvs
    idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p = p_norm)
    
    # Determine which points in commanded are close to any in built_uvs
    fulfilled_mask = np.asarray([bool(idx) for idx in idx_fulfilled])

    # Determine far antpos as those not close to any point in built_uvs
    fulfilled = commanded[fulfilled_mask]
    not_fulfilled = commanded[~fulfilled_mask]
    
    return fulfilled.shape[0], not_fulfilled.shape[0], fulfilled, not_fulfilled


def antpos_to_uv(antpos):
    # Returns the uv points from a given array of antenna positions
    
    n_ants = antpos.shape[0]
    n_bls = int(n_ants*(n_ants-1)/2)
    uv_points = [None] * n_bls
    # Generate all pairwise combinations of antenna positions
    i_pair = 0
    for pos1, pos2 in itertools.combinations(antpos, 2):
        # Compute the difference between positions to get the uv point
        u = pos2[0] - pos1[0]
        v = pos2[1] - pos1[1]
        if v>0:
            uv_points[i_pair] = [u, v]
        elif v==0:
            uv_points[i_pair] = [np.abs(u),v]
        else:
            uv_points[i_pair] = [-u, -v]
        i_pair+=1
    return np.asarray(uv_points)

def generate_uv_grid(uv_cell_size=1., min_u=-100, max_u=100, min_v=0, max_v=100, min_bl=10, max_bl=100, show_plot = True):
    # Returns a grid of antpos in a half annulus, without the (u<0,v=0) segment
    
    uv_points = []
    for u in np.arange(min_u, max_u + uv_cell_size, uv_cell_size):
        for v in np.arange(min_v, max_v + uv_cell_size, uv_cell_size):
            distance = np.sqrt(u**2 + v**2)
            if min_bl < distance <= max_bl:
                if not (u<0 and v==0):
                    #print(u,v)
                    uv_points.append((u, v))
    uv_points = np.asarray(uv_points)
    if show_plot:
        fig,ax = plt.subplots(1,1,figsize=[10,5])
        ax.plot(uv_points[:,0],uv_points[:,1],'.',markersize=1,color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Commanded points')
        ax.set_xlim(min_u-1, max_u+1)
        ax.set_ylim(min_v-1, max_v+1)
        ax.grid()
    return uv_points

