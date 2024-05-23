#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from IPython.display import clear_output, display
from uvComplete.utils import check_fulfillment, get_array_size, get_new_fulfilled, get_antpos_history,plot_array,collision_check


def create_array_random(n=200, commanded = None, built=None, diameter=8.54,max_array_size=300, fulfill_tolerance = 0.5, always_add = False, show_plot = True, show_plot_skip = 10, verbose = True,max_failed_attempts = 1e5, random_seed = 11141):
    # Initialize built array with a single random point
    np.random.seed(random_seed)
    i_loop = 0
    
    if show_plot:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
    
    if commanded is not None:
        try_fulfill = True
        n_not_fulfilled = commanded.shape[0]
        min_not_fulfilled = n_not_fulfilled
        best_loop = 0
    else:
        print(f'No commanded points passed, just generating a random array of {n} points.')
        try_fulfill = False
        n_not_fulfilled = -1
        
    success_whole_array = False
    
    if built is not None:
        built = np.asarray(built)
        built_saved = np.copy(built)
        starting_from_scratch = False
        
        if not (built.shape == (2,) or (len(built.shape) == 2 and built.shape[1] == 2)):
            print('Incorrect passed built array, using [0,0].')
            built = np.asarray([0,0])
            starting_from_scratch = True
    else:
        starting_from_scratch = True    


    
    printout_condition=False
    while success_whole_array==False:
        
        i_loop +=1
        n_failed_attempts = 0
        if starting_from_scratch:
            print('Starting from scratch with built = [0,0]...')
            built = np.asarray([0,0])
            if try_fulfill:
                built = np.vstack([built, commanded[0]])
            else:
                rand_u = np.random.uniform(diameter,max_array_size)
                rand_v = np.random.uniform(diameter,max_array_size)
                built = np.vstack([built, np.asarray([rand_u,rand_v])])
        else:
            built = built_saved
        
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
        n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_antpos_history(commanded, built, fulfill_tolerance)
        tree = cKDTree(built)
        
        while True:
                        
            n_not_fulfilled = 0
            if not try_fulfill:
                if built.shape[0]>=n:
                    success_whole_array = True
                    break
            else:
                #n_min_possible = (1 + (8 * commanded.shape[0] + 1 ) ** .5 ) / 2
                if built.shape[0]>1:
                    _, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built,fulfill_tolerance)
                    if n_not_fulfilled == 0:
                        success_whole_array = True
                        break
            success_new_dish = False
            if try_fulfill and built.shape[0]>1:
                printout_condition = built.shape[0]%10==0 or (not_fulfilled.shape[0]<1000)
            while not success_new_dish:
                # Generate a new antpos at a random direction and distance
                
                new_antpos = np.asarray([(2*np.random.random()-1)*max_array_size/2, (2*np.random.random()-1)*max_array_size/2])
                if tree.query(new_antpos)[0] >= diameter and np.linalg.norm(new_antpos)<=max_array_size/2:
                    # Add the new antpos if it passes all checks and we're at the beginning
                
                    n_new_fulfilled,new_fulfilled = get_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance)
                    if n_new_fulfilled>0 or not try_fulfill or not always_add:
                        # Rebuild KDTree with the new antpos
                        success_new_dish = True
                        n_failed_attempts = 0
                        built = np.vstack([built,new_antpos])
                        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)       
                        n_new_fulfilled_list.append(n_new_fulfilled)
                        n_not_fulfilled_list.append(n_not_fulfilled)
                        new_fulfilled_list.append(new_fulfilled)
                        if n_not_fulfilled < min_not_fulfilled:
                            min_not_fulfilled = n_not_fulfilled
                            best_loop = i_loop
                        tree = cKDTree(built)
                else:
                    n_failed_attempts+=1
                    if n_failed_attempts>0 and n_failed_attempts % 100000 == 0:
                        print(f"Failed attempts: {n_failed_attempts}/{max_failed_attempts}")

                if n_failed_attempts >= max_failed_attempts:
                    print(f"Maximum number of failed attempts reached ({max_failed_attempts}), no spot for new dish found, starting over (breaking out of inner loop).")
                    break

            
            if len(built) % 100 == 0 and not try_fulfill:
                print(f"Generated {built.shape[0]}/{n} antennas...")

            if n_failed_attempts >= max_failed_attempts:
                print(f"Maximum number of failed attempts reached ({max_failed_attempts}), no spot for new dish found, starting over (breaking out of outer loop).")
                break
            
            
            
            if show_plot and built.shape[0]%show_plot_skip==0:
                clear_output(wait=True)
                #plt.pause(0.01)
                fig,ax=plot_array(built,commanded,fulfill_tolerance,just_plot_array=False,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = fulfilled, not_fulfilled = not_fulfilled)
                display(fig)
            if verbose and show_plot and built.shape[0]%show_plot_skip==0:
                n_new_fulfilled = n_new_fulfilled_list[-1]
                print('{:d} newly fulfilled points'.format(n_new_fulfilled))
                print('{:d} total antennas built'.format(built.shape[0]))
                print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
                print(f'On loop {i_loop}.')
                print(f'Best loop is still loop {best_loop}, with {min_not_fulfilled}/{commanded.shape[0]} only left to be fulfilled.')
            if verbose and not show_plot:
                if printout_condition:
                    clear_output(wait=True)
                    n_new_fulfilled = new_fulfilled_list[-1]
                    print('{:d} newly fulfilled points'.format(n_new_fulfilled))
                    print('{:d} total antennas built'.format(built.shape[0]))
                    print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
                    print(f'Best loop is still loop {best_loop}, with {min_not_fulfilled}/{commanded.shape[0]} only left to be fulfilled.')
                    
        if n_not_fulfilled <= min_not_fulfilled:
            min_not_fulfilled = n_not_fulfilled
            best_loop = i_loop
            np.save('built_random'+always_add*'_aa_'+'.npy',built)
                    

            
    print('Done.')
    print('Array size is now: {:.2f} wavelengths'.format(get_array_size(built)))
    print('{:d} total antennas built'.format(built.shape[0]))
    print(f'It took {i_loop} loops to complete the array.')
    if try_fulfill:
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built,fulfill_tolerance)
        print('{:d}/{:d} commanded points remain to be fulfilled'.format(n_not_fulfilled,commanded.shape[0]))
    plt.close()
    return built


def create_array_random_on_grid(n=0, commanded = None, built = None, diameter=8.54, max_array_size=300, fulfill_tolerance=0.5,always_add = False, max_trials_for_new_fulfill = 10, n_side_mesh=1000,show_plot = True,verbose = True,random_seed = 11141):

    np.random.seed(random_seed)
    
    array_min_x = -max_array_size / 2
    array_max_x = max_array_size / 2
    array_min_y = -max_array_size / 2
    array_max_y = max_array_size / 2
    
    x = np.linspace(array_min_x, array_max_x, n_side_mesh)
    y = np.linspace(array_min_y, array_max_y, n_side_mesh)
    xx, yy = np.meshgrid(x, y)
    
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Calculate distances from origin for each point
    distances = np.linalg.norm(grid_points, axis=1)
    
    # Find indices of points within the desired radius
    valid_indices = distances <= max_array_size / 2
    
    # Filter out points outside the desired radius
    grid_points = grid_points[valid_indices]
    grid_points_saved = np.copy(grid_points)

    print('Grid generated...')

    if commanded is not None:
        try_fulfill = True
        just_plot_array = False
        n_not_fulfilled = commanded.shape[0]
        min_not_fulfilled = n_not_fulfilled
        best_loop = 0
    else:
        print(f'No commanded points passed, just generating a random array of {n} points.')
        try_fulfill = False
        just_plot_array = True
        n_not_fulfilled = -1
        

    
    if built is not None:
        built = np.asarray(built)
        starting_from_scratch = False
        
        if not (built.shape == (2,) or (len(built.shape) == 2 and built.shape[1] == 2)):
            print('Incorrect passed built array, using [0,0].')
            built = np.asarray([0,0])
            starting_from_scratch = True
    else:
        built = np.asarray([0,0])
        starting_from_scratch = True    

    
    if verbose:
        print('Before even beginning, have:')
        print('{:d} antennas built'.format(built.shape[0]))
    
    

    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        built = np.vstack([built, commanded[0]])
    else:
        built_saved = np.copy(built)
    
    
    
    success_whole_array = False
        
    i_loop = 0
    
    
    while success_whole_array == False:
        
        i_loop +=1
        
        grid_points = np.copy(grid_points_saved)
        if starting_from_scratch:
            built = np.vstack(np.asarray([[0,0]]))
        else:
            built = np.copy(built_saved)
        
        
        if len(built.shape)<2 or built.shape[0]<2:
            remove_mask = np.linalg.norm(grid_points,axis=1)<diameter
            grid_points = grid_points[~remove_mask]
        else:
            tree = cKDTree(built)
            points_within_diameter = tree.query_ball_point(grid_points, r=diameter)
            remove_mask = np.array([bool(idx) for idx in points_within_diameter])
            grid_points = grid_points[~remove_mask]
                    

        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
        n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_antpos_history(commanded, built, fulfill_tolerance)
        
        grid_points_left_list = [grid_points.shape[0]]
        
        while True:
            n_trials_for_new_fulfill = 0
            if grid_points.shape[0]>1:
                i = int(round(np.random.uniform() * (grid_points.shape[0] - 1)))
                new_antpos = grid_points[i]
            else:
                new_antpos = grid_points[0]
            if try_fulfill and not always_add:
                n_new_fulfilled,new_fulfilled = get_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance)
                if n_new_fulfilled>0 or (n_trials_for_new_fulfill > 10 and max_trials_for_new_fulfill != -1):
                    if n_trials_for_new_fulfill > 10:
                        print(f"Couldn't fulfill a new commanded uv point after {n_trials_for_new_fulfill}, adding the antenna anyway.")
                    if n_trials_for_new_fulfill > 0:
                        print(f"On attempt {n_trials_for_new_fulfill} to fulfill a new commanded uv point...")
                    built = np.vstack([built,new_antpos])
                    distances = np.linalg.norm(grid_points[:, None] - new_antpos.reshape(1,-1), axis=2)
                    within_distance = np.any(distances < diameter, axis=1)
                    grid_points = grid_points[~within_distance]
                    grid_points_left_list.append(grid_points.shape[0])
                    n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)         
                    n_new_fulfilled_list.append(n_new_fulfilled)
                    n_not_fulfilled_list.append(n_not_fulfilled)
                    new_fulfilled_list.append(new_fulfilled)  
                    if n_not_fulfilled < min_not_fulfilled:
                        min_not_fulfilled = n_not_fulfilled
                        best_loop = i_loop
                else:
                    n_trials_for_new_fulfill +=1
            else:
                built = np.vstack([built,grid_points[i]])
                distances = np.linalg.norm(grid_points[:, None] - new_antpos.reshape(1,-1), axis=2)
                within_distance = np.any(distances < diameter, axis=1)
                grid_points = grid_points[~within_distance]
                grid_points_left_list.append(grid_points.shape[0])
                if try_fulfill:
                    n_new_fulfilled,new_fulfilled = get_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance)
                    n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)          
                    n_new_fulfilled_list.append(n_new_fulfilled)
                    n_not_fulfilled_list.append(n_not_fulfilled)
                    new_fulfilled_list.append(new_fulfilled) 
                    if n_not_fulfilled < min_not_fulfilled:
                        min_not_fulfilled = n_not_fulfilled
                        best_loop = i_loop
            
            if built.shape[0]>=n and not try_fulfill:
                print('Built the required number of antennas.')
                success_whole_array = True
                break
            elif n_not_fulfilled==0:
                print('Fulfilled all commanded baselines.')
                success_whole_array = True
                break
            
            elif grid_points.shape[0]<=1:
                print('No more grid points.')
                print(f'{built.shape[0]} antennas placed, {n_not_fulfilled} commanded baselines yet to be fulfilled.')
                break
            if built.shape[0]%10==0 or grid_points.shape[0]<20 and verbose:
                
                if show_plot:
                    clear_output(wait=True)
                    #plt.pause(0.01)
                    fig,ax=plot_array(built,commanded,fulfill_tolerance,just_plot_array=just_plot_array,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = fulfilled, not_fulfilled = not_fulfilled)
                    display(fig)
                print(f'On loop {i_loop}')
                print('{:d} newly fulfilled points'.format(n_new_fulfilled))
                print('{:d} total antennas built'.format(built.shape[0]))
                print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
                print(f'{grid_points.shape[0]} grid points left...')
                print(f'Best loop is still loop {best_loop}, with {min_not_fulfilled}/{commanded.shape[0]} only left to be fulfilled.')
    
            
        if n_not_fulfilled <= min_not_fulfilled:
            min_not_fulfilled = n_not_fulfilled
            best_loop = i_loop
            np.save('built_random_grid'+always_add*'_aa_'+'.npy',built)
        
    print('Array size is now: {:.2f} wavelengths'.format(get_array_size(built)))
    print('{:d} total antennas built'.format(built.shape[0]))
    if try_fulfill:
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built,fulfill_tolerance)
        print('{:d}/{:d} commanded points remain to be fulfilled'.format(n_not_fulfilled,commanded.shape[0]))
    return built


def create_array_truly_random(n=200, diameter=8.54,max_array_size=300, show_plot = True, show_plot_skip = 10, verbose = True, random_seed = 11141, max_failed_attempts = 100000):
    # Initialize built array with a single random point
    np.random.seed(random_seed)
    i_loop = 1
    n_failed_attempts = 0
    
    if show_plot:
        fig,ax = plt.subplots(1,1,figsize=(5,5))
    
    
    
    success_whole_array = False

    while not success_whole_array:
        
        built = np.asarray([0,0])
        rand_u = np.random.uniform(diameter,max_array_size)
        rand_v = np.random.uniform(diameter,(max_array_size**2 - rand_u**2)**.5)
        built = np.vstack([built, np.asarray([rand_u,rand_v])])
        
        if verbose:
            print('Before even beginning, have:')
            print('{:d} antennas built'.format(built.shape[0]))
        
        while True:
            if built.shape[0]>=n:
                success_whole_array=True
                break
            else:
                tree = cKDTree(built)
                rand_u = np.random.uniform(-max_array_size,max_array_size)
                rand_v = np.random.uniform(-max_array_size,max_array_size)
                new_antpos = np.asarray([rand_u,rand_v])
                if tree.query(new_antpos)[0] >= diameter and np.linalg.norm(new_antpos)<=max_array_size/2 and not collision_check(np.vstack([built,new_antpos]),diameter):
                    built = np.vstack([built,new_antpos])
                    if show_plot and built.shape[0]%show_plot_skip==0:
                        clear_output(wait=True)
                        #plt.pause(0.01)
                        fig,ax=plot_array(built,just_plot_array=True)
                        display(fig)
                    if verbose and show_plot and built.shape[0]%show_plot_skip==0:
                       print('{:d}: {:d} total antennas built'.format(i_loop,built.shape[0]))
                    if verbose and not show_plot:
                            print('{:d}: {:d} total antennas built'.format(i_loop,built.shape[0]))
                else:
                    if n_failed_attempts%1000==0:
                        print(".",end='')
                    n_failed_attempts += 1
            if n_failed_attempts>=max_failed_attempts:
                print(f'Max failed attempts reached ({max_failed_attempts}), starting over.')
                n_failed_attempts = 0
                i_loop+=1
                clear_output(wait=True)
                break


        
    print('Done.')
    print('Array size is now: {:.2f} wavelengths'.format(get_array_size(built)))
    print('{:d} total antennas built'.format(built.shape[0]))
    plt.close()
    return built
