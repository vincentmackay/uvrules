#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import itertools
import pickle
from uvComplete.utils import check_fulfillment, get_array_size, get_new_fulfilled, get_min_distance_from_new_antpos, collision_check,plot_array,get_new_fulfilled_list
from multiprocessing import Pool

class ExitLoopsException(Exception):
    pass

def create_array_rules(commanded, built=None, diameter = 8.54, max_array_size = 300, fulfill_tolerance = 0.5, order = -1, show_plot = True, save_file = False,save_name='rules', verbose = True, n_max_antennas = -1,within_bounds = False, check_first = 'built', check_all_built = True, check_all_not_fulfilled = False,show_plot_skip = 10, second_priority = 'array_size'):

    
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
    
    
    # sort by longest to shortest baseline
    commanded = commanded[np.argsort(order * np.linalg.norm(commanded, axis=1))]
    
    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        built = np.vstack([built, commanded[0]])
    
    # check fulfillment
    n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
    
    n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_new_fulfilled_list(commanded, built, fulfill_tolerance)
        

    not_fulfilled = not_fulfilled[np.argsort(order * np.linalg.norm(not_fulfilled, axis=1))]
    flips = np.asarray([-1,1])
    
    
    
    if check_all_built and check_all_not_fulfilled:
        print("It is not recommended to have both check_all_built and check_all_not_fulfilled set to True, this will take weeks. Use the parallelized create_array_rules_parallelized instead.")
    
    skip_built = (check_first == 'not_fulfilled' and not check_all_built and not check_all_not_fulfilled)
    i_skip = 0
    
    while(not_fulfilled.shape[0]>=1):
        if built.shape[0]==n_max_antennas:
            break
        success = False
        max_n_new_fulfilled = 0
        min_built_size = get_array_size(built)
        max_min_distance_from_new_antpos = 0
        
        # Decide which one will be the outer loop, and which one will be the inner loop
        if check_first == 'built':
            inner_loop = built
            outer_loop = not_fulfilled
            check_second = 'not_fulfilled'
        elif check_first == 'not_fulfilled':
            inner_loop = not_fulfilled
            outer_loop = built
            check_second = 'built'
        else:
            raise ValueError("Wrong value for check_first")
        
        
        
        # iterate over all combinations of built antennas and unfulfilled uv point
        for i in range(outer_loop.shape[0])[skip_built*i_skip:]:
            # iterate over all the not fulfilled baselines
            for j in range(inner_loop.shape[0]):
                # try both the positive and negative position
                for k in range(flips.shape[0]):
                    if check_first == 'built':
                        new_antpos = inner_loop[j] + flips[k] * outer_loop[i]
                    elif check_first == 'not_fulfilled':
                        new_antpos = outer_loop[i] + flips[k] * inner_loop[j]
                    if within_bounds:
                        if np.linalg.norm(new_antpos)>max_array_size/2:
                            continue
                    
                    # try and add one antenna at the longest commanded-but-not-fulfilled distance from point i
                    built_temp = np.vstack([built,new_antpos]) # 50 us
    
                    # check the size of the array
                    built_temp_size = get_array_size(built_temp) #241 us
                    
                    
                    # check if there's no collision and that we're still within max size
                    if (not collision_check(built_temp,diameter)) and built_temp_size<max_array_size: #245 us
                        success = True    
                        n_new_fulfilled,new_fulfilled = get_new_fulfilled(new_antpos,built,not_fulfilled,fulfill_tolerance)
                        min_distance_from_new_antpos = get_min_distance_from_new_antpos(built, new_antpos)
                        if n_new_fulfilled > max_n_new_fulfilled:
                            favored_new_antpos = new_antpos
                            max_n_new_fulfilled = n_new_fulfilled
                        elif n_new_fulfilled == max_n_new_fulfilled:
                            if second_priority == 'array_size':
                                if built_temp_size < min_built_size:
                                    favored_new_antpos = new_antpos
                                    min_built_size = built_temp_size
                                elif built_temp_size == min_built_size:
                                    if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                                        favored_new_antpos = new_antpos
                                        max_min_distance_from_new_antpos = min_distance_from_new_antpos
                            elif second_priority == 'min_distance_from_new_antpos':
                                if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                                    favored_new_antpos = new_antpos
                                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                                elif min_distance_from_new_antpos == max_min_distance_from_new_antpos:
                                    if built_temp_size < min_built_size:
                                        favored_new_antpos = new_antpos
                                        min_built_size = built_temp_size
                        if check_first == 'built' and not check_all_built:
                            if skip_built:
                                i_skip = i
                            break # breaks out of the flips loop
                        if check_first == 'not_fulfilled' and not check_all_not_fulfilled:
                            break # breaks out of the flips loop
                if success == True:
                    if check_first == 'built' and not check_all_built:
                        break # breaks out of the inner loop
                    if check_first == 'not_fulfilled' and not check_all_not_fulfilled:
                        break # breaks out of the inner loop
            if success == True:
                if check_second == 'built' and not check_all_built:
                    break # breaks out of the outer loop
                if check_second == 'not_fulfilled' and not check_all_not_fulfilled:
                    break # breaks out of the outer loop
        
            
        if success == False:
            print('Array is full for this set of parameters, quitting.')
            return built
            break
            
        
        n_new_fulfilled,new_fulfilled = get_new_fulfilled(favored_new_antpos,built,not_fulfilled,fulfill_tolerance)
        built = np.vstack([built,favored_new_antpos])
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
        not_fulfilled = not_fulfilled[np.argsort(order * np.linalg.norm(not_fulfilled, axis=1))]
        
        n_new_fulfilled_list.append(n_new_fulfilled)
        n_not_fulfilled_list.append(n_not_fulfilled)
        new_fulfilled_list.append(new_fulfilled)
        
        printout_condition = built.shape[0]%10==0 or (not_fulfilled.shape[0]<1000)
  
        if show_plot and (built.shape[0]%show_plot_skip==0 or not_fulfilled.shape[0]==0):
            
            clear_output(wait=True)
            if verbose:
                print('{:d} newly fulfilled points'.format(n_new_fulfilled))
                print('{:d} total antennas built'.format(built.shape[0]))
                print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
            #built_uvs = antpos_to_uv(built)
            fig,ax=plot_array(built,commanded,fulfill_tolerance,just_plot_array=False,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = fulfilled, not_fulfilled = not_fulfilled)
            plt.pause(0.01)
            
            #display(fig)
        if verbose and not show_plot:
            if printout_condition:
                clear_output(wait=True)
                print('{:d} newly fulfilled points'.format(n_new_fulfilled))
                print('{:d} total antennas built'.format(built.shape[0]))
                print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
                
    
    plt.close()

    if save_file:
        # Saving the variable to disk
        with open('built_'+save_name+'.pkl', 'wb') as file:
            pickle.dump(built, file)

    
    if verbose:
        array_size = get_array_size(built)
        print('Done.')
        print(f'Used {built.shape[0]} antennas to fulfill all baselines, array size is '+'{:.2f} wavelengths.'.format(array_size))
        
    plt.close()
    return built


    
def chunkify(lst, n):
# This function is used for the parallelization
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def find_local_extrema(chunk, built, not_fulfilled, flips, fulfill_tolerance,diameter,max_array_size):
# This function is used for the parallelization
    max_n_new_fulfilled = 0
    min_built_size = get_array_size(built)
    max_min_distance_from_new_antpos = 0
    favored_i = favored_j = favored_k = None
    success = False
    for i,j,k in chunk:
        new_antpos = built[i] + flips[k] * not_fulfilled[j]
        
        # try and add one antenna at the longest commanded-but-not-fulfilled distance from point i
        built_temp = np.vstack([built, new_antpos]) # 50 us

        # check how many are newly fulfilled
        n_new_fulfilled, _ = get_new_fulfilled(new_antpos,built,not_fulfilled, fulfill_tolerance) # 15 ms

        # check what's the shortest distance of the new point from any already built antenna
        min_distance_from_new_antpos = get_min_distance_from_new_antpos(built, new_antpos) # 215 us
        
        # also check the size of the array
        built_temp_size = get_array_size(built_temp) #241 us
        
        # check if there's no collision and that we're still within max size
        if (not collision_check(built_temp,diameter)) and built_temp_size<max_array_size: #245 us
            success = True    
            if n_new_fulfilled > max_n_new_fulfilled:
                max_n_new_fulfilled = n_new_fulfilled
                favored_i = i
                favored_j = j
                favored_k = k
            elif n_new_fulfilled == max_n_new_fulfilled:
                if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                    favored_i = i
                    favored_j = j
                    favored_k = k
                elif min_distance_from_new_antpos == max_min_distance_from_new_antpos:
                    if built_temp_size < min_built_size:
                        min_built_size = built_temp_size
                        favored_i = i
                        favored_j = j
                        favored_k = k
    return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_built_size, favored_i, favored_j, favored_k

        
        
def create_array_rules_parallelized(commanded, built = None, diameter = 8.54, max_array_size = 300, fulfill_tolerance = 0.5, order = -1, show_plot = True, save_file = False, save_name = 'long_rules', verbose = True, num_cores = 64):
    
    if built is not None:
        built = np.asarray(built)
        
        if not (built.shape == (2,) or (len(built.shape) == 2 and built.shape[1] == 2)):
            print('Incorrect passed built array, using [0,0].')
            built = np.asarray([0,0])
    else:
        built = np.asarray([0,0])

    
    if show_plot:
        fig,ax = plt.subplots(1,3,figsize=(15,5))
    
    if verbose:
        print('Before even beginning, have:')
        print('{:d} antennas built'.format(built.shape[0]))
    
    
    # sort by longest to shortest baseline
    commanded = commanded[np.argsort(order * np.linalg.norm(commanded, axis=1))]
    
    n_new_fulfilled_list,_,_ = get_new_fulfilled_list(commanded, built, fulfill_tolerance)
    
    not_fulfilled = np.copy(commanded)
    
    # do the first iteration, which is trivial
    built = np.vstack([built, not_fulfilled[0]])
    n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
    flips = np.asarray([-1,1])
    
    
    while(not_fulfilled.shape[0]>=1):
        
        all_combinations = list(itertools.product(range(built.shape[0]),range(not_fulfilled.shape[0]),range(2)))
        
        
        chunks = list(chunkify(all_combinations, len(all_combinations) // num_cores))
        
        args_for_starmap = [(chunk, built, not_fulfilled, flips, fulfill_tolerance,diameter,max_array_size) for chunk in chunks]
        # Run parallel computation
        with Pool(processes = num_cores) as pool:
            results = pool.starmap(find_local_extrema, args_for_starmap)
        
        # Aggregate results
        global_success = False
        global_max_n_new_fulfilled = 0
        global_max_min_distance_from_new_antpos = 0
        global_min_built_size = 0
        global_favored_i = global_favored_j = global_favored_k = None
        
        for result in results:
            if result[0] == True:
                global_success = True
            if result[1] > global_max_n_new_fulfilled:
                global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_built_size,global_favored_i,global_favored_j,global_favored_k = result[1:]
            elif result[1] == global_max_n_new_fulfilled:
                if result[2]>global_max_min_distance_from_new_antpos:
                    global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_built_size,global_favored_i,global_favored_j,global_favored_k = result[1:]
                elif result[2]==global_max_min_distance_from_new_antpos:
                    if result[3]>global_min_built_size:
                        global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_built_size,global_favored_i,global_favored_j,global_favored_k = result[1:]
                        
        if global_success == False:
            print('Array is full for this set of parameters, quitting.')
            return built
            break
            
        
        built = np.vstack([built,built[global_favored_i] + flips[global_favored_k]*not_fulfilled[global_favored_j]])
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment(commanded,built, fulfill_tolerance)
        not_fulfilled = not_fulfilled[np.argsort(order * np.linalg.norm(not_fulfilled, axis=1))]
        
        n_new_fulfilled_list.append(global_max_n_new_fulfilled)
        
        if show_plot:
            clear_output(wait=True)
            plt.pause(0.01)
            #built_uvs = antpos_to_uv(built)
            ax[0].plot(commanded[:,0],commanded[:,1],'.',color='k',alpha=0.15,label='Commanded points')
            ax[0].plot(fulfilled[:,0],fulfilled[:,1],'.',color='#00ff00',label='Fulfilled points')
            ax[0].set_title('uv plane')
            ax[0].set_xlabel(r'$u$')
            ax[0].set_ylabel(r'$v$')
            #ax[0].legend()
            ax[1].scatter(*zip(*built),marker='o', s=20,color='k')
            ax[1].set_title('Array')
            ax[1].set_xlabel(r'EW [$\lambda$]')
            ax[1].set_ylabel(r'NS [$\lambda$]')
            ax[2].plot(range(len(n_new_fulfilled_list)),n_new_fulfilled_list,color='k')
            ax[2].set_ylabel('Number of newly fulfilled points')
            ax[2].set_xlabel('New antenna rank')
            ax[2].grid()
            for i in range(2):
                ax[i].set_aspect('equal', adjustable='box')
            #if built.shape[0]%10==0:
            display(fig)
            
        if verbose:
            print('Array size is now: {:.2f} wavelengths'.format(global_min_built_size))
            print('{:d} newly fulfilled points'.format(global_max_n_new_fulfilled))
            print('{:d} total antennas built'.format(built.shape[0]))
            print('{:d}/{:d} commanded points remain to be fulfilled'.format(not_fulfilled.shape[0],commanded.shape[0]))
    
    

    if save_file:
        # Saving the variable to disk
        np.save('built_'+save_name+'.npy',built)

    return built
    if verbose:
        print('Done.')