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
from uvComplete.utils import check_fulfillment,check_fulfillment_idx,check_fulfillment_old, get_array_size, get_new_fulfilled_old, get_new_fulfilled, get_min_distance_from_new_antpos, collision_check,plot_array,get_antpos_history
from multiprocessing import Pool
import time
from datetime import timedelta

class ExitLoopsException(Exception):
    pass



    
def chunkify(lst, n):
# This function is used for the parallelization
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def find_local_extrema(chunk, antpos, not_fulfilled, flips, fulfill_tolerance,diameter,max_array_size,center_at_origin,ruled_out_antpos):
# This function is used for the parallelization
    max_n_new_fulfilled = 0
    min_array_size = get_array_size(antpos)
    max_min_distance_from_new_antpos = 0
    favored_i = favored_j = favored_k = None
    success = False
    newly_ruled_out_antpos = None
    for i,j,k in chunk:
        
        collision_pass = False
        array_size_pass = False
        new_antpos = antpos[i] + flips[k] * not_fulfilled[j]
        
        if len(ruled_out_antpos.shape)>1:
            # If we've already tried this position and it didn't work, move on
            if np.any(np.all(np.abs(ruled_out_antpos-new_antpos) < 1e-9, axis=1)):
                return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_ruled_out_antpos
        
        
        
        # try and add one antenna at the longest commanded-but-not-fulfilled distance from point i
        temp_antpos = np.vstack([antpos, new_antpos]) # 50 us

        
        # Check if there's no collision (collision_check returns True if there's a collision)
        if not collision_check(temp_antpos,diameter):
            collision_pass = True
        else:
            newly_ruled_out_antpos = new_antpos
            return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_ruled_out_antpos
        
        
        # Check the size of the array
        temp_array_size = get_array_size(temp_antpos) #241 us
        if max_array_size is None:
            array_size_pass = True
        elif center_at_origin:
            if np.linalg.norm(new_antpos)>max_array_size/2:
                newly_ruled_out_antpos = new_antpos
                return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, newly_ruled_out_antpos
            else:
                array_size_pass = True
        else:
            if temp_array_size < max_array_size:
                array_size_pass = True
            else:
                newly_ruled_out_antpos = new_antpos
                return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_ruled_out_antpos
            

        if collision_pass and array_size_pass:
            success = True    
            # check how many are newly fulfilled
            new_fulfilled = get_new_fulfilled(new_antpos,antpos,not_fulfilled, fulfill_tolerance) # 15 ms

            # check what's the shortest distance of the new point from any already antpos antenna
            min_distance_from_new_antpos = get_min_distance_from_new_antpos(antpos, new_antpos) # 215 us
            

            if len(new_fulfilled) > max_n_new_fulfilled:
                max_n_new_fulfilled = len(new_fulfilled)
                favored_i = i
                favored_j = j
                favored_k = k
            elif len(new_fulfilled) == max_n_new_fulfilled:
                if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                    favored_i = i
                    favored_j = j
                    favored_k = k
                elif min_distance_from_new_antpos == max_min_distance_from_new_antpos:
                    if temp_array_size < min_array_size:
                        min_array_size = temp_array_size
                        favored_i = i
                        favored_j = j
                        favored_k = k
    return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_ruled_out_antpos


        
        
        
        
        
        
        
def initialize_antpos(antpos):
    if antpos is None or np.array_equal(antpos, [0, 0]):
        antpos = np.asarray([0, 0])
        starting_from_scratch = True
    else:
        antpos = np.asarray(antpos)
        if antpos.shape == (2,) or (antpos.ndim == 2 and antpos.shape[1] == 2):
            starting_from_scratch = False
        else:
            print('Incorrect passed antpos array, using [0, 0].')
            antpos = np.asarray([0, 0])
            starting_from_scratch = True
    return starting_from_scratch, antpos
        
        
        
        
        
        
        
def add_ant_rules(commanded, antpos=None, diameter = None, max_array_size = None, fulfill_tolerance = 0.5, order = -1, show_plot = True, save_file = False,save_name='rules', verbose = True, n_max_antennas = np.inf, n_to_add = np.inf, center_at_origin = True, check_first = 'antpos', check_all_antpos = True, check_all_not_fulfilled = False, show_plot_skip = 10, second_priority = 'min_distance_from_new_antpos'):

    n_added = 0

    starting_from_scratch, antpos = initialize_antpos(antpos)

    if verbose:
        print('Before even beginning, have:')
        print('{:d} antennas in the antpos array'.format(len(antpos)))
    
    
    # sort by longest to shortest baseline
    commanded = commanded[np.argsort(order * np.linalg.norm(commanded, axis=1))]
    
    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        antpos = np.vstack([antpos, commanded[0]])
    
    # check fulfillment
    fulfilled, not_fulfilled = check_fulfillment(commanded, antpos, fulfill_tolerance)
    
    if show_plot:
        n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_antpos_history(commanded, antpos, fulfill_tolerance)
        

    not_fulfilled = not_fulfilled[np.argsort(order * np.linalg.norm(not_fulfilled, axis=1))]
    flips = np.asarray([-1,1])
    
    
    
    if check_all_antpos and check_all_not_fulfilled:
        print("It is not recommended to have both check_all_antpos and check_all_not_fulfilled set to True, this will take weeks. Use the parallelized create_array_rules_parallelized instead.")
    
    # This array will contain the combinations that have been ruled out already, so that they are skipped
    ruled_out_antpos = np.asarray([0,0])
    
    while(len(not_fulfilled)>=1 and n_added<n_to_add and len(antpos)<n_max_antennas):
        success = False
        max_n_new_fulfilled = 0
        min_array_size = get_array_size(antpos)
        max_min_distance_from_new_antpos = 0
        
        # Decide which one will be the outer loop, and which one will be the inner loop
        if check_first == 'antpos':
            inner_loop = antpos
            outer_loop = not_fulfilled
            check_second = 'not_fulfilled'
        elif check_first == 'not_fulfilled':
            inner_loop = not_fulfilled
            outer_loop = antpos
            check_second = 'antpos'
        else:
            raise ValueError("Wrong value for check_first")
        
        
        
        # iterate over all combinations of antpos antennas and unfulfilled uv point
        for i in range(len(outer_loop)):
            # iterate over the inner loop
            for j in range(len(inner_loop)):
                # try both the positive and negative position
                for k in range(len(flips)):
                    
                    collision_pass = False
                    array_size_pass = False
                    if check_first == 'antpos':
                        new_antpos = inner_loop[j] + flips[k] * outer_loop[i]
                    elif check_first == 'not_fulfilled':
                        new_antpos = outer_loop[i] + flips[k] * inner_loop[j]
                    
                    if len(ruled_out_antpos.shape)>1:
                        # If we've already tried this position and it didn't work, move on
                        if np.any(np.all(np.abs(ruled_out_antpos-new_antpos) < 1e-9, axis=1)):
                            continue
        
       
                    # Try and add one antenna at the longest commanded-but-not-fulfilled distance from point i
                    temp_antpos = np.vstack([antpos,new_antpos]) # 50 us
    
                    # Check if there's no collision (collision_check returns True if there's a collision)
                    if not collision_check(temp_antpos,diameter):
                        collision_pass = True
                    else:
                        ruled_out_antpos = np.vstack([ruled_out_antpos,new_antpos])
                        continue
                    
                    
                    # Check the size of the array
                    if max_array_size is None:
                        array_size_pass = True
                        if check_all_antpos or check_all_not_fulfilled:
                            temp_array_size = get_array_size(temp_antpos)
                    else:
                        temp_array_size = get_array_size(temp_antpos)
                        if center_at_origin:
                            if np.linalg.norm(new_antpos)>max_array_size/2:
                                ruled_out_antpos = np.vstack([ruled_out_antpos,new_antpos])
                                continue
                            else:
                                array_size_pass = True
                        else:
                            if temp_array_size < max_array_size:
                                array_size_pass = True
                            else:
                                ruled_out_antpos = np.vstack([ruled_out_antpos,new_antpos])
                                continue
                    
                    
                    # if there's no collision and that we're still within max size
                    # (this condition should always be true, otherwise we would've gotten
                    # out of this loop iteration before)
                    if collision_pass and array_size_pass:
                        success = True
                        new_fulfilled = get_new_fulfilled(new_antpos,antpos,not_fulfilled,fulfill_tolerance)
                        min_distance_from_new_antpos = get_min_distance_from_new_antpos(antpos, new_antpos)
                        if check_first == 'antpos' and not check_all_antpos:
                            favored_new_antpos = new_antpos
                            break # breaks out of the flips loop
                        elif check_first == 'not_fulfilled' and not check_all_not_fulfilled:
                            favored_new_antpos = new_antpos
                            break # breaks out of the flips loop
                        elif len(new_fulfilled) > max_n_new_fulfilled:
                            favored_new_antpos = new_antpos
                            max_n_new_fulfilled = len(new_fulfilled)
                        elif len(new_fulfilled) == max_n_new_fulfilled:
                            if second_priority == 'array_size':
                                if temp_array_size < min_array_size:
                                    favored_new_antpos = new_antpos
                                    min_array_size = temp_array_size
                                elif temp_array_size == min_array_size:
                                    if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                                        favored_new_antpos = new_antpos
                                        max_min_distance_from_new_antpos = min_distance_from_new_antpos
                            elif second_priority == 'min_distance_from_new_antpos':
                                if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                                    favored_new_antpos = new_antpos
                                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                                elif min_distance_from_new_antpos == max_min_distance_from_new_antpos:
                                    if temp_array_size < min_array_size:
                                        favored_new_antpos = new_antpos
                                        min_array_size = temp_array_size
                    # If this antpos doesn't work, jot it down
                    else:
                        print('This should not be printing out.')
                        ruled_out_antpos = np.vstack([ruled_out_antpos,new_antpos])
                        
                if success == True:
                    if check_first == 'antpos' and not check_all_antpos:
                        break # breaks out of the inner loop
                    if check_first == 'not_fulfilled' and not check_all_not_fulfilled:
                        break # breaks out of the inner loop
            if success == True:
                if check_second == 'antpos' and not check_all_antpos:
                    break # breaks out of the outer loop
                if check_second == 'not_fulfilled' and not check_all_not_fulfilled:
                    break # breaks out of the outer loop
        
            
        if success == False:
            print('Array is full for this set of parameters, quitting.')
            return antpos
            break
            
        
        new_fulfilled = get_new_fulfilled(favored_new_antpos,antpos,not_fulfilled,fulfill_tolerance)
        antpos = np.vstack([antpos,favored_new_antpos])
        n_added += 1
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment_old(commanded,antpos, fulfill_tolerance)
        not_fulfilled = not_fulfilled[np.argsort(order * np.linalg.norm(not_fulfilled, axis=1))]
        
        n_new_fulfilled_list.append(len(new_fulfilled))
        n_not_fulfilled_list.append(n_not_fulfilled)
        new_fulfilled_list.append(new_fulfilled)
        
        printout_condition = len(antpos)%10==0 or (len(not_fulfilled)<1000)
        show_plot_condition = (len(antpos)%show_plot_skip==0 or len(not_fulfilled)==0)
    
  
        if show_plot:
            if show_plot_condition:
            
                clear_output(wait=True)
                if verbose:
                    print('{:d} newly fulfilled points'.format(len(new_fulfilled)))
                    print('{:d} total antennas antpos'.format(len(antpos)))
                    print('{:d}/{:d} commanded points remain to be fulfilled'.format(len(not_fulfilled),len(commanded)))
                #antpos_uvs = antpos_to_uv(antpos)
                fig,ax=plot_array(antpos,commanded,fulfill_tolerance,just_plot_array=False,plot_new_fulfilled=True,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = fulfilled, not_fulfilled = not_fulfilled)
                plt.pause(0.01)
                
                #display(fig)
        elif verbose:
            if printout_condition:
                clear_output(wait=True)
                print('{:d} newly fulfilled points'.format(len(new_fulfilled)))
                print('{:d} total antennas antpos'.format(len(antpos)))
                print('{:d}/{:d} commanded points remain to be fulfilled'.format(len(not_fulfilled),len(commanded)))
                
    
    plt.close()

    if save_file:
        # Saving the variable to disk
        with open('antpos_'+save_name+'.pkl', 'wb') as file:
            pickle.dump(antpos, file)

    
    if verbose:
        array_size = get_array_size(antpos)
        print('Done.')
        print(f'Used {len(antpos)} antennas to fulfill all baselines, array size is '+'{:.2f} wavelengths.'.format(array_size))
        
    plt.close()
    return antpos





def add_ant_rules_parallelized(commanded, antpos = None, diameter = None, max_array_size = None, fulfill_tolerance = 0.5, center_at_origin=True, n_to_add = np.inf, n_max_antennas = np.inf, save_file = False, save_name = 'default_name', show_plot = False, verbose = True, num_cores = 64):
    
    n_added = 0
    
    starting_from_scratch, antpos = initialize_antpos(antpos)



    if verbose:
        total_start_time = time.time()
        print('Before even beginning, have:')
        print('{:d} antennas in the antpos array'.format(len(antpos)))
    

    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        antpos = np.vstack([antpos, commanded[0]])
    
    # check fulfillment
    fulfilled, not_fulfilled = check_fulfillment(commanded, antpos, fulfill_tolerance)

    not_fulfilled = not_fulfilled[np.argsort(np.linalg.norm(not_fulfilled, axis=1))]
    flips = np.asarray([-1,1])

    
    ruled_out_antpos = np.asarray([0,0])
    while(len(not_fulfilled)>=1 and n_added<n_to_add and len(antpos)<n_max_antennas):
        
        if verbose:
            step_start_time = time.time()
        
        all_combinations = list(itertools.product(range(len(antpos)),range(len(not_fulfilled)),range(2)))
              
        chunks = list(chunkify(all_combinations, len(all_combinations) // num_cores))
        
        args_for_starmap = [(chunk, antpos, not_fulfilled, flips, fulfill_tolerance,diameter,max_array_size,center_at_origin,ruled_out_antpos) for chunk in chunks]
        # Run parallel computation
        with Pool(processes = num_cores) as pool:
            results = pool.starmap(find_local_extrema, args_for_starmap)
        
        # Aggregate results
        global_success = False
        global_max_n_new_fulfilled = 0
        global_max_min_distance_from_new_antpos = 0
        global_min_array_size = 0
        global_favored_i = global_favored_j = global_favored_k = None
        
        for result in results:
            if result[0] == True:
                global_success = True
            if result[1] > global_max_n_new_fulfilled:
                global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
            elif result[1] == global_max_n_new_fulfilled:
                if result[2]>global_max_min_distance_from_new_antpos:
                    global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
                elif result[2]==global_max_min_distance_from_new_antpos:
                    if result[3]>global_min_array_size:
                        global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
            if result[7] is not None:
                ruled_out_antpos = np.vstack([ruled_out_antpos,result[7]])
                
        if global_success == False:
            print('Array is full for this set of parameters, quitting.')
            return antpos
            break
            
        
        antpos = np.vstack([antpos,antpos[global_favored_i] + flips[global_favored_k]*not_fulfilled[global_favored_j]])
        n_fulfilled, n_not_fulfilled, fulfilled, not_fulfilled = check_fulfillment_old(commanded,antpos, fulfill_tolerance)
        not_fulfilled = not_fulfilled[np.argsort(np.linalg.norm(not_fulfilled, axis=1))]
        n_added += 1
        if save_file:
            # Saving the variable to disk
            np.save('antpos_progress_'+save_name+'.npy',antpos)
            
        if verbose:
            if show_plot:
                clear_output(wait=True)
                n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_antpos_history(commanded, antpos, fulfill_tolerance)
                fig,ax=plot_array(antpos,commanded,fulfill_tolerance,just_plot_array=False,plot_new_fulfilled=True,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = fulfilled, not_fulfilled = not_fulfilled)
                plt.pause(0.01)
            print('Array size is now: {:.2f} wavelengths'.format(global_min_array_size))
            print('{:d} newly fulfilled points'.format(global_max_n_new_fulfilled))
            print('{:d} total antennas antpos'.format(len(antpos)))
            print('{:d}/{:d} commanded points remain to be fulfilled'.format(len(not_fulfilled),len(commanded)))
            current_time = time.time()
            step_elapsed_time = current_time - step_start_time
            step_time_str = str(timedelta(seconds=int(step_elapsed_time)))
            total_elapsed_time = current_time - total_start_time
            total_time_str = str(timedelta(seconds=int(total_elapsed_time)))
            print(f'Time for last step: {step_time_str}')
            print(f'Total time so far: {total_time_str}')
    
    

    if save_file:
        # Saving the variable to disk
        np.save('antpos_'+save_name+'.npy',antpos)

    return antpos
    if verbose:
        print('Done.')






























def try_new_antpos(antpos, new_antpos, commanded,fulfill_tolerance, diameter, max_array_size,center_at_origin):
    collision_pass = False
    array_size_pass = False
    temp_array_size = np.inf
    
    # try and add one antenna at the longest commanded-but-not-fulfilled distance from point i
    temp_antpos = np.vstack([antpos, new_antpos]) # 50 us

    # Check if there's no collision (collision_check returns True if there's a collision)
    if not collision_check(temp_antpos,diameter):
        collision_pass = True
    else:
        return collision_pass, array_size_pass, temp_array_size
    
    # Check the size of the array
    temp_array_size = get_array_size(temp_antpos) #241 us
    if max_array_size is None:
        array_size_pass = True
    elif center_at_origin:
        if np.linalg.norm(new_antpos)>max_array_size/2:
            return collision_pass, array_size_pass, temp_array_size
        else:
            array_size_pass = True
    else:
        if temp_array_size < max_array_size:
            array_size_pass = True
        else:
            return collision_pass, array_size_pass, temp_array_size
        
    return collision_pass, array_size_pass, temp_array_size

def find_local_extrema_2(chunk, antpos, commanded, not_fulfilled_idx, fulfill_tolerance,diameter,max_array_size,center_at_origin):
# This function is used for the parallelization
    max_n_new_fulfilled = 0
    min_array_size = get_array_size(antpos)
    max_min_distance_from_new_antpos = 0
    favored_i = favored_j = favored_k = None
    success = False
    newly_ruled_out_combinations = []
    for i,j,k in chunk:
        new_antpos = antpos[i] + (-1)**k * commanded[j]
        collision_pass, array_size_pass, temp_array_size = try_new_antpos(antpos, new_antpos, commanded,fulfill_tolerance, diameter, max_array_size,center_at_origin)
        if collision_pass and array_size_pass:
            success = True    
            # check how many are newly fulfilled
            new_fulfilled = get_new_fulfilled(new_antpos,antpos,commanded[not_fulfilled_idx], fulfill_tolerance) # 15 ms

            # check what's the shortest distance of the new point from any already antpos antenna
            min_distance_from_new_antpos = get_min_distance_from_new_antpos(antpos, new_antpos) # 215 us
            

            if len(new_fulfilled) > max_n_new_fulfilled:
                max_n_new_fulfilled = len(new_fulfilled)
                favored_i = i
                favored_j = j
                favored_k = k
            elif len(new_fulfilled) == max_n_new_fulfilled:
                if min_distance_from_new_antpos > max_min_distance_from_new_antpos:
                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                    favored_i = i
                    favored_j = j
                    favored_k = k
                elif min_distance_from_new_antpos == max_min_distance_from_new_antpos:
                    if temp_array_size < min_array_size:
                        min_array_size = temp_array_size
                        favored_i = i
                        favored_j = j
                        favored_k = k
        else:
            newly_ruled_out_combinations.append((i,j,k))
    return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_ruled_out_combinations






def add_ant_rules_parallelized_2(commanded, antpos = None, diameter = None, max_array_size = None, fulfill_tolerance = 0.5, center_at_origin=True, n_to_add = -1, n_max_antennas = -1, save_file = False, save_name = 'default_name', show_plot = False, verbose = True, num_cores = 64):
    
    n_added = 0
    
    starting_from_scratch, antpos = initialize_antpos(antpos)



    if verbose:
        total_start_time = time.time()
        print('Before even beginning, have:')
        print('{:d} antennas in the antpos array'.format(len(antpos)))
    
    
    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        antpos = np.vstack([antpos, commanded[0]])
    
    # check fulfillment
    fulfilled_idx, not_fulfilled_idx = check_fulfillment_idx(commanded, antpos, fulfill_tolerance)


 

    ruled_out_combinations = []
    while(len(not_fulfilled_idx)>=1 and not n_added==n_to_add and not len(antpos)==n_max_antennas):
        if verbose:
            step_start_time = time.time()
        
        all_combinations = set(itertools.product(range(len(antpos)),not_fulfilled_idx,range(2)))
        remaining_combinations = list(all_combinations - set(ruled_out_combinations))      
        
        
        chunks = list(chunkify(remaining_combinations, len(remaining_combinations) // num_cores))
        
        args_for_starmap = [(chunk, antpos, commanded, not_fulfilled_idx, fulfill_tolerance,diameter,max_array_size,center_at_origin) for chunk in chunks]
        # Run parallel computation
        with Pool(processes = num_cores) as pool:
            results = pool.starmap(find_local_extrema_2, args_for_starmap)
        
        # Aggregate results
        global_success = False
        global_max_n_new_fulfilled = 0
        global_max_min_distance_from_new_antpos = 0
        global_min_array_size = 0
        global_favored_i = global_favored_j = global_favored_k = None
        
        for result in results:
            if result[0] == True:
                global_success = True
            if result[1] > global_max_n_new_fulfilled:
                global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
            elif result[1] == global_max_n_new_fulfilled:
                if result[2]>global_max_min_distance_from_new_antpos:
                    global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
                elif result[2]==global_max_min_distance_from_new_antpos:
                    if result[3]>global_min_array_size:
                        global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
            if result[7] is not None:
                ruled_out_combinations = list( set(ruled_out_combinations) | set(result[7]))
                
        if global_success == False:
            print('Array is full for this set of parameters, quitting.')
            return antpos
            break
            
        antpos = np.vstack([antpos,antpos[global_favored_i] + (-1)**global_favored_k*commanded[global_favored_j]])
        fulfilled_idx, not_fulfilled_idx = check_fulfillment_idx(commanded,antpos, fulfill_tolerance)
        n_added += 1
        if save_file:
            # Saving the variable to disk
            np.save('antpos_progress_'+save_name+'.npy',antpos)
            
        if verbose:
            if show_plot:
                clear_output(wait=True)
                n_new_fulfilled_list,n_not_fulfilled_list,new_fulfilled_list = get_antpos_history(commanded, antpos, fulfill_tolerance)
                fig,ax=plot_array(antpos,commanded,fulfill_tolerance,just_plot_array=False,plot_new_fulfilled=True,n_new_fulfilled_list = n_new_fulfilled_list,n_not_fulfilled_list=n_not_fulfilled_list,new_fulfilled_list=new_fulfilled_list, fulfilled = commanded[fulfilled_idx], not_fulfilled = commanded[not_fulfilled_idx])
                plt.pause(0.01)
            print('Array size is now: {:.2f} wavelengths'.format(global_min_array_size))
            print('{:d} newly fulfilled points'.format(global_max_n_new_fulfilled))
            print('{:d} total antennas antpos'.format(len(antpos)))
            print('{:d}/{:d} commanded points remain to be fulfilled'.format(len(not_fulfilled_idx),len(commanded)))
            current_time = time.time()
            step_elapsed_time = current_time - step_start_time
            step_time_str = str(timedelta(seconds=int(step_elapsed_time)))
            total_elapsed_time = current_time - total_start_time
            total_time_str = str(timedelta(seconds=int(total_elapsed_time)))
            print(f'Time for last step: {step_time_str}')
            print(f'Total time so far: {total_time_str}')
    
    

    if save_file:
        # Saving the variable to disk
        np.save('antpos_'+save_name+'.npy',antpos)

    return antpos
    if verbose:
        print('Done.')









