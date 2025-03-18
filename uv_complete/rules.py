#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import itertools
from uv_complete.utils import check_fulfillment_idx,get_array_size, get_new_fulfilled, get_min_distance_from_new_antpos, collision_check,plot_array,get_antpos_history,get_new_fulfilled_grid, get_efficiency_array, get_array_config, select_baselines
from multiprocessing import Pool, cpu_count
import time
from datetime import timedelta, datetime
import os

    
      
def initialize_antpos(antpos,commanded,fulfill_tolerance,try_continue,start_from, save_name):
    step_time_array = [0.0]
    rejected_combinations = []
    efficiency_array = [1]
    if try_continue:
        path_to_antpos = 'antpos_'+save_name+'.npy'
        if os.path.exists(path_to_antpos):
            print(f'Found antpos_{save_name}.npy, loading and using it... ',end='')
            
            antpos = np.load('antpos_'+save_name+'.npy')
            if start_from is not None:
                if start_from < len(antpos):
                    antpos = antpos[:start_from]
                else:
                    print(f'Value of start_from ({start_from}) too high, it needs to be smaller than len(antpos) ({len(antpos)}). Starting from the end.')
                    start_from = None
            print('done loading.')
            
            path_to_step_time_array = 'step_time_array_'+save_name+'.npy'
            if os.path.exists(path_to_step_time_array):
                print('Loading the step_time_array... ', end='')
                step_time_array = list(np.load(path_to_step_time_array))
                if start_from is not None:
                    step_time_array = step_time_array[:start_from]
                print('done loading.')
            else:
                print(f'File step_time_array_{save_name}.npy not found, creating an empty step_time_array.')
                
            path_to_rejected_combinations = 'rejected_combinations_'+save_name+'.npy'
            if os.path.exists(path_to_rejected_combinations):
                print('Loading the rejected_combinations... ',end='')
                rejected_combinations = list(np.load(path_to_rejected_combinations))
                if start_from is not None:
                    rejected_combinations = rejected_combinations[:start_from]
                print('done loading.')
            else:
                print(f'File rejected_combinations_{save_name}.npy not found, creating a new one, this might take a bit longer than usual.')
            print('Computing the antpos history... ',end='')
            n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list = get_antpos_history(commanded, antpos, fulfill_tolerance)
            print('done.')
            n_not_fulfilled = n_not_fulfilled_list[-1]
            
            path_to_efficiency_array = 'efficiency_array_'+save_name+'.npy'
            if os.path.exists(path_to_efficiency_array):
                print('Loading the efficiency_array... ', end='')
                efficiency_array = list(np.load(path_to_efficiency_array))
                if start_from is not None:
                    efficiency_array = efficiency_array[:start_from]
                print('done loading.')
            else:
                print(f'File efficiency_array_{save_name}.npy not found, creating a new one, this might take a bit longer than usual.')
                efficiency_array = get_efficiency_array(commanded, antpos, n_not_fulfilled_list)
            
        else:
            antpos = None
            print(f'File antpos_{save_name}.npy not found, starting from scratch.')
    
        
    if antpos is None or np.array_equal(antpos, [0, 0]):
        antpos = np.asarray([0, 0])
        starting_from_scratch = True
        n_new_fulfilled_list = [0]
        n_not_fulfilled_list = [len(commanded)]
        new_fulfilled_list = []
        n_not_fulfilled = len(commanded)
    else:
        antpos = np.asarray(antpos)
        if antpos.shape == (2,) or (antpos.ndim == 2 and antpos.shape[1] == 2):
            starting_from_scratch = False
            n_new_fulfilled_list = []
            n_not_fulfilled_list = []
            new_fulfilled_list = []
            _, not_fulfilled = check_fulfillment_idx(commanded,antpos,fulfill_tolerance)
            n_not_fulfilled = len(not_fulfilled)
        else:
            print('Incorrect passed antpos array, using [0, 0].')
            antpos = np.asarray([0, 0])
            starting_from_scratch = True
            
          
    
    # if starting from zero, do the first iteration, which is trivial
    if starting_from_scratch:
        new_antpos = commanded[0]
        antpos = np.vstack([antpos, new_antpos])
        step_time_array.append(0.0)
        efficiency_array.append(1)
        n_new_fulfilled_list.append(1)
        n_not_fulfilled_list.append(len(commanded) - 1)
        new_fulfilled_list.append( get_new_fulfilled(new_antpos,antpos,commanded, fulfill_tolerance) )
    
    array_size = get_array_size(antpos)
    print('Initialization over.')

    return starting_from_scratch, antpos, array_size, step_time_array, rejected_combinations, efficiency_array, n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list, n_not_fulfilled
        
        
        
def chunkify(lst, n):
# This function is used for the parallelization
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

  

def try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance, diameter, max_array_size,center_at_origin):
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
    elif not center_at_origin:
        if temp_array_size <= max_array_size:
            array_size_pass = True
        else:
            return collision_pass, array_size_pass, temp_array_size
        
    return collision_pass, array_size_pass, temp_array_size

def find_local_extrema(chunk, antpos, commanded, not_fulfilled_idx, fulfill_tolerance,diameter,max_array_size,center_at_origin,maximize_antenna_spacing):
# This function is used for the parallelization
    max_n_new_fulfilled = 0
    max_min_distance_from_new_antpos = 0
    favored_i = favored_j = favored_k = None
    success = False
    newly_rejected_combinations = []
    min_array_size = np.inf
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
                if min_distance_from_new_antpos > max_min_distance_from_new_antpos and maximize_antenna_spacing:
                    max_min_distance_from_new_antpos = min_distance_from_new_antpos
                    favored_i = i
                    favored_j = j
                    favored_k = k
                elif temp_array_size < min_array_size:
                        min_array_size = temp_array_size
                        favored_i = i
                        favored_j = j
                        favored_k = k
        else:
            newly_rejected_combinations.append((i,j,k))
    return success, max_n_new_fulfilled, max_min_distance_from_new_antpos, min_array_size, favored_i, favored_j, favored_k, newly_rejected_combinations




def add_ant_rules(commanded, antpos = None, diameter = None, fulfill_tolerance = 0.5, max_array_size = None,  commanded_order = 1, antpos_order = 1, order_antpos_by_magnitude = False, center_at_origin=True, n_to_add = -1, n_max_antennas = -1, compare_all_commanded = False, compare_all_antpos = True, maximize_antenna_spacing = True, start_from = None, save_file = False, save_name = None, show_plot = True, verbose = True,try_continue = True, num_cores = 64):
    
    n_added = 0
    if save_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{timestamp}"
    
    if num_cores is None:
        num_cores = cpu_count()
    
    starting_from_scratch, antpos, array_size, step_time_array, rejected_combinations, efficiency_array, n_new_fulfilled_list, n_not_fulfilled_list, new_fulfilled_list, n_not_fulfilled = initialize_antpos(antpos, commanded, fulfill_tolerance, try_continue, start_from, save_name)


    if verbose:
        if compare_all_commanded:
            compare_all_commanded_str = 'All commanded baselines will be checked at each iteration. This may be very long. For a faster solution, set compare_all_commanded = False.'
            print(compare_all_commanded_str)
        print('Before even beginning, have:')
        print('{:d} antennas in the antpos array'.format(len(antpos)))
    
    


    if commanded_order==0:
        if verbose:
            print('Ordering the commanded points randomly.')
        commanded = commanded[np.random.permutation(commanded.shape[0])]
    else:
        commanded = commanded[np.argsort(commanded_order * np.linalg.norm(commanded, axis=1))]

    # check fulfillment
    fulfilled_idx, not_fulfilled_idx = check_fulfillment_idx(commanded, antpos, fulfill_tolerance)
    array_is_complete = (len(not_fulfilled_idx)==0)
    

    i_commanded = 0
    i_antpos = 0

    rejected_combinations = []
    if verbose:
        step_start_time = time.time()
        
    if save_file:
        np.save('commanded_'+save_name+'.npy',commanded)
    
    if n_not_fulfilled==1:
        print(f'Only {commanded[not_fulfilled_idx[0]]} remains to be fulfilled')
    
    while(len(not_fulfilled_idx)>=1 and not n_added==n_to_add and not len(antpos)==n_max_antennas or not (starting_from_scratch and array_is_complete)):
        global_success = array_is_complete
        
        if not array_is_complete:
            if compare_all_commanded:
                commanded_idx = not_fulfilled_idx
            else:
                if commanded_order==0:
                    commanded_idx = [not_fulfilled_idx[i_commanded]]
                else:
                    commanded_idx = [not_fulfilled_idx[i_commanded * (-1+commanded_order)//2]]
                
                
            if compare_all_antpos:
                antpos_idx = range(len(antpos))
                
                
            if compare_all_commanded and not compare_all_antpos:
                if order_antpos_by_magnitude:
                    antpos_idx = np.arange(len(antpos))[np.argsort(antpos_order * np.linalg.norm(antpos, axis=1))]
                    antpos_idx = [antpos_idx[i_antpos]]
                elif antpos_order == 0:
                    antpos_idx = [np.random.randint(len(antpos))]
                else:
                    antpos_idx = [antpos_order * i_antpos]
                
            if not compare_all_antpos and not compare_all_commanded:
                if order_antpos_by_magnitude:
                        antpos_idx = np.arange(len(antpos))[np.argsort(antpos_order * np.linalg.norm(antpos, axis=1))]
                elif antpos_order == 0:
                    antpos_idx = np.arange(len(antpos))[np.random.permutation(len(antpos))]
                else:
                    antpos_idx = range(len(antpos))[::antpos_order]
            
            all_combinations = set(itertools.product(antpos_idx,commanded_idx,range(2)))
            
            remaining_combinations = list(all_combinations - set(rejected_combinations))  
            
            
            
            if compare_all_commanded or compare_all_antpos:
                
                    
                
                chunks = list(chunkify(remaining_combinations, np.max([len(remaining_combinations) // num_cores,num_cores])))
                args_for_starmap = [(chunk, antpos, commanded, not_fulfilled_idx, fulfill_tolerance,diameter,max_array_size,center_at_origin,maximize_antenna_spacing) for chunk in chunks]
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
                        if result[2]>global_max_min_distance_from_new_antpos and maximize_antenna_spacing:
                            global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
                        elif result[3]<global_min_array_size:
                                global_max_n_new_fulfilled,global_max_min_distance_from_new_antpos,global_min_array_size,global_favored_i,global_favored_j,global_favored_k = result[1:-1]
                    if result[7] is not None:
                        rejected_combinations = list( set(rejected_combinations) | set(result[7]))
                        
                if global_success == False:
                    if (compare_all_commanded and compare_all_antpos) or (i_commanded >= len(not_fulfilled_idx) and compare_all_antpos) or (i_antpos >= len(antpos) and compare_all_commanded) or (i_antpos >= len(antpos) and i_commanded >= len(not_fulfilled_idx)):
                        print('Array is full for this set of parameters, quitting.')
                        return antpos
                        break
                    elif not compare_all_antpos and compare_all_commanded:
                        if i_antpos<len(antpos):
                            i_antpos+=1
                    elif compare_all_antpos and not compare_all_commanded:
                        if i_commanded<len(not_fulfilled_idx):
                            i_commanded+=1
                    else:
                        if i_antpos<=len(antpos):
                            i_antpos+=1
                        else:
                            i_antpos = 0
                            i_commanded += 1
                
            else:
                for i,j,k in remaining_combinations:
                    new_antpos = antpos[i] + (-1)**k * commanded[j]
                    collision_pass, array_size_pass, temp_array_size = try_new_antpos(antpos, new_antpos, commanded,fulfill_tolerance, diameter, max_array_size,center_at_origin)
                    if collision_pass and array_size_pass:
                        global_success = True
                        new_fulfilled = get_new_fulfilled(new_antpos,antpos,commanded[not_fulfilled_idx], fulfill_tolerance)
                        global_favored_i = i
                        global_favored_j = j
                        global_favored_k = k
                        global_max_n_new_fulfilled = len(new_fulfilled)
                        break
                    else:
                        rejected_combinations = list( set(rejected_combinations) | set([(i,j,k)]))
                    
            
        if global_success or (array_is_complete and not starting_from_scratch):
            n_new_fulfilled_list.append(global_max_n_new_fulfilled)
            
            n_not_fulfilled -= global_max_n_new_fulfilled
            n_not_fulfilled_list.append(n_not_fulfilled)
            new_antpos = antpos[global_favored_i] + (-1)**global_favored_k*commanded[global_favored_j]
            new_fulfilled_list.append( get_new_fulfilled(new_antpos,antpos,commanded[not_fulfilled_idx], fulfill_tolerance) )
            i_antpos = 0
            i_commanded = 0
            antpos = np.vstack([antpos,new_antpos])
            fulfilled_idx, not_fulfilled_idx = check_fulfillment_idx(commanded,antpos, fulfill_tolerance)
            n_added += 1
            
            if verbose:
                current_time = time.time()
                step_time = current_time - step_start_time
                step_time_str = str(timedelta(seconds=int(step_time)))
                step_time_array.append(step_time)
                n_baselines = len(antpos) * (len(antpos) - 1) / 2
                efficiency = (  len(fulfilled_idx) / n_baselines ) ** .5
                total_time = np.sum(step_time_array)
                efficiency_array.append(efficiency)
                total_time_str = str(timedelta(seconds=int(total_time)))
                step_start_time = time.time()
                if show_plot:
                    clear_output(wait=True)
                    
                    fig,ax=plot_array(antpos = antpos,
                                      commanded = commanded,
                                      diameter = diameter,
                                      fulfill_tolerance = fulfill_tolerance,
                                      just_plot_array=False,
                                      plot_new_fulfilled=True,
                                      n_new_fulfilled_list = n_new_fulfilled_list,
                                      n_not_fulfilled_list=n_not_fulfilled_list,
                                      new_fulfilled_list=new_fulfilled_list,
                                      fulfilled = commanded[fulfilled_idx],
                                      not_fulfilled = commanded[not_fulfilled_idx],
                                      step_time_array=step_time_array,
                                      efficiency_array = efficiency_array)
                    plt.pause(0.01)
                array_size = get_array_size(antpos)
                if array_is_complete:
                    global_max_n_new_fulfilled = n_new_fulfilled_list[-1]
                if compare_all_commanded:
                    print(compare_all_commanded_str)
                print('\n')
                print(f'Just added an antenna at {new_antpos}.')
                print(f'Array now has {len(antpos)} antennas.')
                print('Array now spans {:.2f} meters in size.'.format(array_size))
                print('{:d} newly fulfilled points at last iteration.'.format(global_max_n_new_fulfilled))
                print('{:d}/{:d} commanded points remain to be fulfilled after last iteration.'.format(len(not_fulfilled_idx),len(commanded)))
                print(f'Number of rejected combinations after last iteration: {len(rejected_combinations)}')
                print(time.strftime('Local time after last iteration: %H:%M:%S', time.localtime()))
                
                print(f'Time for last step: {step_time_str}.')
                print('Total time' + (len(not_fulfilled_idx)>0)*' so far' + f': {total_time_str}.')
                if len(not_fulfilled_idx)==0:
                    print('Antenna cost efficiency: {:.2f} %'.format(100 * (2 * len(commanded)) ** .5 / len(antpos)))
                    
                array_is_complete = len(not_fulfilled_idx)==0
            if save_file:
                # Saving the variable to disk
                np.save('antpos_'+save_name+'.npy',antpos)
                np.save('step_time_array_'+save_name+'.npy',np.array(step_time_array))
                np.save('rejected_combinations_'+save_name+'.npy',np.array(rejected_combinations))
                np.save('efficiency_array_'+save_name+'.npy',np.array(efficiency_array))
        if array_is_complete or n_added == n_to_add:
            break
        

    if save_file:
        # Saving the variable to disk
        array_config = get_array_config(antpos)
        np.save(f'array_config_{save_name}.npy',array_config)
        np.save('antpos_'+save_name+'.npy',antpos)
        baseline_select = select_baselines(commanded, antpos, fulfill_tolerance)
        np.save(f'baseline_select_{save_name}.npy',baseline_select)

    return antpos
    if verbose:
        print('Done.')




def add_ant_grid(commanded_grid, antpos_grid = None, diameter = None, max_array_size = None, n_to_add = -1, n_max_antennas = -1, save_file = False, save_name = None, show_plot = True, verbose = True, try_continue = True, num_cores = 64):
    
    if max_array_size is None:
        max_array_size = 6 * np.max(np.linalg.norm(commanded_grid,axis=1))
    
    grid = [[i, j] for i in range(max_array_size) for j in range(max_array_size)]
    
    antpos_grid = np.array([[0,0]])
    
    not_fulfilled_grid = commanded_grid
    points_still_in_grid = np.array(grid)
    
    while(len(not_fulfilled_grid) > 0 and len(points_still_in_grid) > 0):
        
        max_new_fulfilled_grid = 0
        favored_new_antpos_grid = None
        for new_antpos_grid in points_still_in_grid:
            new_fulfilled_grid = get_new_fulfilled_grid(new_antpos_grid, antpos_grid, not_fulfilled_grid)
            if len(new_fulfilled_grid) > max_new_fulfilled_grid:
                max_new_fulfilled_grid = len(new_fulfilled_grid)
                favored_new_antpos_grid = 0
        
        if favored_new_antpos_grid is None:
            print('Failed to place any antenna. Stopping.')
            break
        else:
            antpos_grid = np.vstack(antpos_grid,favored_new_antpos_grid)

    print('Done')
            
            
            
            
            
            
