#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:13:22 2025

@author: vincent
"""

### OLD IMPLEMENTATION OF ADD_ANT_RULES
### IGNORE, WILL RETIRE EVENTUALLY

import numpy as np
from IPython.display import clear_output
import time
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import warnings
from itertools import product
from .. import geometry
from .. import plotting
from .. import utils
from datetime import datetime

def add_ant_greedy(AA, commanded_order=1, compare_all_commanded=False, compare_all_antpos=False,
             order_antpos_by_magnitude=False, antpos_order=1, center_at_origin=True,
             max_array_size=None, n_max_antennas=-1, n_to_add=-1, minimize_array_size = True, maximize_antenna_spacing=False, 
             save_file=False, path_to_file=None, verbose=True, show_plot=False, additional_output = None,
             num_cores=None, log_path = None):

    
    
    prepare_algorithm(AA,commanded_order,compare_all_commanded, center_at_origin, max_array_size, minimize_array_size, maximize_antenna_spacing, path_to_file, verbose, show_plot, num_cores)
    last_step_time = time.time()
    while not AA.array_is_complete:
        # Step 1: Generate valid commanded-reference antenna combinations
        remaining_combinations = generate_candidate_combinations(AA,compare_all_commanded, compare_all_antpos, 
                                                                       order_antpos_by_magnitude, antpos_order)

        if not remaining_combinations:
            print("Array is full for this set of parameters, quitting.")
            break

        # Step 2: Evaluate combinations
        if compare_all_commanded or compare_all_antpos:
            global_success, best_candidate = evaluate_candidates_parallel(AA,remaining_combinations)
        else:
            global_success, best_candidate = evaluate_candidate_sequential(AA,remaining_combinations)
        
        # Step 3: If no valid placement is found, terminate or iterate
        if not global_success:
            if should_terminate(AA, compare_all_commanded, compare_all_antpos):
                print("Array is full, terminating.")
                break
            continue  # Otherwise, iterate to the next candidate

        # Step 4: Apply placement
        place_antenna(AA,best_candidate)
        # Track time taken to place this antenna
        step_time = time.time() - last_step_time
        AA.history["step_time"].append(step_time)
        

        # Step 5: Save progress
        if save_file:
            AA.save(path_to_file)


        # Step 6: Print status dynamically (if verbose)
        if verbose:
            print_status(AA, additional_output = additional_output, log_path = log_path)
            
        # Step 7: Plot (if show_plot)
        if show_plot:
            AA.plot_fig, AA.plot_ax = plotting.plot_history(AA, AA.plot_fig, AA.plot_ax)

        # Start the timer now such that it isn't affected by plotting and printing
        last_step_time = time.time()            


        # Step 8: Check stopping conditions
        if AA.array_is_complete or (0 < n_to_add == AA.n_added) or (0 < n_max_antennas <= len(AA.antpos)):
            break

    print("Done.")
    


def prepare_algorithm(AA, commanded_order, compare_all_commanded, center_at_origin, max_array_size, minimize_array_size, maximize_antenna_spacing, path_to_file, verbose, show_plot, num_cores):
    """Handles initialization and setup logic, assuming the AntArray instance is already loaded if needed."""
    
    # ====== STEP 1: Determine If Starting from Scratch ======
    is_default_antpos = np.array_equal(AA.antpos, np.array([[0., 0.]]))
    is_invalid_shape = AA.antpos.shape[1] != 2  # If antpos is malformed

    AA.starting_from_scratch = is_default_antpos or is_invalid_shape

    if AA.starting_from_scratch:
        print("Starting from scratch...")

        # Reset Antenna Positions
        AA.antpos = np.array([[0, 0]])  # First antenna at origin
        
        # Reset History Dictionary
        AA.history = {
            "n_new_fulfilled": [0],
            "n_fulfilled": [0],
            "n_not_fulfilled": [len(AA.commanded)],
            "new_fulfilled": [],
            "step_time": [0.0],
            "efficiency": [1],
            "rejected_combinations": [],# Now stored in history
        }
        AA.n_not_fulfilled = len(AA.commanded)
    elif not hasattr(AA, 'history'):
        AA.history = {
            "n_new_fulfilled": [0],
            "n_fulfilled": [0],
            "n_not_fulfilled": [len(AA.commanded)],
            "new_fulfilled": [],
            "step_time": [0.0],
            "efficiency": [1],
            "rejected_combinations": [],# Now stored in history
        }
    
    # ====== STEP 2: Setup Metadata ======
    if path_to_file is None:
        AA.path_to_file = './AntArray_'+datetime.now().strftime("%Y%m%d_%H%M%S")+'.pkl'
    if num_cores is None or not isinstance(num_cores, int):
        AA.num_cores = cpu_count()
    else:
        AA.num_cores = num_cores

    # ====== STEP 3: Handle Verbose Warnings ======
    if verbose:
        if compare_all_commanded:
            warnings.warn(
                "All commanded baselines will be checked at each iteration. "
                "This may be very long. For a faster solution, set compare_all_commanded = False."
            )
        print(f'Before even beginning, {len(AA.antpos)} antennas in the antpos array.')
    


    # ====== STEP 4: Sorting Commanded Points ======
    if commanded_order == 0:
        if verbose:
            print('Ordering the commanded points randomly.')
        AA.commanded = AA.commanded[np.random.permutation(AA.commanded.shape[0])]
    else:
        AA.commanded = AA.commanded[np.argsort(commanded_order * np.linalg.norm(AA.commanded, axis=1))]

    # ====== STEP 5: Fulfillment Check ======

    AA.fulfilled_idx, AA.not_fulfilled_idx = geometry.check_fulfillment(
        commanded = AA.commanded,
        antpos = AA.antpos,
        fulfill_tolerance = AA.fulfill_tolerance,
    )
    

    
    
    AA.array_is_complete = len(AA.not_fulfilled_idx) == 0
    AA.not_fulfilled_array = AA.commanded[AA.not_fulfilled_idx]
    AA.not_fulfilled_tree = KDTree(AA.not_fulfilled_array)
    AA.n_not_fulfilled = len(AA.not_fulfilled_idx)

    # Initial indices for the algorithm
    AA.i_commanded = 0
    AA.i_antpos = 0

    # ====== STEP 6: Other Attributes ======
    AA.center_at_origin = center_at_origin
    if max_array_size is not None:
        AA.max_array_size = max_array_size
    AA.minimize_array_size = minimize_array_size
    AA.maximize_antenna_spacing = maximize_antenna_spacing
    AA.antpos_norms = np.linalg.norm(AA.antpos, axis=1)
    AA.n_added = 0  # Track number of antennas added for this given run
    # is always zero regardless of if starting from scratch

    # ====== STEP 7: Verbose Output for Fulfillment Status and Plotting ======
    if verbose and AA.n_not_fulfilled == 1:
        print(f'Only {AA.commanded[AA.not_fulfilled_idx[0]]} remains to be fulfilled')
    if show_plot:
        AA.plot_fig, AA.plot_ax = None, None
      
        
       
def get_commanded_indices(AA, compare_all_commanded):
    """Returns a list containing one commanded index to consider."""
    
    if compare_all_commanded:
        return list(AA.not_fulfilled_idx)  # Consider all unfulfilled points

    # Select a single commanded index in sequential order
    return [AA.not_fulfilled_idx[AA.i_commanded]]

def get_reference_antennas(AA, compare_all_antpos, order_antpos_by_magnitude, antpos_order):
    """Returns a list of reference antenna indices based on user settings."""
    
    if compare_all_antpos:
        return list(range(len(AA.antpos)))  # Consider all existing antennas

    
    if order_antpos_by_magnitude:
        sorted_indices = np.argsort(antpos_order * AA.antpos_norms)  # Sort by magnitude
        return [sorted_indices[AA.i_antpos]]
    elif antpos_order == 0:
        return [np.random.randint(len(AA.antpos))]  # Pick randomly
    else:
        return [antpos_order * AA.i_antpos]


def generate_candidate_combinations(AA, compare_all_commanded, compare_all_antpos, 
                                 order_antpos_by_magnitude, antpos_order):
    """Generates valid commanded-reference antenna pairs while filtering rejected ones."""
    
    commanded_idx = get_commanded_indices(AA, compare_all_commanded)
    antpos_idx = get_reference_antennas(AA,compare_all_antpos, order_antpos_by_magnitude, antpos_order)

    all_combinations = set(product(antpos_idx, commanded_idx, range(2)))
    remaining_combinations = list(all_combinations - set(AA.history["rejected_combinations"]))

    return remaining_combinations

def chunkify(AA, lst, n):
# This function is used for the parallelization
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance, diameter, max_array_size, center_at_origin):
    """Checks if a new antenna placement is valid based on collision and array size constraints."""
    
    # Step 1: Check for collisions before adding new_antpos
    if geometry.collision_check(np.vstack([antpos, new_antpos]), diameter):
        return False, False, np.inf  # Early exit if there's a collision

    # Step 2: Compute array size *only if no collision*
    temp_antpos = np.vstack([antpos, new_antpos])
    temp_array_size = geometry.get_array_size(temp_antpos)

    # Step 3: Check array size constraints
    if max_array_size is None:
        return True, True, temp_array_size  # No size constraint, always valid

    if center_at_origin:
        if np.linalg.norm(new_antpos) > max_array_size / 2:
            return True, False, temp_array_size  # Exceeds max size from origin
    else:
        if temp_array_size > max_array_size:
            return True, False, temp_array_size  # Exceeds max array size

    return True, True, temp_array_size  # Passes both collision and size constraints




def evaluate_candidates_parallel(AA, remaining_combinations):
    """Evaluates and selects the best antenna placement in parallel."""
    
    # Step 1: Split combinations into chunks for parallel processing
    chunks = list(chunkify(AA,remaining_combinations, max(len(remaining_combinations) // AA.num_cores, AA.num_cores)))
    args_for_starmap = [(chunk, AA.antpos, AA.commanded, AA.not_fulfilled_tree, AA.not_fulfilled_array,
                 AA.fulfill_tolerance, AA.diameter, AA.max_array_size, AA.center_at_origin, AA.minimize_array_size,
                 AA.maximize_antenna_spacing, try_new_antpos, geometry.get_min_distance_from_new_antpos, AA.uv_cell_size)
                for chunk in chunks]

    # Step 2: Run parallel evaluation
    with Pool(processes=AA.num_cores) as pool:
        results = pool.starmap(find_local_extrema, args_for_starmap)

    # Step 3: Process results to determine the best candidate
    best_fulfilled_count = -1
    best_spacing_metric = 0
    best_size_metric = np.inf
    best_candidate = None
    global_success = False

    for result in results:
        success, n_new_fulfilled, min_spacing, array_size, i, j, k, rejected_combos = result
        
        if success:
            global_success = True

        # Prioritize placements that fulfill the most commanded points
        if n_new_fulfilled > best_fulfilled_count:
            best_fulfilled_count = n_new_fulfilled
            best_candidate = (i, j, k)
            best_spacing_metric = min_spacing
            best_size_metric = array_size

        elif n_new_fulfilled == best_fulfilled_count:
            if AA.minimize_array_size and array_size < best_size_metric:
                best_candidate = (i, j, k)
                best_spacing_metric = min_spacing
                best_size_metric = array_size
            # Otherwise, prioritize placements with larger minimum spacing
            elif AA.maximize_antenna_spacing and min_spacing > best_spacing_metric:
                best_candidate = (i, j, k)
                best_size_metric = array_size

        # Store rejected combinations to avoid re-evaluating them
        if rejected_combos is not None:
            AA.history["rejected_combinations"].extend(rejected_combos)

    return global_success, best_candidate


def evaluate_candidate_sequential(AA, remaining_combinations):
    """Evaluates antenna placements in a sequential greedy manner."""

    global_success = False
    best_candidate = None

    for i, j, k in remaining_combinations:
        new_antpos = geometry.compute_new_antpos(i,j,k, AA.antpos, AA.commanded)
        collision_pass, array_size_pass, temp_array_size = try_new_antpos(antpos = AA.antpos,
                                                                                new_antpos = new_antpos,
                                                                                commanded = AA.commanded,
                                                                                fulfill_tolerance = AA.fulfill_tolerance,
                                                                                diameter = AA.diameter,
                                                                                max_array_size = AA.max_array_size,
                                                                                center_at_origin = AA.center_at_origin)
        
        if collision_pass and array_size_pass:
            global_success = True
            best_candidate = (i, j, k)
            break
        else:
            AA.history["rejected_combinations"].append((i, j, k))

    return global_success, best_candidate


def place_antenna(AA, best_candidate):
    """Adds the best antenna placement to the array and updates fulfillment tracking."""

    i, j, k = best_candidate  # Unpack candidate without new_antpos
    
    new_antpos = geometry.compute_new_antpos(i,j,k, AA.antpos, AA.commanded)  # Recompute new_antpos
    
    new_fulfilled = geometry.get_new_fulfilled(new_antpos = new_antpos,
                                               antpos = AA.antpos,
                                               not_fulfilled_tree = AA.not_fulfilled_tree,
                                               not_fulfilled_array = AA.not_fulfilled_array,
                                               fulfill_tolerance = AA.fulfill_tolerance,
                                               uv_cell_size = AA.uv_cell_size,
                                               )
    AA.antpos = np.vstack([AA.antpos, new_antpos])
    AA.n_added += 1
    AA.fulfilled_idx, AA.not_fulfilled_idx = geometry.check_fulfillment(
        commanded = AA.commanded,
        antpos = AA.antpos,
        fulfill_tolerance = AA.fulfill_tolerance,
    )
    AA.history["new_fulfilled"].append(new_fulfilled)
    AA.history['n_fulfilled'].append(len(AA.fulfilled_idx))
    AA.history["n_new_fulfilled"].append(len(new_fulfilled))
    AA.history["n_not_fulfilled"].append(AA.history["n_not_fulfilled"][-1] - len(new_fulfilled))
    AA.history["efficiency"].append(utils.get_efficiency(len(AA.fulfilled_idx),AA.antpos))
    
    AA.array_is_complete = len(AA.not_fulfilled_idx) == 0        
    AA.not_fulfilled_array = AA.commanded[AA.not_fulfilled_idx]
    AA.not_fulfilled_tree = KDTree(AA.not_fulfilled_array)
    AA.antpos_norms = np.linalg.norm(AA.antpos, axis=1)

def should_terminate(AA, compare_all_commanded, compare_all_antpos):
    """Determines whether the algorithm should terminate when no valid placement is found."""
    
    if compare_all_commanded and compare_all_antpos:
        return True  # All possible commanded and reference antennas have been checked

    if compare_all_commanded and not compare_all_antpos:
        if AA.i_antpos < len(AA.antpos) - 1:
            AA.i_antpos += 1  # Try next reference antenna
            return False
        return True  # Exhausted all reference antennas

    if compare_all_antpos and not compare_all_commanded:
        if AA.i_commanded < len(AA.not_fulfilled_idx) - 1:
            AA.i_commanded += 1  # Try next commanded point
            return False
        return True  # Exhausted all commanded points

    # Default case: iterate over both indices
    if AA.i_antpos < len(AA.antpos) - 1:
        AA.i_antpos += 1
        return False
    elif AA.i_commanded < len(AA.not_fulfilled_idx) - 1:
        AA.i_antpos = 0
        AA.i_commanded += 1
        return False
    return True  # No candidates left



def print_status(AA, additional_output = None, log_path=None):
    """
    Print and optionally log the current algorithm status.

    Parameters
    ----------
    AA : AntArray
        The current AntArray instance.
    log_path : str or None
        Path to the log file. If None, doesn't log to file.
    use_clear_output : bool
        If True, clears Jupyter output before printing (only use in notebooks).
    """
    num_antennas = len(AA.antpos)
    num_fulfilled = len(AA.commanded) - len(AA.not_fulfilled_idx)
    num_remaining = len(AA.not_fulfilled_idx)

    formatted_status = [
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"📡 Antennas in array: {num_antennas}",
        f"✅ uv points fulfilled: {num_fulfilled}/{len(AA.commanded)}",
        f"🗄️ uv points remaining: {num_remaining}/{len(AA.commanded)}",
        f" uv points newly fulfilled: {AA.history['n_new_fulfilled'][-1]}",
        "📈 array efficiency: {:.2f}".format(AA.history['efficiency'][-1]),
        "⏳ time for last iteration:", utils.format_time(AA.history['step_time'][-1]),
        "⌛ total time so far:", utils.format_time(np.array(AA.history['step_time']).sum()),
        "-" * 40
    ]
    
    status_str = "\n".join(formatted_status)



    if log_path:
        with open(log_path, "a") as f:
            f.write(status_str + "\n")
    else:
        clear_output(wait=True)
        if additional_output is not None:
            print(additional_output)
        print(status_str)

    
    
    
    
    
def find_local_extrema(chunk, antpos, commanded,  not_fulfilled_tree, not_fulfilled_array,
                       fulfill_tolerance, diameter, max_array_size, center_at_origin, minimize_array_size,
                       maximize_antenna_spacing, try_new_antpos, get_min_distance, uv_cell_size):
    """Evaluates candidate antenna placements within a chunk in parallel."""
    max_n_new_fulfilled = 0
    max_min_distance = 0
    favored_i = favored_j = favored_k = None
    success = False
    newly_rejected_combinations = []
    min_array_size = np.inf



    for i, j, k in chunk:
        new_antpos = geometry.compute_new_antpos(i,j,k, antpos, commanded)
            
        collision_pass, array_size_pass, temp_array_size = try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance, diameter, max_array_size, center_at_origin)
        
        if collision_pass and array_size_pass:
            success = True  
            new_fulfilled = geometry.get_new_fulfilled(new_antpos = new_antpos,
                                                       antpos = antpos,
                                                       not_fulfilled_tree = not_fulfilled_tree,
                                                       not_fulfilled_array = not_fulfilled_array,
                                                       fulfill_tolerance = fulfill_tolerance,
                                                       uv_cell_size = uv_cell_size)
            
            min_distance = get_min_distance(antpos, new_antpos)

            if len(new_fulfilled) > max_n_new_fulfilled:
                max_n_new_fulfilled = len(new_fulfilled)
                favored_i, favored_j, favored_k = i, j, k
                max_min_distance = min_distance
                min_array_size = temp_array_size
            elif len(new_fulfilled) == max_n_new_fulfilled:
                if minimize_array_size and temp_array_size < min_array_size:
                    min_array_size = temp_array_size
                    favored_i, favored_j, favored_k = i, j, k
                    min_array_size = temp_array_size
                elif maximize_antenna_spacing and min_distance > max_min_distance:
                    max_min_distance = min_distance
                    favored_i, favored_j, favored_k = i, j, k
        else:
            newly_rejected_combinations.append((i, j, k))

    return (success, max_n_new_fulfilled, max_min_distance, min_array_size, favored_i, favored_j, favored_k, newly_rejected_combinations)

        
        
    
