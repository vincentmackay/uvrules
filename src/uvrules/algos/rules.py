# rules.py

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count
import warnings
from itertools import product
from .. import geometry
from .. import plotting
from .. import utils
from datetime import datetime

def add_ant_rules(AA, **kwargs):
    """Public module-level entry point."""
    return AA.add_ant_rules(**kwargs)

def _add_ant_rules_implementation(
    AA,
    commanded_order=1,
    compare_all_commanded=False,
    compare_all_antpos=True,
    order_antpos_by_magnitude=False,
    antpos_order=1,
    center_at_origin=True,
    max_array_size=None,
    n_max_antennas=-1,
    n_to_add=-1,
    minimize_array_size=True,
    maximize_antenna_spacing=False,
    nsamples = 1,
    save_file=False,
    path_to_file=None,
    verbose=True,
    show_plot=False,
    additional_output=None,
    num_cores=None,
    log_path=None
):
    """
    Main function to populate an AntArray with antennas to fulfill commanded uv points.
    Implements the RULES (Regular UV Layout Engineering Strategy) placement logic.

    Parameters
    ----------
    AA : AntArray
        AntArray instance to modify.
    commanded_order : int
        +1 = nearest first, -1 = farthest first, 0 = random.
    compare_all_commanded : bool
        Evaluate all commanded uv points at each step.
    compare_all_antpos : bool
        Evaluate all antennas as references.
    order_antpos_by_magnitude : bool
        If True, sort reference antennas by distance from origin.
    antpos_order : int
        +1 = closest first, -1 = farthest first, 0 = random.
    center_at_origin : bool
        Whether to enforce circular bounding around origin.
    max_array_size : float or None
        Max array size in meters.
    n_max_antennas : int
        Stop after this many antennas added (if > 0).
    n_to_add : int
        Add at most this many antennas during the run.
    minimize_array_size : bool
        Break ties by minimizing array size.
    maximize_antenna_spacing : bool
        Break ties by maximizing minimum antenna spacing.
    nsamples : int
        Number of samples per baseline required for fulfillment.
    save_file : bool
        Whether to save AntArray to disk after each step.
    path_to_file : str or None
        File path to save to (default: timestamped).
    verbose : bool
        Print status updates.
    show_plot : bool
        Update 2x2 plot after each step.
    additional_output : str or None
        Extra text to include in console/log output.
    num_cores : int or None
        Number of CPU cores for parallel evaluation.
    log_path : str or None
        File path to log status updates.

    Returns
    -------
    None
    """

    prepare_algorithm(
        AA,
        commanded_order,
        compare_all_commanded,
        center_at_origin,
        max_array_size,
        minimize_array_size,
        maximize_antenna_spacing,
        nsamples,
        path_to_file,
        verbose,
        show_plot,
        num_cores
    )

    last_step_time = time.time()
    while not AA.array_is_complete:
        remaining_combinations = generate_candidate_combinations(
            AA,
            compare_all_commanded,
            compare_all_antpos,
            order_antpos_by_magnitude,
            antpos_order
        )

        if not remaining_combinations:
            print("Array is full for this set of parameters, quitting.")
            break

        if compare_all_commanded or compare_all_antpos:
            global_success, best_candidate = evaluate_candidates_parallel(AA, remaining_combinations)
        else:
            global_success, best_candidate = evaluate_candidate_sequential(AA, remaining_combinations)

        if not global_success:
            if should_terminate(AA, compare_all_commanded, compare_all_antpos):
                print("Array is full, terminating.")
                break
            continue

        place_antenna(AA, best_candidate)
        
        
        step_time = time.time() - last_step_time
        AA.history["step_time"].append(step_time)

        if save_file:
            AA.save(path_to_file)
            
        if verbose or show_plot:
            clear_output(wait=True)

        if verbose:
            print_status(AA, additional_output=additional_output, log_path=log_path)

        if show_plot:
            AA.plot_fig, AA.plot_ax = plotting.plot_history(AA, None, None)
            plt.pause(0.01)

        last_step_time = time.time()

        if AA.array_is_complete or (0 < n_to_add == AA.n_added) or (0 < n_max_antennas <= len(AA.antpos)):
            break

    print("Done.")




def prepare_algorithm(AA, commanded_order, compare_all_commanded, center_at_origin, max_array_size,
                      minimize_array_size, maximize_antenna_spacing, nsamples, path_to_file, verbose, show_plot, num_cores):
    """Prepare the AntArray for a RULES run."""
    is_default_antpos = np.array_equal(AA.antpos, np.array([[0., 0.]]))
    is_invalid_shape = AA.antpos.shape[1] != 2
    AA.starting_from_scratch = is_default_antpos or is_invalid_shape

    AA.nsamples = nsamples

    if AA.starting_from_scratch:
        AA.antpos = np.array([[0, 0]])
        AA.history = {
            "n_new_fulfilled": [0],
            "n_fulfilled": [0],
            "n_not_fulfilled": [len(AA.commanded)],
            "new_fulfilled": [],
            "step_time": [0.0],
            "efficiency": [1],
            "rejected_combinations": [],
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
            "rejected_combinations": [],
        }

    AA.path_to_file = path_to_file or f"./AntArray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    AA.num_cores = num_cores if isinstance(num_cores, int) else cpu_count()

    if verbose and compare_all_commanded:
        warnings.warn("Comparing all commanded points — this may take a long time.")

    if commanded_order == 0:
        AA.commanded = AA.commanded[np.random.permutation(AA.commanded.shape[0])]
    else:
        AA.commanded = AA.commanded[np.argsort(commanded_order * np.linalg.norm(AA.commanded, axis=1))]


    AA.nsamples = nsamples
    
    # Get fulfillment status and remaining fulfillments in one call
    AA.fulfilled_idx, AA.not_fulfilled_idx, AA.n_remaining_fulfillments = geometry.check_fulfillment(
        commanded=AA.commanded, antpos=AA.antpos, fulfill_tolerance=AA.fulfill_tolerance, nsamples=AA.nsamples
    )



    AA.array_is_complete = len(AA.not_fulfilled_idx) == 0
    AA.not_fulfilled_array = AA.commanded[AA.not_fulfilled_idx]
    AA.not_fulfilled_tree = KDTree(AA.not_fulfilled_array)

    AA.i_commanded = 0
    AA.i_antpos = 0
    AA.center_at_origin = center_at_origin
    AA.max_array_size = max_array_size or AA.max_array_size
    AA.minimize_array_size = minimize_array_size
    AA.maximize_antenna_spacing = maximize_antenna_spacing
    AA.antpos_norms = np.linalg.norm(AA.antpos, axis=1)
    AA.n_added = 0
    AA.flip_tolerance = 1e-5


    if show_plot:
        AA.plot_fig, AA.plot_ax = None, None




def get_commanded_indices(AA, compare_all_commanded):
    return list(AA.not_fulfilled_idx) if compare_all_commanded else [AA.not_fulfilled_idx[AA.i_commanded]]


def get_reference_antennas(AA, compare_all_antpos, order_antpos_by_magnitude, antpos_order):
    if compare_all_antpos:
        return list(range(len(AA.antpos)))
    if order_antpos_by_magnitude:
        sorted_indices = np.argsort(antpos_order * AA.antpos_norms)
        return [sorted_indices[AA.i_antpos]]
    if antpos_order == 0:
        return [np.random.randint(len(AA.antpos))]
    return [antpos_order * AA.i_antpos]




def generate_candidate_combinations(AA, compare_all_commanded, compare_all_antpos, order_antpos_by_magnitude, antpos_order):
    commanded_idx = get_commanded_indices(AA, compare_all_commanded)
    antpos_idx = get_reference_antennas(AA, compare_all_antpos, order_antpos_by_magnitude, antpos_order)
    all_combinations = set(product(antpos_idx, commanded_idx, range(2)))
    return list(all_combinations - set(AA.history["rejected_combinations"]))



def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance, diameter, max_array_size, center_at_origin):
    if geometry.collision_check(np.vstack([antpos, new_antpos]), diameter):
        return False, False, float('inf')
    temp_array_size = geometry.get_array_size(np.vstack([antpos, new_antpos]))
    if max_array_size is None:
        return True, True, temp_array_size
    if center_at_origin and np.linalg.norm(new_antpos) > max_array_size / 2:
        return True, False, temp_array_size
    if not center_at_origin and temp_array_size > max_array_size:
        return True, False, temp_array_size
    return True, True, temp_array_size




def evaluate_candidates_parallel(AA, remaining_combinations):
    chunks = list(chunkify(remaining_combinations,
                           max(len(remaining_combinations) // AA.num_cores, AA.num_cores)))
    args_for_starmap = [
        (chunk, AA.antpos, AA.commanded, AA.not_fulfilled_tree, AA.not_fulfilled_array,
         AA.fulfill_tolerance, AA.diameter, AA.max_array_size, AA.center_at_origin,
         AA.minimize_array_size, AA.maximize_antenna_spacing, AA.nsamples, try_new_antpos,
         geometry.get_min_distance_from_new_antpos, AA.uv_cell_size, AA.flip_tolerance,
         AA.not_fulfilled_idx, AA.n_remaining_fulfillments)
        for chunk in chunks
    ]
    
    with Pool(processes=AA.num_cores) as pool:
        results = pool.starmap(find_local_extrema, args_for_starmap)

    best_fulfilled_count = -1
    best_spacing_metric = 0
    best_size_metric = float('inf')
    best_candidate = None
    global_success = False

    for result in results:
        success, n_new, min_spacing, array_size, i, j, k, rejected = result
        if success:
            global_success = True
        if n_new > best_fulfilled_count:
            best_candidate = (i, j, k)
            best_fulfilled_count = n_new
            best_spacing_metric = min_spacing
            best_size_metric = array_size
        elif n_new == best_fulfilled_count:
            if AA.minimize_array_size and array_size < best_size_metric:
                best_candidate = (i, j, k)
                best_size_metric = array_size
            elif AA.maximize_antenna_spacing and min_spacing > best_spacing_metric:
                best_candidate = (i, j, k)
                best_spacing_metric = min_spacing
        AA.history["rejected_combinations"].extend(rejected)

    return global_success, best_candidate



def evaluate_candidate_sequential(AA, remaining_combinations):
    global_success = False
    best_candidate = None

    for i, j, k in remaining_combinations:
        new_antpos = geometry.compute_new_antpos(i, j, k, AA.antpos, AA.commanded)
        collision, valid_size, _ = try_new_antpos(
            antpos=AA.antpos,
            new_antpos=new_antpos,
            commanded=AA.commanded,
            fulfill_tolerance=AA.fulfill_tolerance,
            diameter=AA.diameter,
            max_array_size=AA.max_array_size,
            center_at_origin=AA.center_at_origin,
        )
        if collision and valid_size:
            return True, (i, j, k)
        AA.history["rejected_combinations"].append((i, j, k))

    return global_success, best_candidate


def place_antenna_old(AA, best_candidate):
    i, j, k = best_candidate
    
    new_antpos = geometry.compute_new_antpos(i, j, k, AA.antpos, AA.commanded)
    new_fulfilled = geometry.get_new_fulfilled(
        new_antpos=new_antpos,
        antpos=AA.antpos,
        not_fulfilled_tree=AA.not_fulfilled_tree,
        not_fulfilled_array=AA.not_fulfilled_array,
        fulfill_tolerance=AA.fulfill_tolerance,
        uv_cell_size=AA.uv_cell_size,
        nsamples = AA.nsamples
    )

    AA.antpos = np.vstack([AA.antpos, new_antpos])
    AA.n_added += 1
    AA.fulfilled_idx, AA.not_fulfilled_idx = geometry.check_fulfillment(
        commanded=AA.commanded, antpos=AA.antpos, fulfill_tolerance=AA.fulfill_tolerance, nsamples = AA.nsamples
    )
    AA.history["new_fulfilled"].append(new_fulfilled)
    AA.history["n_fulfilled"].append(len(AA.fulfilled_idx))
    AA.history["n_new_fulfilled"].append(len(new_fulfilled))
    AA.history["n_not_fulfilled"].append(AA.history["n_not_fulfilled"][-1] - len(new_fulfilled))
    AA.history["efficiency"].append(utils.get_efficiency(len(AA.fulfilled_idx), AA.antpos))
    AA.array_is_complete = len(AA.not_fulfilled_idx) == 0
    AA.not_fulfilled_array = AA.commanded[AA.not_fulfilled_idx]
    AA.not_fulfilled_tree = KDTree(AA.not_fulfilled_array)
    AA.antpos_norms = np.linalg.norm(AA.antpos, axis=1)
    
def place_antenna(AA, best_candidate):
    i, j, k = best_candidate
    
    new_antpos = geometry.compute_new_antpos(i, j, k, AA.antpos, AA.commanded)
    
    # Get new fulfilled points and score
    new_fulfilled, fulfillment_score = geometry.get_new_fulfilled(
            new_antpos=new_antpos,
            antpos=AA.antpos,
            not_fulfilled_tree=AA.not_fulfilled_tree,
            not_fulfilled_array=AA.not_fulfilled_array,
            fulfill_tolerance=AA.fulfill_tolerance,
            uv_cell_size=AA.uv_cell_size,
            nsamples=AA.nsamples,
            not_fulfilled_idx=AA.not_fulfilled_idx,
            n_remaining_fulfillments=AA.n_remaining_fulfillments)

    # Place the antenna
    AA.antpos = np.vstack([AA.antpos, new_antpos])
    AA.n_added += 1
    
    # Update fulfillment status and remaining fulfillments in one efficient call
    AA.fulfilled_idx, AA.not_fulfilled_idx, AA.n_remaining_fulfillments = geometry.check_fulfillment(
        commanded=AA.commanded, antpos=AA.antpos, fulfill_tolerance=AA.fulfill_tolerance, nsamples=AA.nsamples
    )
    
    # Update history
    AA.history["new_fulfilled"].append(new_fulfilled)
    AA.history["n_fulfilled"].append(len(AA.fulfilled_idx))
    AA.history["n_new_fulfilled"].append(len(new_fulfilled))
    AA.history["n_not_fulfilled"].append(AA.history["n_not_fulfilled"][-1] - len(new_fulfilled))
    AA.history["efficiency"].append(utils.get_efficiency(len(AA.fulfilled_idx), AA.antpos))
    
    # Update array completion status and spatial structures
    AA.array_is_complete = len(AA.not_fulfilled_idx) == 0
    AA.not_fulfilled_array = AA.commanded[AA.not_fulfilled_idx]
    AA.not_fulfilled_tree = KDTree(AA.not_fulfilled_array)
    AA.antpos_norms = np.linalg.norm(AA.antpos, axis=1)


def should_terminate(AA, compare_all_commanded, compare_all_antpos):
    if compare_all_commanded and compare_all_antpos:
        return True
    if compare_all_commanded and not compare_all_antpos:
        if AA.i_antpos < len(AA.antpos) - 1:
            AA.i_antpos += 1
            return False
        return True
    if compare_all_antpos and not compare_all_commanded:
        if AA.i_commanded < len(AA.not_fulfilled_idx) - 1:
            AA.i_commanded += 1
            return False
        return True
    if AA.i_antpos < len(AA.antpos) - 1:
        AA.i_antpos += 1
        return False
    if AA.i_commanded < len(AA.not_fulfilled_idx) - 1:
        AA.i_commanded += 1
        AA.i_antpos = 0
        return False
    return True

def print_status(AA, additional_output=None, log_path=None):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status = [
        f"[{now}]",
        f"📡 Antennas in array: {len(AA.antpos)}",
        "Last antenna placed: ({:.2f} m , {:.2f} m)".format(AA.antpos[-1][0],AA.antpos[-1][1]),
        f"✅ # uv points fulfilled: {len(AA.fulfilled_idx)}/{len(AA.commanded)}",
        f"🗄️ # uv points remaining: {len(AA.not_fulfilled_idx)}/{len(AA.commanded)}",
        f"🆕 # uv points fulfilled at last step: {AA.history['n_new_fulfilled'][-1]}",
        f"📈 efficiency: {AA.history['efficiency'][-1]:.2f}",
        f"⏳ last step: {utils.format_time(AA.history['step_time'][-1])}",
        f"⌛ total time: {utils.format_time(np.sum(AA.history['step_time']))}",
        "-" * 40
    ]
    status_str = "\n".join(status)
    if log_path:
        with open(log_path, "a") as f:
            f.write(status_str + "\n")
    else:
        if additional_output:
            print(additional_output)
        print(status_str)
        
        
def find_local_extrema_old(chunk, antpos, commanded, not_fulfilled_tree, not_fulfilled_array,
                       fulfill_tolerance, diameter, max_array_size, center_at_origin,
                       minimize_array_size, maximize_antenna_spacing, nsamples, try_new_antpos,
                       get_min_distance, uv_cell_size, flip_tolerance):
    
    max_n_new = 0
    max_spacing = 0
    best_i = best_j = best_k = None
    min_array_size = float('inf')
    success = False
    rejected = []

    for i, j, k in chunk:
        new_antpos = geometry.compute_new_antpos(i, j, k, antpos, commanded)
        collision, valid_size, temp_size = try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance,
                                                           diameter, max_array_size, center_at_origin)
        if collision and valid_size:
            success = True
            new_fulfilled = geometry.get_new_fulfilled(new_antpos, antpos, not_fulfilled_tree, not_fulfilled_array,
                                                       fulfill_tolerance, uv_cell_size, nsamples, flip_tolerance)
            spacing = get_min_distance(antpos, new_antpos)
            if len(new_fulfilled) > max_n_new:
                best_i, best_j, best_k = i, j, k
                max_n_new = len(new_fulfilled)
                max_spacing = spacing
                min_array_size = temp_size
            elif len(new_fulfilled) == max_n_new:
                if minimize_array_size and temp_size < min_array_size:
                    best_i, best_j, best_k = i, j, k
                    min_array_size = temp_size
                elif maximize_antenna_spacing and spacing > max_spacing:
                    best_i, best_j, best_k = i, j, k
                    max_spacing = spacing
        else:
            rejected.append((i, j, k))

    return (success, max_n_new, max_spacing, min_array_size, best_i, best_j, best_k, rejected)

def find_local_extrema(chunk, antpos, commanded, not_fulfilled_tree, not_fulfilled_array,
                       fulfill_tolerance, diameter, max_array_size, center_at_origin,
                       minimize_array_size, maximize_antenna_spacing, nsamples, try_new_antpos,
                       get_min_distance, uv_cell_size, flip_tolerance, not_fulfilled_idx=None, 
                       n_remaining_fulfillments=None):
    
    max_fulfillment_score = 0  # Changed from max_n_new
    max_spacing = 0
    best_i = best_j = best_k = None
    min_array_size = float('inf')
    success = False
    rejected = []

    for i, j, k in chunk:
        new_antpos = geometry.compute_new_antpos(i, j, k, antpos, commanded)
        collision, valid_size, temp_size = try_new_antpos(antpos, new_antpos, commanded, fulfill_tolerance,
                                                           diameter, max_array_size, center_at_origin)
        if collision and valid_size:
            success = True
            
            # Use new get_new_fulfilled that returns fulfillment_score
            
            new_fulfilled, fulfillment_score = geometry.get_new_fulfilled(
                new_antpos = new_antpos, 
                antpos = antpos, 
                not_fulfilled_tree = not_fulfilled_tree,
                not_fulfilled_array = not_fulfilled_array,
                fulfill_tolerance = fulfill_tolerance,
                uv_cell_size = uv_cell_size,
                nsamples = nsamples,
                flip_tolerance = flip_tolerance,
                not_fulfilled_idx = not_fulfilled_idx,
                n_remaining_fulfillments = n_remaining_fulfillments
            )
            
            spacing = get_min_distance(antpos, new_antpos)
            
            if fulfillment_score > max_fulfillment_score:
                best_i, best_j, best_k = i, j, k
                max_fulfillment_score = fulfillment_score
                max_spacing = spacing
                min_array_size = temp_size
            elif fulfillment_score == max_fulfillment_score:
                if minimize_array_size and temp_size < min_array_size:
                    best_i, best_j, best_k = i, j, k
                    min_array_size = temp_size
                elif maximize_antenna_spacing and spacing > max_spacing:
                    best_i, best_j, best_k = i, j, k
                    max_spacing = spacing
        else:
            rejected.append((i, j, k))

    return (success, max_fulfillment_score, max_spacing, min_array_size, best_i, best_j, best_k, rejected)
