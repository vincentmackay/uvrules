#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for uv-complete array construction.

Created on Thu Feb 15 12:27:51 2024
@author: Vincent
"""

import numpy as np
from . import geometry
import matplotlib.pyplot as plt
import copy
from datetime import timedelta
from IPython.display import clear_output

random_seed = 11141  # Fixed random seed for reproducibility


def get_uvs_from_baseline_select(array_config: dict, baseline_select: list,
                                  flip_tolerance: float = 0.0) -> np.ndarray:
    """
    Compute uv points from selected baseline pairs.

    Parameters
    ----------
    array_config : dict
        Mapping of antenna indices to positions {i: (u, v)}.
    baseline_select : list of tuple
        List of (i, j) antenna index pairs.
    flip_tolerance : float, default=0.0
        Tolerance for flipping uv points.

    Returns
    -------
    np.ndarray
        Array of uv points, shape (N_baselines, 2).
    """
    uvs = []
    for bl in baseline_select:
        u = array_config[bl[1]][0] - array_config[bl[0]][0]
        v = array_config[bl[1]][1] - array_config[bl[0]][1]
        if v == 0:
            u = np.abs(u)
        elif v < 0:
            u *= -1
            v *= -1
        uvs.append([u, v])
    return np.asarray(uvs)


def select_baselines(AA = None,
                     commanded = None,
                     antpos = None,
                     use_commanded = True,
                     max_bl = None,
                     min_bl = None,
                     fulfill_tolerance = None,
                     flip_tolerance = 0.0,
                     avoid_redundancies = True) -> list:
    """
    Select baseline pairs that fulfill commanded uv points.

    Parameters
    ----------
    AA : AntArray, optional
        AntArray object containing commanded uv points and antenna positions.
    commanded : np.ndarray, optional
        Commanded uv points.
    antpos : np.ndarray, optional
        Antenna positions.
    use_commanded : bool, default=True
        If True, use commanded uv points; otherwise, do all points within min_bl and max_bl.
    max_bl : float, optional
        Maximum baseline length to consider.
    min_bl : float, optional
        Minimum baseline length to consider.
    fulfill_tolerance : float, default = 1e-5
        Distance tolerance to match uv points.
    flip_tolerance : float, optional
        Tolerance for flipping uv points.
    avoid_redundancies : bool, default=True
        If True, avoid selecting redundant baselines.

    Returns
    -------
    list of tuple
        List of antenna index pairs.
    """

    if AA is not None:
        if commanded is None:
            commanded = AA.commanded
        if antpos is None:
            antpos = AA.antpos
        if fulfill_tolerance is None:
            fulfill_tolerance = AA.fulfill_tolerance
        if min_bl is None:
            min_bl = AA.min_bl
        if max_bl is None:
            max_bl = AA.max_bl
    elif commanded is None or antpos is None or fulfill_tolerance is None:
        raise ValueError("Either AA or commanded, antpos, and fulfill_tolerance must be provided.")


    fi, nfi, _ = AA.check_fulfillment()
    commanded_fulfilled = commanded[fi]
    not_fulfilled = np.copy(commanded_fulfilled)
    antpairs = []

    if not avoid_redundancies:
        n_total_baselines = len(antpos) * (len(antpos) - 1) // 2

    if not use_commanded:
        uv_list = []




    for i, pos1 in enumerate(antpos):
        for j, pos2 in enumerate(antpos):
            if j <= i:
                continue

            pos1= antpos[i]
            pos2 = antpos[j]
            u = pos2[0] - pos1[0]
            v = pos2[1] - pos1[1]
            if (v < -flip_tolerance) | ((np.abs(v) <= flip_tolerance) & (u < 0)):
                u *= -1
                v *= -1
            uv = np.array([u, v])
            length = np.linalg.norm(uv)

            if min_bl is not None and length < min_bl:
                continue
            elif max_bl is not None and length > max_bl:
                continue
            
            if use_commanded:

                if avoid_redundancies: # Check whether the new uv point is one of the unfulfilled one
                    idx_new_uv = np.where(np.linalg.norm(not_fulfilled - uv, axis=1, ord=np.inf) < fulfill_tolerance)
                else: # Just check whether the new uv point is commanded
                    idx_new_uv = np.where(np.linalg.norm(commanded_fulfilled - uv, axis=1, ord=np.inf) < fulfill_tolerance)

                if len(idx_new_uv[0]) < 1:
                    continue
                else:
                    if avoid_redundancies:
                        not_fulfilled = np.delete(not_fulfilled, idx_new_uv, axis=0)
                    antpairs.append((i, j))

                if len(antpairs) % 1000 == 0:
                    if avoid_redundancies:
                        print(f'{len(not_fulfilled)}/{len(commanded_fulfilled)} remaining, {len(antpairs)} baselines selected.')
                    else:
                        print(f'{len(antpairs)}/{n_total_baselines} done.')
                if len(not_fulfilled) < 1 and avoid_redundancies:
                    print('Baseline selection complete.')
                    return antpairs
            else:
                if avoid_redundancies and len(uv_list) > 0:
                    # Check whether the new uv point is already in the list
                    if np.any(np.linalg.norm(np.asarray(uv_list) - uv, axis=1) < fulfill_tolerance):
                        continue
                    else:
                        if len(uv_list) % 1000 == 0:
                            print(f'{len(antpairs)} baselines selected so far.')
                        antpairs.append((i, j))
                        uv_list.append(uv)
                else:
                    antpairs.append((i, j))
                    uv_list.append(uv)

    print('All baselines done.')
    return antpairs


def get_array_config(antpos: np.ndarray) -> dict:
    """
    Build a dictionary mapping antenna indices to positions.

    Parameters
    ----------
    antpos : np.ndarray
        Antenna positions.

    Returns
    -------
    dict
        Dictionary {index: (u, v)}.
    """
    if antpos.ndim == 1:
        return {0: [antpos[0], antpos[1]]}
    return {i: [antpos[i, 0], antpos[i, 1]] for i in range(len(antpos))}


def generate_commanded_square(uv_cell_size: float = 1.0,
                               min_bl: float = 10,
                               max_bl: float = 100,
                               show_plot: bool = True,
                               ax: plt.Axes = None) -> np.ndarray:
    """
    Generate a set of commanded uv points forming a half-annulus on a square grid.

    Parameters
    ----------
    uv_cell_size : float, default=1.0
        Grid cell size.
    min_bl : float, default=10
        Minimum baseline distance.
    max_bl : float, default=100
        Maximum baseline distance.
    show_plot : bool, default=True
        If True, plot the commanded points.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot on; if None, create new figure.

    Returns
    -------
    np.ndarray
        Generated uv points, sorted by radius.
    """
    uv_points = []

    for u in np.arange(0, max_bl + uv_cell_size, uv_cell_size):
        for v in np.arange(0, max_bl + uv_cell_size, uv_cell_size):
            distance = np.sqrt(u**2 + v**2)
            if min_bl < distance <= max_bl:
                uv_points.append((u, v))

    for u in np.arange(-uv_cell_size, -max_bl - uv_cell_size, -uv_cell_size):
        for v in np.arange(uv_cell_size, max_bl + uv_cell_size, uv_cell_size):
            distance = np.sqrt(u**2 + v**2)
            if min_bl < distance <= max_bl:
                uv_points.append((u, v))

    uv_points = np.asarray(uv_points)

    if show_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(uv_points[:, 0], uv_points[:, 1], '.', markersize=1, color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Commanded uv points')
        ax.set_xlabel(r'$u$ [m]')
        ax.set_ylabel(r'$v$ [m]')
        ax.grid(True)

    uv_points = uv_points[np.argsort(np.linalg.norm(uv_points, axis=1))]
    return uv_points


def generate_commanded_hexa(min_bl_lambda=10, max_bl_lambda=100, packing_density=2.):
    """
    Generate a set of commanded uv points forming a half-annulus on a hexagonal grid.

    Parameters
    ----------
        min_bl_lambda : float, default=10
            Minimum baseline length in wavelengths.
        max_bl_lambda : float, default=100
            Maximum baseline length in wavelengths.
        packing_density : float, default=2.
            Packing density of the hexagonal grid.


    Returns
    -------
    np.ndarray
        Generated uv points, sorted by radius.
    """
    spacing = 1 / packing_density
    row_height = spacing * np.sqrt(3) / 2
    
    estimated_size = int(np.pi * (max_bl_lambda**2 - min_bl_lambda**2) * packing_density**2 * 1.2)
    uv_points = np.zeros((estimated_size, 2))
    count = 0
    
    row = 0
    while True:
        y = row * row_height
        if y > max_bl_lambda:
            break
            
        # Calculate row offset for triangular pattern
        x_offset = (spacing / 2) * (row % 2)
        
        if row == 0:
            # First row: start at x=0, only go positive
            x = 0
        else:
            # Other rows: start from left edge
            x = -max_bl_lambda + x_offset
        
        
        while x <= max_bl_lambda:
            uv = np.array([x, y])
            norm = np.linalg.norm(uv)
            
            if norm >= min_bl_lambda and norm <= max_bl_lambda:
                uv_points[count] = uv
                count += 1
            
            x += spacing
        
        row += 1
    
    uv_points = uv_points[:count]
    uv_points = uv_points[np.argsort(np.linalg.norm(uv_points, axis=1))]
    return uv_points

def generate_commanded_radexp(min_bl_lambda=10, max_bl_lambda=100, 
                                     exp_base=1.1, radial_density=1.0, angular_density=1.0):
    angular_spacing = np.pi / (angular_density * 20)
    # Include angle = 0 (positive u-axis) but exclude angle = pi (negative u-axis)
    angles = np.arange(0, np.pi, angular_spacing)
    
    uv_points = []
    
    for angle in angles:
        radial_step = 1.0 / radial_density
        r = min_bl_lambda
        radial_index = 0
        
        while r <= max_bl_lambda:
            if r >= min_bl_lambda:
                u = r * np.cos(angle)
                v = r * np.sin(angle)
                
                uv_points.append([u, v])
            
            radial_index += radial_step
            r = min_bl_lambda * (exp_base ** radial_index)
    
    return np.array(uv_points) if uv_points else np.array([]).reshape(0, 2)

def generate_commanded_radexp_gridded(min_bl_lambda=10, max_bl_lambda=100, 
                                       exp_base=1.1, radial_density=1.0, 
                                       angular_density=1.0, cell_size=1.0):
    angular_spacing = np.pi / (angular_density * 20)
    angles = np.arange(0, np.pi, angular_spacing)
    
    commanded = []
    
    for angle in angles:
        radial_step = 1.0 / radial_density
        r = min_bl_lambda
        radial_index = 0
        
        # For each spoke, keep track of used grid positions to avoid duplicates
        used_positions = set()
        
        while r <= max_bl_lambda:
            if r >= min_bl_lambda:
                # Calculate ideal position
                u_ideal = r * np.cos(angle)
                v_ideal = r * np.sin(angle)
                
                # Snap to grid
                i = round(u_ideal / cell_size)
                j = round(v_ideal / cell_size)
                u_grid = i * cell_size
                v_grid = j * cell_size
                
                # Check if this grid position is already used on this spoke
                grid_pos = (i, j)
                if grid_pos not in used_positions:
                    # Verify the gridded point is still in valid range
                    r_grid = np.sqrt(u_grid**2 + v_grid**2)
                    if min_bl_lambda <= r_grid <= max_bl_lambda:
                        commanded.append([u_grid, v_grid])
                        used_positions.add(grid_pos)
            
            radial_index += radial_step
            r = min_bl_lambda * (exp_base ** radial_index)
    
    return np.array(commanded) if commanded else np.array([]).reshape(0, 2)



def generate_commanded_gridded_spokes(packing_density=2.0, min_bl_lambda=10, max_bl_lambda=100,
                                     min_points_per_spoke=10):
    """
    Generate points on a square grid, keeping only those that form radial spokes.
    
    Parameters:
    - packing_density: grid density (grid_size = 1/packing_density)
    - min_bl_lambda, max_bl_lambda: annulus bounds
    - min_points_per_spoke: minimum points required to keep a spoke
    """
    uv_points = generate_commanded_square(min_bl=min_bl_lambda,max_bl=max_bl_lambda,
                                          uv_cell_size=1.0/packing_density, show_plot=False)
    
    ratios = uv_points[:,0] / uv_points[:,1]

    # Create groups of points that share the same ratio
    unique_ratios, inverse_indices = np.unique(ratios, return_inverse=True)
    counts = np.bincount(inverse_indices)

    # Keep only points from groups with more than min_points_per_spoke
    valid_indices = np.where(counts[inverse_indices] >= min_points_per_spoke)[0]
    commanded = uv_points[valid_indices]
    
    # Sort by radius
    if len(commanded) > 0:
        commanded = commanded[np.argsort(np.linalg.norm(commanded, axis=1))]
    
    return commanded



def get_efficiency(n_fulfilled: int, antpos: np.ndarray) -> float:
    """
    Compute array efficiency given number of fulfilled uv points.

    Parameters
    ----------
    n_fulfilled : int
        Number of uv points fulfilled.
    antpos : np.ndarray
        Antenna positions.

    Returns
    -------
    float
        Efficiency metric.
    """
    if len(antpos) <= 1:
        return 1.0
    n_baselines = len(antpos) * (len(antpos) - 1) / 2
    return (n_fulfilled / n_baselines)**0.5


def get_efficiency_array(commanded: np.ndarray, antpos: np.ndarray,
                         n_not_fulfilled_list: list) -> list:
    """
    Compute efficiency evolution over antenna additions.

    Parameters
    ----------
    commanded : np.ndarray
        Commanded uv points.
    antpos : np.ndarray
        Antenna positions.
    n_not_fulfilled_list : list
        Number of unfulfilled points at each step.

    Returns
    -------
    list of float
        Efficiency values per antenna number.
    """
    efficiency_array = []
    for i in range(len(antpos)):
        n_fulfilled = len(commanded) - n_not_fulfilled_list[i]
        n_baselines = i * (i - 1) / 2
        if n_baselines == 0:
            efficiency_array.append(1.0)
        else:
            efficiency_array.append((n_fulfilled / n_baselines) ** 0.5)
    return efficiency_array


def shuffle_antpos(antpos: np.ndarray, diameter: float,
                   tightness: float = 8, max_n_attempts: int = int(1e6),
                   verbose: bool = True) -> np.ndarray:
    """
    Shuffle antenna positions randomly while avoiding collisions.

    Parameters
    ----------
    antpos : np.ndarray
        Original antenna positions.
    diameter : float
        Dish diameter (meters).
    tightness : float, default=8
        Controls the clustering tightness.
    max_n_attempts : int, default=1e6
        Maximum attempts to place antennas.
    verbose : bool, default=True
        If True, print progress.

    Returns
    -------
    np.ndarray
        Shuffled antenna positions.
    """
    shuffled_antpos = np.asarray([[0, 0]])
    array_size = geometry.get_array_size(antpos)
    n_attempts = 0

    while len(shuffled_antpos) < len(antpos) and n_attempts < max_n_attempts:
        if verbose and n_attempts > 0.9 * max_n_attempts and n_attempts % 10 == 0:
            print(f'\nAt attempt {n_attempts}/{max_n_attempts}... ', end='')

        theta = 2 * np.pi * np.random.random()
        r = np.abs(np.random.normal()) * (array_size / tightness)
        u, v = r * np.cos(theta), r * np.sin(theta)
        new_antpos = np.array([u, v])

        if np.linalg.norm(new_antpos) < array_size / 2:
            temp_antpos = np.vstack([shuffled_antpos, new_antpos])
            if not geometry.collision_check(temp_antpos, diameter):
                shuffled_antpos = temp_antpos
                if verbose and len(shuffled_antpos) % 10 == 0:
                    print(f'Placed {len(shuffled_antpos)}/{len(antpos)} antennas...', end='')
                n_attempts = 0
            else:
                n_attempts += 1
        else:
            n_attempts += 1

    if verbose:
        print('\nDone.')
    return shuffled_antpos


def nudge_antpos(antpos: np.ndarray, diameter: float, fulfill_tolerance: float) -> np.ndarray:
    """
    Slightly nudge antenna positions to reduce collisions.

    Parameters
    ----------
    antpos : np.ndarray
        Antenna positions.
    diameter : float
        Dish diameter.
    fulfill_tolerance : float
        Maximum displacement for nudging.

    Returns
    -------
    np.ndarray
        Nudged antenna positions.
    """
    for i in range(len(antpos)):
        success = False
        while not success:
            nudge_u = (np.random.random() - 0.5) * 2 * fulfill_tolerance
            nudge_v = (np.random.random() - 0.5) * 2 * fulfill_tolerance
            temp_antpos = copy.deepcopy(antpos)
            temp_antpos[i, 0] += nudge_u
            temp_antpos[i, 1] += nudge_v
            if not geometry.collision_check(temp_antpos, diameter):
                antpos = temp_antpos
                success = True
    return antpos


def generate_position_errors(antpos: np.ndarray, diameter: float, sigma_pos: float = 0.1, max_attempts: int = 100000) -> np.ndarray:
    """
    Generate random position errors for antenna positions.
    """
    result_antpos = antpos.copy()  # Shallow copy
    collision_count = 0
    
    for i in range(len(antpos)):
        original_pos = result_antpos[i].copy()
        while True:
            # Generate nudge
            nudge = np.random.uniform(-sigma_pos, sigma_pos, 2)
            result_antpos[i] = original_pos + nudge
            
            # Only check if THIS antenna collides with others
            # Create array with just this antenna and all others
            if not geometry.collision_check_single_antenna(result_antpos, i, diameter):
                break  # Success
            else:
                collision_count += 1
                if collision_count % 1000 == 0:
                    print(f"Collision count: {collision_count}")
                    clear_output(wait=True)
        
    
    return result_antpos




def format_time(seconds: float) -> str:
    """
    Format time from seconds to HH:MM:SS.

    Parameters
    ----------
    seconds : float
        Time duration in seconds.

    Returns
    -------
    str
        Formatted time string.
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"


def export_antpos_csv(antpos: np.ndarray, path: str, include_index: bool = False):
    """
    Export antenna positions to a CSV file.

    Parameters
    ----------
    antpos : np.ndarray
        Antenna positions, shape (N, 2).
    path : str
        Output path.
    include_index : bool, default=False
        Whether to include an antenna index column.
    """
    if include_index:
        data = np.hstack([np.arange(1, len(antpos) + 1)[:, None], antpos])
    else:
        data = antpos
    np.savetxt(path, data, delimiter=",", fmt="%.8f")


def get_baseline_counts(AA, nsamples = 1, verbose: bool = True) -> list:
    """
    Calculate number of unfulfilled baselines when removing each antenna.

    Parameters
    ----------
    AA : AntArray
        AntArray object.
    verbose : bool, default=True
        Whether to display progress.

    Returns
    -------
    list
        List of number of missing baselines per antenna.
    """
    baseline_counts = []
    for i in range(len(AA.antpos)):
        if verbose:
            clear_output(wait=True)
            print(f'{i+1}/{len(AA.antpos)}')
        antpos_temp = np.delete(AA.antpos, i, axis=0)
        _, not_fulfilled_idx, n_remaining_fulfillments  = geometry.check_fulfillment(
            commanded=AA.commanded,
            antpos=antpos_temp,
            fulfill_tolerance=AA.fulfill_tolerance,
            nsamples = nsamples
        )
        baseline_counts.append(np.sum(n_remaining_fulfillments))
    return baseline_counts



def array_config_to_array_layout(array_config,filename = None):
    """
    Write array_layout file from array_config object.

    Parameters
    ----------
    array_config : dict
        A dictionary mapping antenna indices to their positions.
    filename : str, optional
        The name of the output CSV file. If None, a default name based on the current date and time will be used.
        
    Returns
    -------
     None
    """

    
    # Define the header
    header = "Name\tNumber\tBeamID\tE\tN\tU\n"

    # If filename is provided, use it; otherwise, generate a default name
    if filename is not None:
        output_filename = filename + '.csv'
    else: # Name the array based on date
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"array_layout_{date_str}.csv"
    
    # Open the file to write
    with open(output_filename, "w") as file:
        file.write(header)  # Write the header


        for key, value in array_config.items():
            # Ensure the value has at least two components (E, N) and handle optional U
            E = value[0]
            N = value[1]
            U = value[2] if len(value) > 2 else 0.0

            # Write the formatted line
            file.write(f"ANT{key+1}\t{key}\t0\t{E:.8f}\t{N:.8f}\t{U:.8f}\n")



def baseline_select_to_string(baseline_select):
    """ Convert a list of baseline pairs to a string representation.

    Parameters
    ----------
    baseline_select : list of tuples representing baseline pairs

    Returns
    -------
    bs_str : str
        String representation of the baseline pairs in the format [(i,j),(i,j),...].
    """
    bs_str = '['
    for row in baseline_select:
        bs_str += f'({row[0]},{row[1]}),'
    bs_str = bs_str[:-1] + ']'
    return bs_str
