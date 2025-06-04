#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for uv-complete array construction.

Created on Thu Feb 15 12:27:51 2024
@author: Vincent
"""

import numpy as np
import uvrules.geometry as geometry
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


def select_baselines(commanded: np.ndarray, antpos: np.ndarray,
                     fulfill_tolerance: float) -> list:
    """
    Select baseline pairs that fulfill commanded uv points.

    Parameters
    ----------
    commanded : np.ndarray
        Commanded uv points.
    antpos : np.ndarray
        Antenna positions.
    fulfill_tolerance : float
        Distance tolerance to match uv points.

    Returns
    -------
    list of tuple
        List of antenna index pairs.
    """
    not_fulfilled = np.copy(commanded)
    antpairs = []

    for i, pos1 in enumerate(antpos):
        for j, pos2 in enumerate(antpos):
            if j <= i:
                continue
            u = pos2[0] - pos1[0]
            v = pos2[1] - pos1[1]
            if v == 0:
                u = np.abs(u)
            elif v < 0:
                u *= -1
                v *= -1
            uv = np.array([u, v])

            idx_new_fulfilled = np.where(np.linalg.norm(not_fulfilled - uv, axis=1, ord=np.inf) < fulfill_tolerance)

            if len(idx_new_fulfilled[0]) < 1:
                continue
            else:
                not_fulfilled = np.delete(not_fulfilled, idx_new_fulfilled, axis=0)
                antpairs.append((i, j))

            if len(not_fulfilled) % 1000 == 0 and len(not_fulfilled) > 0:
                print(f'{len(not_fulfilled)}/{len(commanded)} remaining, {len(antpairs)} baselines selected.')

            if len(not_fulfilled) < 1:
                print('Baseline selection complete.')
                return antpairs

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


def generate_commanded_points(uv_cell_size: float = 1.0,
                               min_bl: float = 10,
                               max_bl: float = 100,
                               show_plot: bool = True,
                               ax: plt.Axes = None) -> np.ndarray:
    """
    Generate a set of commanded uv points forming a half-annulus.

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


def get_baseline_counts(AA, verbose: bool = True) -> list:
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
        _, not_fulfilled_idx = geometry.check_fulfillment(
            commanded=AA.commanded,
            antpos=antpos_temp,
            fulfill_tolerance=AA.fulfill_tolerance,
        )
        baseline_counts.append(len(not_fulfilled_idx))
    return baseline_counts
