#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:35:28 2025

@author: vincent
"""

import numpy as np
from datetime import datetime
from ..geometry import collision_check
from .. import plotting
from IPython.display import clear_output

def add_ant_random(AA, n_to_add=None, n_total=100,
                   min_radius=0.0, max_radius=1e3,
                   on_grid=False, sigma=None,
                   save_file=False, path_to_file=None,
                   show_plot=False, verbose=False):
    """
    Add antennas randomly within radius constraints, avoiding collisions.

    Parameters
    ----------
    AA : AntArray
        Instance of the AntArray class.
    n_to_add : int or None
        Number of antennas to add. If None, determined from n_total - len(AA.antpos).
    n_total : int
        Total number of antennas desired (ignored if n_to_add is specified).
    min_radius : float
        Minimum radius from origin for new antennas.
    max_radius : float
        Maximum radius from origin for new antennas.
    on_grid : bool
        If True, antennas are placed on a grid.
    sigma : float or None
        If provided and positive, sample x and y from a Gaussian centered at 0 with this std deviation.
    save_file : bool
        If True, saves AA object at each step.
    path_to_file : str or None
        File path for saving the AA object.
    show_plot : bool
        If True, shows array progress after each addition.
    verbose : bool
        If True, prints progress messages.
    """

    if sigma is not None:
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number or None")

    if path_to_file is None:
        path_to_file = './AntArray_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'

    if np.array_equal(AA.antpos, np.array([[0., 0.]])):
        AA.antpos = np.array([[0., 0.]])
    elif AA.antpos.shape[1] != 2:
        raise ValueError("AA.antpos must be of shape (n, 2)")

    existing_n = len(AA.antpos)
    if n_to_add is None:
        n_to_add = max(0, n_total - existing_n)

    n_added = 0
    while n_added < n_to_add:
        if sigma is None:
            x = np.random.uniform(-max_radius, max_radius)
            y = np.random.uniform(-max_radius, max_radius)
        else:
            x = np.random.normal(loc=0.0, scale=sigma)
            y = np.random.normal(loc=0.0, scale=sigma)

        if on_grid:
            x = np.round(x / AA.uv_cell_size) * AA.uv_cell_size
            y = np.round(y / AA.uv_cell_size) * AA.uv_cell_size

        new_antpos = np.array([[x, y]])
        r = np.linalg.norm(new_antpos, axis=1)
        if r > max_radius or r < min_radius:
            continue

        temp_antpos = np.vstack([AA.antpos, new_antpos])

        if not collision_check(temp_antpos, AA.diameter):
            AA.antpos = temp_antpos
            n_added += 1

            if verbose:
                clear_output(wait=True)
                print(f"Added antenna #{existing_n + n_added}/{n_total}: ({x:.2f}, {y:.2f})")

            if save_file:
                AA.save(path_to_file)

            if show_plot:
                AA.plot_fig, AA.plot_ax = plotting.plot_array(AA, AA.plot_fig, AA.plot_ax)

    print(f"Done. Added {n_added} antennas.")

