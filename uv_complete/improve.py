#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 17:34:48 2025

@author: vincent
"""

import numpy as np
from copy import deepcopy
from uv_complete.antarray import AntArray
import uv_complete.geometry as geometry
import time


def improve_array(AA: AntArray, use_grid = True, verbose = False, **add_ant_algo_kwargs):
    """
    Attempts to improve an AntArray by replacing isolated antennas
    that are involved in only one commanded uv point.

    Parameters
    ----------
    AA : AntArray
        The original AntArray instance.
    **add_ant_algo_kwargs : dict
        Any kwargs to pass to AA.add_ant_algo() (e.g., compare_all_commanded, commanded_order, etc.)

    Returns
    -------
    AntArray
        A potentially improved AntArray instance.
    """

    # Copy original array to avoid modifying in place
    AA = deepcopy(AA)

    if not hasattr(AA, "commanded_set"):
        AA.commanded_set = AA._make_commanded_grid_set()

    while True:
        # Compute baseline count for each antenna
        baseline_counts = []
        antpos = AA.antpos
        commanded = AA.commanded
        uv_cell_size = AA.uv_cell_size
        fulfill_tolerance = AA.fulfill_tolerance
        commanded_grid = AA.commanded_grid
        
        # Precompute full commanded set for fast indexing if needed
        last_step_time = time.time()
        for i in range(len(antpos)):
            
            print(f'⏳ Evaluating antenna {i+1}/{len(antpos)}')
            print(time.time() - last_step_time)
            last_step_time = time.time()
            # Temporarily remove antenna i
            antpos_temp = np.delete(antpos, i, axis=0)
        
            # Round the new antpos array to grid
            antpos_grid_temp = np.round(antpos_temp / uv_cell_size).astype(int)
        
            # Run grid-based fulfillment check
            fulfilled_idx, _ = geometry.check_fulfillment(
                commanded=commanded,
                antpos=antpos_temp,
                fulfill_tolerance=fulfill_tolerance,
                uv_cell_size=uv_cell_size,
                commanded_grid=commanded_grid,
                antpos_grid=antpos_grid_temp,
                use_grid=use_grid
            )
        
            # Count how many baselines are lost by removing this antenna
            baseline_counts.append(len(fulfilled_idx))
            
        # Convert to numpy array for further processing
        baseline_counts = np.array(baseline_counts)

        print(baseline_counts)
        # Find antenna involved in only 1 uv point
        candidate_idxs = np.where(baseline_counts == 1)[0]
        if len(candidate_idxs) == 0:
            if verbose:
                print('No more antennas to move.')
            break  # No more lone antennas to remove

        # Among those, pick the most isolated one
        dists = np.array([
            np.min(np.linalg.norm(AA.antpos[i] - np.delete(AA.antpos, i, axis=0), axis=1))
            for i in candidate_idxs
        ])
        worst_idx = candidate_idxs[np.argmax(dists)]

        # Remove it and improve
        print(f"🔁 Replacing antenna {worst_idx} (only 1 uv, isolated)")

        antpos_new = np.delete(AA.antpos, worst_idx, axis=0)
        AA_new = deepcopy(AA)
        AA_new.antpos = antpos_new
        AA_new.commanded = AA.commanded
        AA_new.commanded_grid = AA.commanded_grid
        AA_new.commanded_set = AA.commanded_set
        AA_new.antpos = antpos_new

        AA_new.add_ant_algo(**add_ant_algo_kwargs)

        AA = AA_new  # Update AA and repeat

    return AA
