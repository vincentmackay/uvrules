#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 17:34:48 2025

@author: vincent
"""

import numpy as np
from copy import deepcopy
from datetime import datetime

#from IPython.display import clear_output
from . import AntArray
from . import geometry
from . import utils

#import time

def compactify_array(AA, verbose=False, n_max_baseline_counts=1, **add_ant_greedy_kwargs):
    """
    Attempts to improve an AntArray by replacing isolated antennas
    that are involved in only one commanded uv point.

    Parameters
    ----------
    AA : AntArray
        The original AntArray instance.
    **add_ant_greedy_kwargs : dict
        Any kwargs to pass to AA.add_ant_greedy()

    Returns
    -------
    AntArray
        A potentially improved AntArray instance.
    """
    AA = deepcopy(AA)  # avoid in-place modification
    round_num = 0

    while True:
        round_num += 1
        if verbose:
            print(f"\n🔁 Round {round_num}: Looking for isolated antennas...")

        original_size = geometry.get_array_size(AA.antpos)
        isolated_idxs = []

        for i in range(len(AA.antpos)):
            antpos_temp = np.delete(AA.antpos, i, axis=0)
            size_temp = geometry.get_array_size(antpos_temp)
            if size_temp < original_size:
                isolated_idxs.append(i)

        if not isolated_idxs:
            if verbose:
                print("✅ No isolated antennas found. Exiting.")
            break

        if verbose:
            print(f"Found {len(isolated_idxs)} isolated antennas.")

        # Try improving by removing each isolated antenna
        improved = False
        for i in isolated_idxs:
            if verbose:
                print(f"➡️  Trying antenna {i}...")

            antpos_temp = np.delete(AA.antpos, i, axis=0)

            # Check fulfillment without antenna i
            fulfilled_idx, not_fulfilled_idx = geometry.check_fulfillment(
                commanded=AA.commanded,
                antpos=antpos_temp,
                fulfill_tolerance=AA.fulfill_tolerance,
            )


            # If it causes exactly fewer uv points to be lost than n_max_baseline_counts
            if len(not_fulfilled_idx) <= n_max_baseline_counts or n_max_baseline_counts<0:
                if verbose:
                    print(f"🔍 Antenna {i} is responsible for {len(not_fulfilled_idx)} uv point.")

                # Create a copy of the AntArray with that antenna removed
                AA_temp = deepcopy(AA)
                AA_temp.antpos = antpos_temp
                max_array_size = geometry.get_array_size(antpos_temp)

                # Try to re-add one antenna, prioritizing smaller array
                AA_temp.add_ant_greedy(
                    compare_all_antpos=True,
                    compare_all_commanded=True,
                    minimize_array_size=True,
                    max_array_size=max_array_size,
                    n_to_add=1,
                    **add_ant_greedy_kwargs
                )
                
                fulfilled_idx, not_fulfilled_idx = geometry.check_fulfillment(AA_temp)
                
                new_size = geometry.get_array_size(AA_temp.antpos)

                if new_size < original_size and len(not_fulfilled_idx)==0:
                    if verbose:
                        print(f"✅ Success: array size reduced from {original_size:.3f} → {new_size:.3f}")
                    AA = AA_temp
                    improved = True
                    break  # Restart from scratch

                elif verbose:
                    if len(not_fulfilled_idx)>0:
                        print(f"❌ Couldn't re-fulfill all removed points, {len(not_fulfilled_idx)} remaining")
                    else:
                        print(f"❌ No improvement (array size {new_size:.3f} ≥ {original_size:.3f})")
            elif verbose:
                print(f"Antenna {i} was involved in {len(not_fulfilled_idx)} baselines, going to the next one.")

        if not improved:
            if verbose:
                print("🛑 No replacements improved the array.")
            break

    return AA



def use_fewer_antennas(
    AA,
    verbose=False,
    max_max_bl_per_ant=5,
    path_to_file="default.pkl",
    **add_ant_greedy_kwargs
):
    """
    Iteratively removes antennas involved in few baselines and re-adds them efficiently.

    Parameters
    ----------
    AA : AntArray
        The original AntArray instance.
    max_max_bl_per_ant : int, optional
        Maximum baseline threshold to iterate up to. Starts at 1.
    path_to_file : str, optional
        Path to save AA every time there's a successful improvement.
    **add_ant_greedy_kwargs : dict
        Any kwargs to pass to add_ant_greedy()

    Returns
    -------
    AntArray
        The optimized AntArray instance.
    """
    AA = deepcopy(AA)
    n_max_antennas = len(AA.antpos)
    progress_str = 'Starting...'
    # Check if baseline_counts is valid
    if not (hasattr(AA, "baseline_counts") and isinstance(AA.baseline_counts, list) and len(AA.baseline_counts) == len(AA.antpos)):
        if verbose:
            print("⚙️  Computing baseline counts...")
        AA.baseline_counts = []
        AA = utils.get_baseline_counts(AA)

    try:
        for max_bl_per_ant in range(1, max_max_bl_per_ant + 1):
            progress_str = f"Starting max_bl_per_ant = {max_bl_per_ant}..." + "\n" + progress_str
            round_num = 0
            while True:
                round_num += 1
                if verbose:
                    print(f"\n🔁 max_bl_per_ant = {max_bl_per_ant} — Round {round_num}")

                antpos = AA.antpos
                baseline_counts = np.array(AA.baseline_counts)
                to_remove_idxs = np.where(baseline_counts <= max_bl_per_ant)[0]

                if verbose:
                    print(f"📉 Found {len(to_remove_idxs)} antennas with ≤ {max_bl_per_ant} baselines.")

                if len(to_remove_idxs) == 0:
                    if verbose:
                        print("✅ No qualifying antennas left at this level.")
                    break

                # Remove and try to re-add
                antpos_reduced = np.delete(antpos, to_remove_idxs, axis=0)
                AA_reduced = deepcopy(AA)
                AA_reduced.antpos = antpos_reduced

                n_before = len(antpos)

                AA_new = deepcopy(AA_reduced)
                AA_new.add_ant_greedy(
                    compare_all_antpos=True,
                    compare_all_commanded=True,
                    minimize_array_size = True,
                    additional_output = progress_str,
                    n_max_antennas = n_max_antennas,
                    show_plot = False
                )
                n_after = len(AA_new.antpos)
                fulfilled_idx, not_fulfilled_idx = geometry.check_fulfillment(AA_new)
                if verbose:
                    print(f"📊 Antennas before: {n_before}, after: {n_after}")

                if n_after < n_before and len(not_fulfilled_idx) == 0:
                    progress_str = f"📌 On max_bl_per_ant = {max_bl_per_ant}, antennas reduced to {n_before} → {n_after}" + "\n" + progress_str
                    if verbose:
                        print("✅ Improvement! Fewer antennas.")
                    AA = AA_new
                    AA.baseline_counts = []
                    AA = utils.get_baseline_counts(AA)
                    AA.save(path_to_file)
                    continue

                elif n_after == n_before and len(not_fulfilled_idx) == 0:
                    size_before = geometry.get_array_size(AA.antpos)
                    size_after = geometry.get_array_size(AA_new.antpos)
                    if size_after < size_before:
                        if verbose:
                            print(f"🪶 Same #antennas but smaller array: {size_before:.3f} → {size_after:.3f}")
                        AA = AA_new
                        AA.baseline_counts = utils.get_baseline_counts(AA)
                        AA.save(path_to_file)
                    else:
                        if verbose:
                            print("➖ Same antennas, no size improvement.")
                    break

                else:
                    if verbose:
                        if len(not_fulfilled_idx) > 0:
                            print("❌ Not all antennas fulfilled. Reverting.")
                        else:
                            print("❌ More antennas than before. Reverting.")
                    break

    except KeyboardInterrupt:
        print("🛑 Interrupted by user. Returning current AntArray.")

    return AA


def random_amputation_loop(
    AA,
    n_ant_to_remove=5,
    path_to_file="default.pkl",
    n_max_loops=None,
    verbose=True,
):
    """
    Repeatedly applies random amputation + re-adding to reduce antenna count or array size.

    Parameters
    ----------
    AA : AntArray
        The AntArray instance to improve.
    n_ant_to_remove : int or callable
        Number of antennas to remove each round, or a function returning an int.
    path_to_file : str
        Where to save the array if improved.
    n_max_loops : int or None
        Maximum number of iterations to run. If None, runs until interrupted.
    verbose : bool
        Whether to print progress.
    **add_ant_greedy_kwargs : dict
        Arguments passed to add_ant_greedy()

    Returns
    -------
    AntArray
        The final AntArray after looping.
    """


    AA = deepcopy(AA)
    loop_counter = 0

    try:
        while True:
            loop_counter += 1
            if n_max_loops is not None and loop_counter > n_max_loops:
                if verbose:
                    print("🛑 Reached max loop count.")
                break

            antpos = AA.antpos
            n_before = len(antpos)

            # Determine how many antennas to remove
            n_remove = n_ant_to_remove() if callable(n_ant_to_remove) else n_ant_to_remove
            if n_remove >= n_before:
                if verbose:
                    print(f"⚠️ Skipping: n_ant_to_remove = {n_remove} ≥ number of antennas = {n_before}")
                continue

            remove_idxs = np.random.choice(n_before, size=n_remove, replace=False)
            if verbose:
                print(f"\n🔁 Loop {loop_counter}: Removing {n_remove} antennas...")

            antpos_reduced = np.delete(antpos, remove_idxs, axis=0)
            AA_reduced = deepcopy(AA)
            AA_reduced.antpos = antpos_reduced

            AA_new = deepcopy(AA_reduced)
            additional_output = (
                f"[{datetime.now().strftime('%H:%M:%S')}] Loop {loop_counter}: "
                f"Removed {n_remove} antennas from {n_before}"
            )

            AA_new.add_ant_greedy(
                compare_all_antpos=True,
                compare_all_commanded=True,
                minimize_array_size=True,
                additional_output=additional_output,
                show_plot=False,
            )

            n_after = len(AA_new.antpos)
            if verbose:
                print(f"📊 Antennas before: {n_before}, after re-adding: {n_after}")

            if n_after < n_before:
                if verbose:
                    print("✅ Improvement! Fewer antennas.")
                AA = AA_new
                AA.save(path_to_file)
                continue

            elif n_after == n_before:
                size_before = geometry.get_array_size(AA.antpos)
                size_after = geometry.get_array_size(AA_new.antpos)
                if size_before - size_after > 1e-3 :
                    if verbose:
                        print(f"🪶 Same #antennas but smaller array: {size_before:.3f} → {size_after:.3f}")
                    AA = AA_new
                    AA.save(path_to_file)
                else:
                    if verbose:
                        print("➖ Same antennas, no size improvement.")
                continue

            else:
                if verbose:
                    print("❌ More antennas than before. Reverting.")
                continue

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Returning current AntArray.")

    return AA




def nudge_antennas(antpos, nudge_amount=0.1, diameter=10.0, max_attempts=100):
    """
    Nudge antenna positions randomly within a maximum displacement, avoiding collisions.

    Parameters
    ----------
    antpos : ndarray of shape (N, 2)
        Original antenna positions.
    nudge_amount : float
        Maximum distance by which to nudge an antenna (uniform in [0, nudge_amount]).
    diameter : float
        Minimum allowed separation between any two antennas (collision threshold).
    max_attempts : int
        Maximum number of attempts to find a valid nudge for each antenna.

    Returns
    -------
    nudged_antpos : ndarray of shape (N, 2)
        Nudged antenna positions, collision-free.
    """
    N = len(antpos)
    nudged_antpos = antpos.copy()

    for i in range(N):
        original = nudged_antpos[i].copy()
        success = False
        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, nudge_amount)
            displacement = dist * np.array([np.cos(angle), np.sin(angle)])
            candidate = original + displacement

            # Try the nudged position
            temp_antpos = nudged_antpos.copy()
            temp_antpos[i] = candidate

            if not geometry.collision_check(temp_antpos, diameter):
                nudged_antpos[i] = candidate
                success = True
                break

        if not success:
            nudged_antpos[i] = original  # Revert to original if no valid move found

    return nudged_antpos
