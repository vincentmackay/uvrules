#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:27:51 2024

@author: vincent
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
from collections import Counter
random_seed = 11141


def antpos_to_uv(antpos, flip_tolerance=1e-5, unique_only=False, fill_all_plane=False):
    """Converts antenna positions to sampled uv points, applying flipping logic for u-v symmetry.
    
    """

    # Get all unique baseline pairs
    n_ants = antpos.shape[0]
    idx1, idx2 = np.triu_indices(n_ants, k=1)  # Upper triangle index pairs
    u = antpos[idx2, 0] - antpos[idx1, 0]  # u = x2 - x1
    v = antpos[idx2, 1] - antpos[idx1, 1]  # v = y2 - y1

    # Apply flipping logic
    flip_mask = (v < -flip_tolerance) | ((np.abs(v) <= flip_tolerance) & (u < 0))
    u[flip_mask] *= -1
    v[flip_mask] *= -1

    # Stack into an array
    uv_points = np.column_stack((u, v))

    # If full-plane mirroring is enabled
    if fill_all_plane:
        uv_points = np.vstack([uv_points, -uv_points])

    # If unique-only mode is enabled
    if unique_only:
        uv_points = np.round(uv_points, decimals=4)
        uv_points = np.unique(uv_points, axis=0)
    
    return uv_points


from collections import defaultdict

def get_redundant_baseline_groups(antpos, r_tol=1e-3, include_autos=False):
    """
    Group baselines from 2D antenna positions into redundancy classes.

    Parameters
    ----------
    antpos : ndarray of shape (N, 2)
        Antenna positions in 2D (units: meters or wavelengths).
    r_tol : float
        Redundancy grouping tolerance. Two baselines are considered redundant
        if their (u,v) vectors are within this distance, or are negatives of each other.
    include_autos : bool
        If True, include auto-correlations (i,i) as baseline entries.

    Returns
    -------
    redundant_groups : list of list of tuple
        A list where each element is a list of antenna index pairs (i,j)
        corresponding to one redundancy group.
    """
    n_antennas = antpos.shape[0]
    bl_vectors = []
    bl_pairs = []

    for i in range(n_antennas):
        for j in range(i if include_autos else i+1, n_antennas):
            dx = antpos[j, 0] - antpos[i, 0]
            dy = antpos[j, 1] - antpos[i, 1]

            # Flip convention to ensure (u,v) ≡ (-u,-v)
            if dy < -1e-5 or (abs(dy) < 1e-5 and dx < 0):
                dx *= -1
                dy *= -1

            bl_vectors.append((dx, dy))
            bl_pairs.append((i, j))

    bl_vectors = np.array(bl_vectors)
    bl_pairs = np.array(bl_pairs)

    # Round baseline vectors to a grid of size r_tol for hashing
    keys = np.round(bl_vectors / r_tol).astype(int)

    # Use hashed keys to group baseline pairs
    hash_map = defaultdict(list)
    for key, bl in zip(map(tuple, keys), bl_pairs):
        hash_map[key].append(tuple(bl))

    return list(hash_map.values())


def get_array_size(antpos):
    if len(antpos.shape)==1:
        return 0
    # Returns the physical size of the array
    return np.max(pdist(antpos, 'euclidean'))


def nonzero_sign(x):
    return np.where (x>=0,1,-1)



def get_min_distance(antpos):
    # Returns the minimum distance between any two points
    return np.min(pdist(antpos, 'euclidean'))

def get_min_distance_from_new_antpos(antpos, new_antpos):
    # Returns the minimum distance between an array that is currently antpos and a new point
    return np.min(np.linalg.norm(antpos-new_antpos,axis=1))




def collision_check(antpos,diameter):
    # Returns true if there is a collision
    if diameter is None:
        return False
    else:
        return get_min_distance(antpos)<=diameter


def collision_check_single_antenna(antpos: np.ndarray, antenna_idx: int, diameter: float) -> bool:
    """Check if a single antenna collides with any other antenna."""
    if diameter is None:
        return False
    
    # Calculate distances from this antenna to all others
    this_antenna = antpos[antenna_idx]
    other_antennas = np.delete(antpos, antenna_idx, axis=0)
    
    if len(other_antennas) == 0:
        return False
    
    distances = np.linalg.norm(other_antennas - this_antenna, axis=1)
    return np.min(distances) <= diameter

def get_redundancy(AA = None, antpos = None, ref_wl = None, red_tol_lambda = None, round = False):
    if AA is not None:
        antpos = AA.antpos
        ref_wl = AA.ref_wl
    if red_tol_lambda is None:
        print('Using the default redundancy tolerance of 0.1 lambda.')
        red_tol = 0.1 * ref_wl
    else:
        red_tol = red_tol_lambda * ref_wl
    
    uvs = antpos_to_uv(antpos, unique_only = False)
    if round:
        uvs = np.round(uvs, decimals=10)
    
    # Determine the bounds of the rectangle that contains all points
    min_coords = np.floor(uvs.min(axis=0))
    
    # Map points to lattice squares using integer division
    square_indices = np.floor((uvs - min_coords) / red_tol).astype(int)

    # Count the occurrences of each square index
    counts = Counter(map(tuple, square_indices))
    
    # Return the non-zero counts as an array of integers
    return np.array(list(counts.values()))






def compute_new_antpos(i, j, k, antpos, commanded):
    """Compute the new antenna position based on indices and flipping."""
    return antpos[i] + (-1) ** k * commanded[j]




def check_fulfillment_very_old(AA = None, commanded = None, antpos = None, fulfill_tolerance = None,
                      p_norm=float('inf'), flip_tolerance=0.0, verbose = False):
    """Returns the indices of fulfilled and unfulfilled points."""
    
    if AA is not None:
        commanded = AA.commanded
        antpos = AA.antpos
        fulfill_tolerance = AA.fulfill_tolerance

    if len(antpos.shape) < 2 or len(antpos) < 2:
        return np.array([], dtype=int), np.arange(len(commanded), dtype=int)

    antpos_uvs = antpos_to_uv(antpos, flip_tolerance=flip_tolerance)
    tree = KDTree(antpos_uvs)
    idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p=p_norm)
    fulfilled_mask = np.asarray([bool(idx) for idx in idx_fulfilled])

    fulfilled_indices = np.where(fulfilled_mask)[0]
    not_fulfilled_indices = np.where(~fulfilled_mask)[0]

    if verbose:
        print(f'{len(fulfilled_indices)}/{len(commanded)} fulfilled, {len(not_fulfilled_indices)}/{len(commanded)} remaining.')

    return fulfilled_indices, not_fulfilled_indices


def check_fulfillment_old(AA=None, commanded=None, antpos=None, fulfill_tolerance=None,
                      p_norm=float('inf'), flip_tolerance=0.0, verbose=False, nsamples=1):
    """Returns the indices of fulfilled and unfulfilled points.
    
    A commanded point is considered fulfilled if at least `nsamples` realized
    uv points fall within `fulfill_tolerance` of it.
    """

    if AA is not None:
        commanded = AA.commanded
        antpos = AA.antpos
        fulfill_tolerance = AA.fulfill_tolerance

    if len(antpos.shape) < 2 or len(antpos) < 2:
        return np.array([], dtype=int), np.arange(len(commanded), dtype=int)

    # Get realized uv points from antenna positions
    antpos_uvs = antpos_to_uv(antpos, flip_tolerance=flip_tolerance)
    tree = KDTree(antpos_uvs)

    # Find all nearby realized uvs within tolerance for each commanded point
    idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p=p_norm)

    # Count how many realized points are within the ball for each commanded point
    fulfilled_mask = np.asarray([len(idx) >= nsamples for idx in idx_fulfilled])

    fulfilled_indices = np.where(fulfilled_mask)[0]
    not_fulfilled_indices = np.where(~fulfilled_mask)[0]

    if verbose:
        print(f'{len(fulfilled_indices)}/{len(commanded)} fulfilled (≥{nsamples} samples), '
              f'{len(not_fulfilled_indices)}/{len(commanded)} remaining.')

    return fulfilled_indices, not_fulfilled_indices


def check_fulfillment(AA=None, commanded=None, antpos=None, fulfill_tolerance=None,
                     p_norm=float('inf'), flip_tolerance=0.0, verbose=False, nsamples=1):
    """
    Returns the indices of fulfilled and unfulfilled points.
    A commanded point is considered fulfilled if at least `nsamples` realized
    uv points fall within `fulfill_tolerance` of it.
    
    Returns:
    --------
    fulfilled_indices : ndarray
    not_fulfilled_indices : ndarray  
    n_remaining_fulfillments : ndarray or None
        If nsamples > 1, returns array of remaining fulfillments needed per commanded point.
        If nsamples == 1, returns None.
    """
    if AA is not None:
        commanded = AA.commanded
        antpos = AA.antpos
        fulfill_tolerance = AA.fulfill_tolerance
        
    if len(antpos.shape) < 2 or len(antpos) < 2:
        empty_fulfilled = np.array([], dtype=int)
        all_not_fulfilled = np.arange(len(commanded), dtype=int)
        if nsamples > 1:
            n_remaining = nsamples * np.ones(len(commanded), dtype=int)
            return empty_fulfilled, all_not_fulfilled, n_remaining
        else:
            return empty_fulfilled, all_not_fulfilled, None
    
    # Get realized uv points from antenna positions
    antpos_uvs = antpos_to_uv(antpos, flip_tolerance=flip_tolerance)
    tree = KDTree(antpos_uvs)
    
    # Find all nearby realized uvs within tolerance for each commanded point
    idx_fulfilled = tree.query_ball_point(commanded, r=fulfill_tolerance, p=p_norm)
    
    # Count how many realized points are within the ball for each commanded point
    hit_counts = np.asarray([len(idx) for idx in idx_fulfilled])
    
    # Determine fulfilled vs not fulfilled based on nsamples threshold
    fulfilled_mask = hit_counts >= nsamples
    fulfilled_indices = np.where(fulfilled_mask)[0]
    not_fulfilled_indices = np.where(~fulfilled_mask)[0]
    
    if verbose:
        print(f'{len(fulfilled_indices)}/{len(commanded)} fulfilled (≥{nsamples} samples), '
              f'{len(not_fulfilled_indices)}/{len(commanded)} remaining.')
    
    # Calculate remaining fulfillments if nsamples > 1
    if nsamples > 1:
        n_remaining_fulfillments = np.maximum(0, nsamples - hit_counts)
        return fulfilled_indices, not_fulfilled_indices, n_remaining_fulfillments
    else:
        return fulfilled_indices, not_fulfilled_indices, len(not_fulfilled_indices)

def get_new_fulfilled_old(new_antpos=None, antpos=None, not_fulfilled_tree=None, not_fulfilled_array=None,
                      fulfill_tolerance=0., uv_cell_size=1., nsamples = 1, flip_tolerance = 1e-5, p_norm=float('inf')):
    """
    Returns the set of newly fulfilled uv points for a candidate antenna placement.

    As of the current implementation, even if the array asks for nsamples > 1, it will return the uv points that
    are fulfilled at least once, i.e. it does not check if the number of fulfilled points is at least nsamples.
    Parameters:
    -----------
    - new_antpos : ndarray of shape (2,), position of candidate antenna
    - antpos : ndarray of shape (N, 2), existing antenna positions
    - not_fulfilled_tree : KDTree over not_fulfilled_array
    - not_fulfilled_array : ndarray of shape (M, 2), uv points still needing fulfillment

    Returns:
    --------
    ndarray of matched uv coordinates (int) that are newly fulfilled.
    """

    if new_antpos is None or antpos is None or not_fulfilled_tree is None or not_fulfilled_array is None:
        raise ValueError("new_antpos, antpos, not_fulfilled_tree and not_fulfilled_array must be provided.")

    # Compute new uv vectors
    new_uvs = antpos - new_antpos
    new_uvs *= np.where(new_uvs[:, 1] >= -flip_tolerance, 1, -1).reshape(-1, 1) # This multiplies all elements where v >= 0 by 1, and where v < 0 by -1
    new_uvs[(np.abs(new_uvs[:, 1]) <= flip_tolerance), 0] = np.abs(new_uvs[(np.abs(new_uvs[:, 1]) <= flip_tolerance), 0])



    # KDTree lookup
    matches = not_fulfilled_tree.query_ball_point(new_uvs, r=fulfill_tolerance, p=p_norm)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=float)

    matched_indices = np.unique(np.concatenate(matches)).astype(int)
    return not_fulfilled_array[matched_indices]


def get_new_fulfilled(new_antpos=None, antpos=None, not_fulfilled_tree=None, not_fulfilled_array=None,
                     fulfill_tolerance=0., uv_cell_size=1., nsamples=1, flip_tolerance=1e-5,
                     not_fulfilled_idx=None, n_remaining_fulfillments=None, p_norm=float('inf')):
    """
    Returns the set of newly fulfilled uv points for a candidate antenna placement.
    Now supports nsamples > 1 by counting incremental progress toward fulfillment.
    
    Parameters:
    -----------
    - new_antpos : ndarray of shape (2,), position of candidate antenna
    - antpos : ndarray of shape (N, 2), existing antenna positions
    - not_fulfilled_tree : KDTree over not_fulfilled_array
    - not_fulfilled_array : ndarray of shape (M, 2), uv points still needing fulfillment
    - not_fulfilled_idx : ndarray, indices into commanded array for each point in not_fulfilled_array
    - n_remaining_fulfillments : ndarray of shape (len(commanded),), remaining samples needed per commanded point
    
    Returns:
    --------
    - newly_fulfilled_points : ndarray of matched uv coordinates that are newly fulfilled
    - fulfillment_score : int, total "fulfillment credit" accounting for nsamples
    """
    if new_antpos is None or antpos is None or not_fulfilled_tree is None or not_fulfilled_array is None:
        raise ValueError("new_antpos, antpos, not_fulfilled_tree and not_fulfilled_array must be provided.")
    
    # Compute new uv vectors
    new_uvs = antpos - new_antpos
    new_uvs *= np.where(new_uvs[:, 1] >= -flip_tolerance, 1, -1).reshape(-1, 1)
    new_uvs[(np.abs(new_uvs[:, 1]) <= flip_tolerance), 0] = np.abs(new_uvs[(np.abs(new_uvs[:, 1]) <= flip_tolerance), 0])
    
    # KDTree lookup
    matches = not_fulfilled_tree.query_ball_point(new_uvs, r=fulfill_tolerance, p=p_norm)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=float), 0
    
    matched_indices = np.unique(np.concatenate(matches)).astype(int)
    newly_fulfilled_points = not_fulfilled_array[matched_indices]
    
    # If nsamples = 1 or no remaining fulfillments tracking, return old behavior
    if nsamples == 1 or n_remaining_fulfillments is None or not_fulfilled_idx is None:
        return newly_fulfilled_points, len(newly_fulfilled_points)
    
    # For nsamples > 1, calculate fulfillment score
    fulfillment_score = 0
    
    # Count hits per commanded point
    commanded_hits = {}  # commanded_idx -> number of new UV realizations hitting it
    
    for match_list_idx, match_list in enumerate(matches):
        for not_fulfilled_local_idx in match_list:
            # Map from not_fulfilled_array index to commanded array index
            commanded_idx = not_fulfilled_idx[not_fulfilled_local_idx]
            commanded_hits[commanded_idx] = commanded_hits.get(commanded_idx, 0) + 1
    
    # Calculate score based on remaining fulfillments needed
    for commanded_idx, hits in commanded_hits.items():
        remaining = n_remaining_fulfillments[commanded_idx]
        contribution = min(hits, remaining)
        fulfillment_score += contribution
    
    return newly_fulfilled_points, fulfillment_score