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




def check_fulfillment(AA = None, commanded = None, antpos = None, fulfill_tolerance = None,
                      p_norm=np.inf, flip_tolerance=0.0, verbose = False):
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


def get_new_fulfilled(new_antpos=None, antpos=None, not_fulfilled_tree=None, not_fulfilled_array=None,
                      fulfill_tolerance=0., uv_cell_size=1., p_norm=np.inf):
    """
    Returns the set of newly fulfilled uv points for a candidate antenna placement.

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
    new_uvs *= np.where(new_uvs[:, 1] >= 0, 1, -1).reshape(-1, 1)
    new_uvs[(new_uvs[:, 1] == 0), 0] = np.abs(new_uvs[(new_uvs[:, 1] == 0), 0])

    # KDTree lookup
    matches = not_fulfilled_tree.query_ball_point(new_uvs, r=fulfill_tolerance, p=p_norm)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=float)

    matched_indices = np.unique(np.concatenate(matches)).astype(int)
    return not_fulfilled_array[matched_indices]


    



