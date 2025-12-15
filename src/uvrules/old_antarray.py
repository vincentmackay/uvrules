#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AntArray class for designing uv-complete antenna layouts.

Created on Tue May 21 16:45:26 2024
@author: Vincent
"""

import numpy as np
import os
import pickle
import warnings
import inspect
from astropy import constants

from . import utils
from . import geometry
from .algos import rules as _rules
from .algos import random

_RULES_PARAM_NAMES = [
    name for name in inspect.signature(rules.add_ant_rules).parameters
    if name != "AA"  # first positional is the AntArray instance
]
_RULES_PARAM_SET = set(_RULES_PARAM_NAMES)

ANTARRAY_PARAM_NAMES = {
    "antpos",
    "ref_wl",
    "ref_freq",
    "diameter",
    "diameter_lambda",
    "max_bl",
    "max_bl_lambda",
    "min_bl",
    "min_bl_lambda",
    "packing_density",   # aka "rho"
    "uv_cell_size",
    "fulfill_tolerance",
    "max_array_size",
    "commanded",
}




class AntArray:
    """
    Class for managing antenna array layouts for uv-complete sampling.

    Attributes
    ----------
    verbose : bool
        Whether to print detailed messages.
    antpos : np.ndarray
        Antenna positions, shape (n_antennas, 2).
    ref_wl : float
        Reference wavelength, in meters.
    ref_freq : float
        Reference frequency, in Hz.
    diameter : float
        Antenna dish diameter, in meters.
    diameter_lambda : float
        Antenna diameter in wavelengths.
    max_bl : float
        Maximum baseline length in meters.
    max_bl_lambda : float
        Maximum baseline length in wavelengths.
    min_bl : float
        Minimum baseline length in meters.
    min_bl_lambda : float
        Minimum baseline length in wavelengths.
    packing_density : float
        Packing density defining uv cell size.
    uv_cell_size : float
        Size of a uv cell in meters.
    fulfill_tolerance : float
        Tolerance for uv point fulfillment in meters.
    max_array_size : float or None
        Maximum physical array size (optional).
    commanded : np.ndarray
        Commanded uv points to be fulfilled.
    array_config : dict
        Dictionary mapping antenna indices to coordinates.
    fulfilled_idx : np.ndarray
        Indices of fulfilled commanded uv points.
    not_fulfilled_idx : np.ndarray
        Indices of not-fulfilled commanded uv points.
    redundancy : np.ndarray
        Redundancy histogram.
    baseline_counts : list
        Baseline loss counts when removing each antenna.
    efficiency : float
        Overall array efficiency.
    antpairs : list of tuple
        Selected baseline pairs.
    """

    def __init__(self,
                 diameter: float = None,
                 diameter_lambda: float = None,
                 max_array_size: float = None,
                 ref_freq: float = None,
                 ref_wl: float = None,
                 packing_density: float = 2,
                 min_bl_lambda: float = None,
                 max_bl_lambda: float = None,
                 min_bl: float = None,
                 max_bl: float = None,
                 fulfill_tolerance: float = None,
                 p_norm: float = np.inf,
                 verbose: bool = False,
                 **rules_kwargs):
        """
        Initialize an AntArray instance.

        **rules_kwargs :
            Additional keyword arguments that will be passed by default to
            :func:`uvrules.algos.rules.add_ant_rules`. These correspond to the
            parameters of :func:`add_ant_rules` (e.g. ``commanded_order``,
            ``n_to_add``, ``compare_all_commanded``, etc.). Any values supplied
            here can be overridden by keyword arguments passed directly to
            :meth:`AntArray.add_ant_rules`. 
        """
        self.verbose = verbose
        self.antpos = np.array([[0.0, 0.0]])

        self.ref_wl, self.ref_freq = self._initialize_ref_wl(ref_wl, ref_freq)
        self.max_bl, self.max_bl_lambda = self._resolve_units(max_bl, max_bl_lambda, self.ref_wl, name='max_bl', default_lambda=100)
        self.min_bl, self.min_bl_lambda = self._resolve_units(min_bl, min_bl_lambda, self.ref_wl, name='min_bl', default_lambda=10)
        self.diameter, self.diameter_lambda = self._resolve_units(diameter, diameter_lambda, self.ref_wl, name='diameter', default_lambda=5)

        if self.min_bl < self.diameter:
            self._verbose_print('Minimum baseline shorter than diameter; setting min_bl = diameter.')
            self.min_bl = self.diameter

        if self.max_bl < self.min_bl:
            self._verbose_print('Maximum baseline shorter than minimum baseline; setting max_bl = 10 × min_bl.')
            self.max_bl = 10 * self.min_bl
            self.max_bl_lambda = self.max_bl / self.ref_wl

        self.packing_density = packing_density
        self.p_norm = p_norm
        self.uv_cell_size = self.ref_wl / self.packing_density
        self.fulfill_tolerance = fulfill_tolerance if fulfill_tolerance is not None else 1e-5# self.uv_cell_size / 2
        self.max_array_size = max_array_size

        self.commanded = utils.generate_commanded_square(uv_cell_size=self.uv_cell_size, min_bl=self.min_bl, max_bl=self.max_bl, show_plot=False)
        self.array_config = utils.get_array_config(self.antpos)

        # Initialized as None
        self.fulfilled_idx = None
        self.not_fulfilled_idx = None
        self.redundancy = None
        self.baseline_counts = None
        self.efficiency = None
        self.antpairs = None

        self._rules_defaults: dict = {}
        self._set_rules_defaults(**rules_kwargs)

    def _verbose_print(self, message: str):
        """Print a message if verbose mode is active."""
        if self.verbose:
            print(message)

    def _initialize_ref_wl(self, ref_wl: float = None, ref_freq: float = None):
        """Initialize reference wavelength and frequency."""
        if ref_wl is not None and ref_freq is not None:
            self._verbose_print("Both ref_freq and ref_wl provided; using ref_freq.")

        if ref_freq is None and ref_wl is None:
            self._verbose_print("Using default frequency of 150 MHz.")
            ref_freq = 150e6
            ref_wl = constants.c.value / ref_freq
        elif ref_freq is None:
            ref_freq = constants.c.value / ref_wl
        else:
            ref_wl = constants.c.value / ref_freq

        return ref_wl, ref_freq

    def _resolve_units(self, value: float, value_lambda: float, ref_wl: float, name: str, default_lambda: float = 2):
        """Resolve values in meters and/or wavelengths."""
        if value is None and value_lambda is None:
            self._verbose_print(f"Using default {name}_lambda = {default_lambda}.")
            value_lambda = default_lambda
            value = value_lambda * ref_wl
        elif value is None:
            value = value_lambda * ref_wl
        elif value_lambda is None:
            value_lambda = value / ref_wl
        return value, value_lambda

    # ---------- Utility functions as instance methods -----------

    def get_array_size(self) -> float:
        """Compute and store the maximum array size."""
        self.array_size = geometry.get_array_size(self.antpos)
        return self.array_size

    def get_array_config(self) -> dict:
        """Compute and store the array configuration dictionary."""
        self.array_config = utils.get_array_config(self.antpos)
        return self.array_config

    def select_baselines(self) -> list:
        """Select and store baseline pairs fulfilling commanded uv points."""
        self.antpairs = utils.select_baselines(self.commanded, self.antpos, self.fulfill_tolerance)
        return self.antpairs

    def get_redundancy(self, red_tol_lambda: float = None) -> np.ndarray:
        """Compute and store the baseline redundancy."""
        self.redundancy = utils.get_redundancy(antpos=self.antpos, ref_wl=self.ref_wl, red_tol_lambda=red_tol_lambda)
        return self.redundancy

    def generate_commanded_square(self, uv_cell_size=None, min_bl=None, max_bl=None, show_plot=True, shape = 'square') -> np.ndarray:
        """Generate and store a new set of commanded uv points."""
        if uv_cell_size is None:
            uv_cell_size = self.uv_cell_size
        if min_bl is None:
            min_bl = self.min_bl
        if max_bl is None:
            max_bl = self.max_bl
        self.commanded = utils.generate_commanded_square(uv_cell_size, min_bl, max_bl, show_plot=show_plot)
        return self.commanded

    def check_fulfillment(self, flip_tolerance=0.0, verbose=False) -> tuple:
        """Check and store which commanded points are fulfilled."""
        self.fulfilled_idx, self.not_fulfilled_idx, self.n_remaining_fulfillments = geometry.check_fulfillment(self, flip_tolerance=flip_tolerance, verbose=verbose)
        return self.fulfilled_idx, self.not_fulfilled_idx, self.n_remaining_fulfillments

    def get_baseline_counts(self, verbose=True) -> list:
        """Compute and store baseline loss counts."""
        self.baseline_counts = utils.get_baseline_counts(self, verbose=verbose)
        return self.baseline_counts

    def export_antpos_csv(self, path: str, include_index: bool = False):
        """Export antenna positions to a CSV file."""
        utils.export_antpos_csv(self.antpos, path, include_index=include_index)

    def get_efficiency(self) -> float:
        """Retrieve and store current efficiency."""
        if hasattr(self, 'history') and 'efficiency' in self.history:
            self.efficiency = self.history['efficiency'][-1]
            return self.efficiency
        else:
            raise AttributeError("Efficiency history not found. Run antenna placement first.")

    def get_efficiency_array(self) -> list:
        """Retrieve the full efficiency history."""
        if hasattr(self, 'history') and 'efficiency' in self.history:
            return self.history['efficiency']
        else:
            raise AttributeError("Efficiency history not found. Run antenna placement first.")

    def _set_rules_defaults(self, **rules_kwargs):
        """
        Update the default keyword arguments that will be passed to add_ant_rules.

        This is used internally by __init__, but can also be used by advanced
        users to tweak defaults between runs.
        """
        if not hasattr(self, "_rules_defaults"):
            self._rules_defaults = {}
        self._rules_defaults.update(rules_kwargs)


        for key, value in rules_kwargs.items():
            if key not in _RULES_PARAM_SET:
                raise TypeError(
                    f"AntArray got unexpected RULES parameter '{key}'. "
                    f"Valid parameters are: {_RULES_PARAM_NAMES}"
                )
            self._rules_defaults[key] = value


    def _merge_rules_kwargs(self, call_kwargs):
        """
        Merge RULES defaults stored on this AntArray with call-time kwargs.
        Returns a merged dictionary.

        call-time args override defaults.
        """
        # Get defaults stored during __init__
        defaults = getattr(self, "_rules_defaults", {}).copy()

        # Override with call-time arguments
        for key, value in call_kwargs.items():
            defaults[key] = value

        return defaults



    # ---------- Missing attributes management -----------

    def check_missing_attributes(self) -> dict:
        """Check which important attributes are missing."""
        missing = {}
        missing['redundancy'] = self.redundancy is None
        missing['baseline_counts'] = not hasattr(self, 'baseline_counts') or self.baseline_counts is None
        missing['commanded'] = self.commanded is None
        missing['array_size'] = not hasattr(self, 'array_size') or self.array_size is None
        missing['fulfilled_idx'] = not hasattr(self, 'fulfilled_idx') or self.fulfilled_idx is None
        missing['not_fulfilled_idx'] = not hasattr(self, 'not_fulfilled_idx') or self.not_fulfilled_idx is None
        missing['array_config'] = not hasattr(self, 'array_config') or self.array_config is None
        missing['antpairs'] = not hasattr(self, 'antpairs') or self.antpairs is None
        missing['efficiency_array'] = not (hasattr(self, 'history') and 'efficiency' in self.history)
        missing['efficiency'] = not hasattr(self, 'efficiency') or self.efficiency is None
        return missing

    def compute_missing_attributes(self,
                                    redundancy=True,
                                    baseline_counts=True,
                                    commanded=True,
                                    array_size=True,
                                    fulfillment=True,
                                    array_config=True,
                                    antpairs=True,
                                    efficiency_array=True,
                                    efficiency=True,
                                    force_recompute=False,
                                    verbose=True):
        """Compute missing attributes, optionally forcing recomputation."""
        missing = self.check_missing_attributes()

        if redundancy and (missing['redundancy'] or force_recompute):
            if verbose:
                print("Computing redundancy...")
            self.get_redundancy()

        if baseline_counts and (missing['baseline_counts'] or force_recompute):
            if verbose:
                print("Computing baseline counts...")
            self.get_baseline_counts()

        if commanded and (missing['commanded'] or force_recompute):
            if verbose:
                print("Generating commanded points...")
            self.generate_commanded_square()

        if array_size and (missing['array_size'] or force_recompute):
            if verbose:
                print("Computing array size...")
            self.get_array_size()

        if fulfillment and (missing['fulfilled_idx'] or missing['not_fulfilled_idx'] or force_recompute):
            if verbose:
                print("Checking fulfillment...")
            self.check_fulfillment()

        if array_config and (missing['array_config'] or force_recompute):
            if verbose:
                print("Computing array configuration...")
            self.get_array_config()

        if antpairs and (missing['antpairs'] or force_recompute):
            if verbose:
                print("Selecting baselines...")
            self.select_baselines()

        if efficiency_array and (missing['efficiency_array'] or force_recompute):
            if verbose:
                print("Warning: Efficiency array is built during antenna placement.")

        if efficiency and (missing['efficiency'] or force_recompute):
            if hasattr(self, 'history') and 'efficiency' in self.history:
                if verbose:
                    print("Computing current efficiency...")
                self.get_efficiency()

    # ---------- Algorithm launching wrappers -----------


    
    def add_ant_rules(self, **kwargs):
        """
        Run the RULES algorithm using defaults stored on this AntArray,
        unless overridden by explicit call-time arguments.
        """
        # Merge defaults + overrides
        merged = self._merge_rules_kwargs(kwargs)

        # Optional: warn if overriding defaults
        for k, v in kwargs.items():
            if k in self._rules_defaults:
                warnings.warn(
                    f"Overriding default RULES parameter '{k}' defined on "
                    f"AntArray (default={self._rules_defaults[k]!r}) with {v!r}",
                    UserWarning,
                    stacklevel=2,
                )

        # Call the internal algorithm
        return _rules._add_ant_rules_impl(self, **merged)



    def add_ant_random(self, **kwargs):
        """Add antennas using random algorithm."""
        return random.add_ant_random(self, **kwargs)

    # ---------- Save/load -----------

    def save(self, path_to_file: str):
        """Save AntArray instance."""
        if not isinstance(path_to_file, str):
            raise TypeError("path_to_file must be a string.")
        with open(path_to_file, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path_to_file: str, inplace: bool = True):
        """Load AntArray instance from file."""
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"File not found: {path_to_file}")
        with open(path_to_file, 'rb') as f:
            loaded = pickle.load(f)
        if inplace:
            self.__dict__.update(loaded.__dict__)
        else:
            return loaded
