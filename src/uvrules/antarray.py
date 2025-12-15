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

# Names of keyword parameters accepted by the RULES implementation (excluding the AntArray itself).
_RULES_PARAM_NAMES = [
    name for name in inspect.signature(_rules._add_ant_rules_implementation).parameters
    if name != 'AA'
]
_RULES_PARAM_SET = set(_RULES_PARAM_NAMES)

# Names of parameters that configure the AntArray itself and can be overridden
# via AntArray.add_ant_rules(...).
ANTARRAY_PARAM_NAMES = {
    'antpos',
    'ref_wl',
    'ref_freq',
    'diameter',
    'diameter_lambda',
    'max_bl',
    'max_bl_lambda',
    'min_bl',
    'min_bl_lambda',
    'packing_density',
    'uv_cell_size',
    'fulfill_tolerance',
    'max_array_size',
    'commanded',
    'p_norm',
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
        Antenna dish diameter, in wavelengths.
    max_bl : float
        Maximum baseline length to consider, in meters.
    max_bl_lambda : float
        Maximum baseline length to consider, in wavelengths.
    min_bl : float
        Minimum baseline length to consider, in meters.
    min_bl_lambda : float
        Minimum baseline length to consider, in wavelengths.
    packing_density : float
        Packing density (rho) controlling uv cell size.
    uv_cell_size : float
        Cell size in the uv-plane, in wavelengths.
    fulfill_tolerance : float
        Tolerance for considering a uv point fulfilled.
    max_array_size : float or None
        Maximum physical size of the array, in meters.
    p_norm : float
        Norm used for distance calculations (e.g. np.inf for Chebyshev norm).
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

    def __init__(
        self,
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
        uv_cell_size: float = None,
        antpos=None,
        commanded=None,
        **rules_kwargs,
    ):
        """
        Initialize an AntArray instance.

        Parameters
        ----------
        diameter, diameter_lambda, max_array_size, ref_freq, ref_wl, packing_density,
        min_bl_lambda, max_bl_lambda, min_bl, max_bl, fulfill_tolerance, p_norm,
        uv_cell_size, antpos, commanded :
            See class documentation for details.

        **rules_kwargs :
            Additional keyword arguments that will be passed by default to
            :func:`uvrules.algos.rules.add_ant_rules`. These correspond to the
            parameters of :func:`add_ant_rules` (e.g. ``commanded_order``,
            ``n_to_add``, ``compare_all_commanded``, etc.). Any values supplied
            here can be overridden by keyword arguments passed directly to
            :meth:`AntArray.add_ant_rules`.
        """
        self.verbose = verbose

        # (Re)configure the core AntArray state.
        self._configure_from_params(
            antpos=antpos,
            ref_wl=ref_wl,
            ref_freq=ref_freq,
            diameter=diameter,
            diameter_lambda=diameter_lambda,
            max_bl=max_bl,
            max_bl_lambda=max_bl_lambda,
            min_bl=min_bl,
            min_bl_lambda=min_bl_lambda,
            packing_density=packing_density,
            uv_cell_size=uv_cell_size,
            fulfill_tolerance=fulfill_tolerance,
            max_array_size=max_array_size,
            commanded=commanded,
            p_norm=p_norm,
            force_regenerate_commanded=True,
        )

        # Basic configuration derived from antenna positions.
        self.array_config = utils.get_array_config(self.antpos)

        # Initialized as None; may be populated by other methods.
        self.fulfilled_idx = None
        self.not_fulfilled_idx = None
        self.redundancy = None
        self.baseline_counts = None
        self.efficiency = None
        self.antpairs = None

        # Defaults for RULES algorithm keyword arguments.
        self._rules_defaults = {}
        self._set_rules_defaults(**rules_kwargs)

    def _configure_from_params(
        self,
        *,
        antpos=None,
        ref_wl=None,
        ref_freq=None,
        diameter=None,
        diameter_lambda=None,
        max_bl=None,
        max_bl_lambda=None,
        min_bl=None,
        min_bl_lambda=None,
        packing_density=None,
        uv_cell_size=None,
        fulfill_tolerance=None,
        max_array_size=None,
        commanded=None,
        p_norm=None,
        force_regenerate_commanded: bool = False,
    ):
        """
        (Re)configure AntArray state variables from high-level parameters.

        This centralizes the logic that keeps quantities such as baselines,
        diameter and wavelength consistent with one another. It can be called
        both during ``__init__`` and later (e.g. from :meth:`add_ant_rules`)
        when the user overrides AntArray-level parameters.
        """
        # Antenna positions
        if antpos is None:
            if hasattr(self, "antpos"):
                antpos = self.antpos
            else:
                antpos = np.array([[0.0, 0.0]])
        self.antpos = np.asarray(antpos, dtype=float)

        # Reference wavelength / frequency
        if ref_wl is None and ref_freq is None and hasattr(self, "ref_wl"):
            ref_wl = self.ref_wl
            ref_freq = self.ref_freq
        self.ref_wl, self.ref_freq = self._initialize_ref_wl(ref_wl, ref_freq)

        # Baseline ranges: max baseline
        if max_bl is None and max_bl_lambda is None and hasattr(self, "max_bl"):
            max_bl = self.max_bl
            max_bl_lambda = self.max_bl_lambda
        self.max_bl, self.max_bl_lambda = self._resolve_units(
            max_bl, max_bl_lambda, self.ref_wl, name="max_bl", default_lambda=100
        )

        # Baseline ranges: min baseline
        if min_bl is None and min_bl_lambda is None and hasattr(self, "min_bl"):
            min_bl = self.min_bl
            min_bl_lambda = self.min_bl_lambda
        self.min_bl, self.min_bl_lambda = self._resolve_units(
            min_bl, min_bl_lambda, self.ref_wl, name="min_bl", default_lambda=10
        )

        # Dish diameter
        if diameter is None and diameter_lambda is None and hasattr(self, "diameter"):
            diameter = self.diameter
            diameter_lambda = self.diameter_lambda
        self.diameter, self.diameter_lambda = self._resolve_units(
            diameter, diameter_lambda, self.ref_wl, name="diameter", default_lambda=5
        )

        # Enforce basic invariants between baselines and diameter.
        if self.min_bl < self.diameter:
            self._verbose_print(
                "Minimum baseline shorter than diameter; setting min_bl = diameter."
            )
            self.min_bl = self.diameter
            self.min_bl_lambda = self.min_bl / self.ref_wl

        if self.max_bl < self.min_bl:
            self._verbose_print(
                "Maximum baseline shorter than minimum baseline; setting max_bl = min_bl."
            )
            self.max_bl = self.min_bl
            self.max_bl_lambda = self.max_bl / self.ref_wl

        # Packing density / uv_cell_size
        if packing_density is None:
            if hasattr(self, "packing_density"):
                packing_density = self.packing_density
            else:
                packing_density = 2.0
        self.packing_density = packing_density

        if uv_cell_size is not None:
            self.uv_cell_size = uv_cell_size
        else:
            # Default derived from wavelength and packing density.
            self.uv_cell_size = self.ref_wl / self.packing_density

        # Fulfillment tolerance
        if fulfill_tolerance is None:
            if hasattr(self, "fulfill_tolerance"):
                fulfill_tolerance = self.fulfill_tolerance
            else:
                fulfill_tolerance = 1e-5
        self.fulfill_tolerance = fulfill_tolerance

        # Max array size
        if max_array_size is None and hasattr(self, "max_array_size"):
            max_array_size = self.max_array_size
        self.max_array_size = max_array_size

        # p-norm for distances
        if p_norm is None:
            if not hasattr(self, "p_norm"):
                self.p_norm = np.inf
        else:
            self.p_norm = p_norm

        # Commanded uv points
        if commanded is not None:
            self.commanded = commanded
        else:
            if force_regenerate_commanded or not hasattr(self, "commanded"):
                self.commanded = utils.generate_commanded_square(
                    self.uv_cell_size,
                    min_bl=self.min_bl,
                    max_bl=self.max_bl,
                    show_plot=False,
                )

    # ---------- Utility functions as instance methods -----------

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

    def _resolve_units(
        self,
        value: float,
        value_lambda: float,
        ref_wl: float,
        name: str,
        default_lambda: float = 2,
    ):
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
        self.antpairs = utils.select_baselines(self.commanded, self.antpos)
        return self.antpairs

    def get_redundancy(self, bins: int = 10) -> np.ndarray:
        """Compute and store redundancy histogram."""
        self.redundancy = utils.get_redundancy(self.antpos, bins=bins)
        return self.redundancy

    def generate_commanded_square(
        self,
        uv_cell_size: float = None,
        min_bl: float = None,
        max_bl: float = None,
        show_plot: bool = True,
        ax=None,
    ):
        """
        Generate and store a square grid of commanded uv points.

        This is a convenience wrapper around utils.generate_commanded_square
        that updates ``self.commanded``.
        """
        if uv_cell_size is None:
            uv_cell_size = self.uv_cell_size
        if min_bl is None:
            min_bl = self.min_bl
        if max_bl is None:
            max_bl = self.max_bl

        self.commanded = utils.generate_commanded_square(
            uv_cell_size=uv_cell_size,
            min_bl=min_bl,
            max_bl=max_bl,
            show_plot=show_plot,
            ax=ax,
        )
        return self.commanded

    def check_fulfillment(self):
        """Check and store fulfillment indices for commanded uv points."""
        (
            self.fulfilled_idx,
            self.not_fulfilled_idx,
        ) = geometry.check_fulfillment(self.commanded, self.antpos, self.uv_cell_size)
        return self.fulfilled_idx, self.not_fulfilled_idx

    def get_baseline_counts(self):
        """Compute and store baseline loss counts for each antenna."""
        self.baseline_counts = utils.get_baseline_counts(self)
        return self.baseline_counts

    def export_antpos_csv(self, filename: str):
        """Export antenna positions to a CSV file."""
        utils.export_antpos_csv(self.antpos, filename)

    def get_efficiency(self) -> float:
        """Compute and store overall array efficiency."""
        self.efficiency = utils.get_efficiency(self)
        return self.efficiency

    def get_efficiency_array(self) -> np.ndarray:
        """Get efficiency as an array, if available."""
        if hasattr(self, "history") and "efficiency" in self.history:
            return np.array(self.history["efficiency"])
        raise AttributeError("Efficiency history is not available.")

    # ---------- Attribute checks / completion -----------

    def check_missing_attributes(self, verbose: bool = True) -> dict:
        """
        Check for missing attributes that are useful for analysis.

        Parameters
        ----------
        verbose : bool, default True
            If True, prints information about missing attributes.

        Returns
        -------
        dict
            Dictionary with attribute names as keys and booleans indicating
            whether they are missing.
        """
        attributes = [
            'antpos',
            'commanded',
            'fulfilled_idx',
            'not_fulfilled_idx',
            'redundancy',
            'baseline_counts',
            'array_size',
            'array_config',
            'antpairs',
            'efficiency',
            'efficiency_array',
        ]

        missing = {attr: not hasattr(self, attr) for attr in attributes}

        if verbose:
            print("Missing attributes:")
            for attr, is_missing in missing.items():
                print(f"  {attr}: {'missing' if is_missing else 'present'}")

        return missing

    def compute_missing_attributes(
        self,
        *,
        select_baselines: bool = True,
        array_size: bool = True,
        array_config: bool = True,
        redundancy: bool = True,
        baseline_counts: bool = True,
        efficiency_array: bool = False,
        efficiency: bool = True,
        force_recompute: bool = False,
        verbose: bool = True,
    ):
        """
        Compute missing attributes as needed.

        Parameters
        ----------
        select_baselines, array_size, array_config, redundancy,
        baseline_counts, efficiency_array, efficiency : bool, optional
            Flags indicating which attributes to compute.
        force_recompute : bool, optional
            If True, recompute attributes even if they already exist.
        verbose : bool, optional
            If True, print progress messages.
        """
        missing = self.check_missing_attributes(verbose=False)

        # Antenna positions and commanded grid should already exist.
        if missing['antpos']:
            raise ValueError("Antenna positions (antpos) are missing.")

        if missing['commanded']:
            if verbose:
                print("Generating commanded uv points...")
            self.generate_commanded_square(show_plot=False)

        if array_size and (missing['array_size'] or force_recompute):
            if verbose:
                print("Computing array size...")
            self.get_array_size()

        if array_config and (missing['array_config'] or force_recompute):
            if verbose:
                print("Computing array configuration...")
            self.get_array_config()

        if redundancy and (missing['redundancy'] or force_recompute):
            if verbose:
                print("Computing redundancy histogram...")
            self.get_redundancy()

        if baseline_counts and (missing['baseline_counts'] or force_recompute):
            if verbose:
                print("Computing baseline counts...")
            self.get_baseline_counts()

        if select_baselines and (missing['antpairs'] or force_recompute):
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

    # ---------- Random placement convenience -----------

    def add_ant_random(self, *args, **kwargs):
        """
        Convenience wrapper around the random placement algorithm.

        See :func:`uvrules.algos.random.add_ant_random` for details.
        """
        return random.add_ant_random(self, *args, **kwargs)

    # ---------- Save/load -----------

    def save(self, filename: str = None):
        """
        Save the AntArray instance to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Path to the file. If None, a default name is used.
        """
        if filename is None:
            filename = getattr(self, "path_to_file", "AntArray.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self._verbose_print(f"AntArray saved to {filename}.")

    @staticmethod
    def load(filename: str):
        """
        Load an AntArray instance from a pickle file.

        Parameters
        ----------
        filename : str
            Path to the pickle file.

        Returns
        -------
        AntArray
            Loaded instance.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No such file: {filename}")
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    # ---------- RULES defaults and entry point -----------

    def _set_rules_defaults(self, **rules_kwargs):
        """
        Update the default keyword arguments that will be passed to
        :meth:`add_ant_rules`.

        This is used internally by ``__init__``, but can also be used by
        advanced users to tweak defaults between runs.
        """
        if not hasattr(self, "_rules_defaults"):
            self._rules_defaults = {}

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

        Call-time arguments override defaults.
        """
        defaults = getattr(self, "_rules_defaults", {}).copy()
        for key, value in call_kwargs.items():
            defaults[key] = value
        return defaults

    def add_ant_rules(self, **kwargs):
        """
        Wrapper to run the RULES algorithm.

        Any keyword arguments that correspond to AntArray parameters
        (e.g. ``diameter``, ``max_bl``, ``rho``) will update this instance
        before the algorithm runs. All remaining keyword arguments are treated
        as parameters to the RULES algorithm itself, which are as follows:


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

        
       
        """
        # Split kwargs into AntArray-level overrides vs RULES parameters.
        antarray_overrides = {}
        rules_kwargs = {}

        for key, value in kwargs.items():
            if key == "rho":
                # Alias for packing_density; if both are provided, rho wins.
                antarray_overrides["packing_density"] = value
            elif key in ANTARRAY_PARAM_NAMES:
                antarray_overrides[key] = value
            else:
                rules_kwargs[key] = value

        # If any AntArray parameters were overridden, reconfigure self.
        if antarray_overrides:
            base_config = {
                "antpos": getattr(self, "antpos", None),
                "ref_wl": getattr(self, "ref_wl", None),
                "ref_freq": getattr(self, "ref_freq", None),
                "diameter": getattr(self, "diameter", None),
                "diameter_lambda": getattr(self, "diameter_lambda", None),
                "max_bl": getattr(self, "max_bl", None),
                "max_bl_lambda": getattr(self, "max_bl_lambda", None),
                "min_bl": getattr(self, "min_bl", None),
                "min_bl_lambda": getattr(self, "min_bl_lambda", None),
                "packing_density": getattr(self, "packing_density", None),
                "uv_cell_size": getattr(self, "uv_cell_size", None),
                "fulfill_tolerance": getattr(self, "fulfill_tolerance", None),
                "max_array_size": getattr(self, "max_array_size", None),
                "commanded": getattr(self, "commanded", None),
                "p_norm": getattr(self, "p_norm", None),
            }

            # Apply overrides on top of the current configuration.
            base_config.update(antarray_overrides)

            # Decide whether we need to regenerate the commanded uv grid.
            force_regenerate_commanded = any(
                key in antarray_overrides
                for key in (
                    "min_bl",
                    "min_bl_lambda",
                    "max_bl",
                    "max_bl_lambda",
                    "packing_density",
                    "uv_cell_size",
                )
            )

            self._configure_from_params(
                **base_config,
                force_regenerate_commanded=force_regenerate_commanded,
            )

            # Array configuration may have changed.
            self.array_config = utils.get_array_config(self.antpos)

        # Merge RULES defaults with remaining RULES kwargs (call-time override).
        merged_rules_kwargs = self._merge_rules_kwargs(rules_kwargs)

        # Warn if call-time RULES args override defaults defined on the instance.
        import warnings as _warnings
        for k, v in rules_kwargs.items():
            if hasattr(self, "_rules_defaults") and k in self._rules_defaults:
                _warnings.warn(
                    f"Overriding default RULES parameter '{k}' defined on "
                    f"AntArray (default={self._rules_defaults[k]!r}) with {v!r}",
                    UserWarning,
                    stacklevel=2,
                )

        # Call the internal RULES implementation.
        return _rules._add_ant_rules_implementation(self, **merged_rules_kwargs)

