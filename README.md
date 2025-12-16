# uvrules: Generating *uv*-Complete Arrays

`uvrules` implements the **Regular** ***uv*** **Layout Engineering Strategy (RULES)** algorithm for generating radio interferometer arrays with complete and efficient *uv*-plane coverage.

This tool generates antenna layouts that fulfill a specified set of baselines, and is optimized for sets of *uv*-complete baselines, as defined in MacKay et al. 2025 (*ApJ*, accepted), with applications to 21 cm cosmology and radio interferometry more broadly.

## Features

- Generate sets of *uv* points that are *uv* complete
- Generate *uv*-complete layouts from commanded set of *uv* points
- Enforce physical layout constraints (dish diameter, max array size, etc.)  
- Optimize placements for coverage, compactness, or spacing  
- Parallel and sequential modes available  
- Visualization of array layout, *uv* coverage, synthesized beam, and algorithmic statistics  


## Installation

You can install `uvrules` via pip:

```bash
pip install uvrules
```


## Documentation

See examples/RULES_tutorial.ipynb for a hands-on demonstration of building a uv-complete layout with AntArray and add_ant_rules.

In the meantime, see:

- `uvrules/antarray.py` for the core `AntArray` class
- `uvrules/algo/rules.py` for the RULES algorithm
- `uvrules/plotting.py` for basic visualizations of arrays and their associated PSFs


## Version used in the accompanying paper

The `v0.1.0` release of `uvrules` corresponds to the implementation used in
the analysis for:
> MacKay et al., *Complete Sampling of the Plane with Realistic Radio Arrays: Introducing the RULES Algorithm, with Application to 21 cm Foreground Wedge Removal*, 2025, ApJ (accepted, in review).


## License

This project is licensed under the MIT License - see the LICENSE file for details.
