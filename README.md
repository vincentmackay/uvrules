# uvrules

`uvrules` implements the RULES algorithm — **Regular UV Layout Engineering Strategy** — for generating radio interferometer arrays with complete and efficient uv-plane coverage.

This is a tool for designing antenna layouts that fulfill a specified set of baseline vectors (e.g., a hexagonal uv grid), with applications to 21 cm cosmology and radio interferometry more broadly.

---

## Version used in the accompanying paper

The `v0.1.0` release of `uvrules` corresponds to the implementation used in
the analysis for:
> MacKay et al., *Complete Sampling of the Plane with Realistic Radio Arrays: Introducing the RULES Algorithm, with Application to 21 cm Foreground Wedge Removal*, 2025, ApJ (accepted, in review).

---

## Features

- Generate uv-complete layouts from commanded uv points  
- Enforce physical layout constraints (dish diameter, max array size, etc.)  
- Optimize placements for coverage, compactness, or spacing  
- Parallel and sequential modes available  
- Visualization of array layout, uv coverage, and synthesized beam  

---

## Installation

Clone the repo and install locally:

```bash
git clone https://github.com/yourusername/uvrules.git
cd uvrules
pip install .
```

---


## Documentation

See examples/RULES_tutorial.ipynb for a hands-on demonstration of building a uv-complete layout with AntArray and add_ant_rules.

In the meantime, see:

- `uvrules/antarray.py` for the core `AntArray` class
- `uvrules/algo/rules.py` for the RULES algorithm
- `uvrules/plotting.py` for basic visualizations of arrays and their associated PSFs

