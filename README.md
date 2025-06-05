# uvrules

`uvrules` implements the **Radio-array _uv_ Layout Engineering Strategy (RULES)** algorithm for generating radio interferometer arrays with complete and efficient uv-plane coverage.

This is a tool for designing antenna layouts that fulfill a regular square lattice of _uv_ points, with applications to 21 cm cosmology and radio interferometry more broadly.

---

## Early version notice

This is an early release of the package. The interface may change, and documentation is still minimal.

If you're interested in using this package, please reach out. Contributions and suggestions are welcome!

---

## Features

- Generate uv-complete layouts from commanded _uv_ points  
- Enforce physical layout constraints (dish diameter, max array size, number of antennas, etc.) 
- Parallelized 
- Visualization of array layout, _uv_ coverage, and synthesized beam  

---

## Installation

Clone the repo and install locally:

```bash
git clone https://github.com/vincentmackay/uvrules.git
cd uvrules
pip install .
```

---


## Documentation

See Jupyter notebook tutorial. Full documentation will be added soon.

In the meantime, see:

- `uvrules/antarray.py` for the core `AntArray` class
- `uvrules/algo/rules.py` for the RULES algorithm
- `uvrules/plotting.py` for basic visualizations of arrays and their associated PSFs

