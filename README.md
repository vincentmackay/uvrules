# uvrules

`uvrules` implements the RULES algorithm — **Regular UV Layout Engineering Strategy** — for generating radio interferometer arrays with complete and efficient uv-plane coverage.

This is a tool for designing antenna layouts that fulfill a specified set of baseline vectors (e.g., a hexagonal uv grid), with applications to 21 cm cosmology and radio interferometry more broadly.

---

## 🚧 Early version notice

This is an early release of the package. The interface may change, and documentation is still minimal.

If you're interested in using this package, please reach out — contributions and suggestions are welcome.

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

Full documentation and tutorials will be added soon.

In the meantime, see:

- `uvrules/antarray.py` for the core `AntArray` class
- `uvrules/algo/rules.py` for the RULES algorithm
- `uvrules/plotting.py` for basic visualizations of arrays and their associated PSFs

