# Modified Dubiner hp-cg TFEM (cg-TFEM Prototype)

This repository implements a finite element framework based on a modified Dubiner modal basis and structured pseudo-mass matrix factorization for continuous Galerkin (CG) methods on triangular elements.

The goal is to supplement the findings from:

* Appleton & Helenbrook (2019)
  *A High-Order Lower-Triangular Pseudo-Mass Matrix for Explicit Time Advancement of hp Triangular Finite Element Methods*
  https://arxiv.org/abs/1906.10774

and develop a decoupled DOF explicit time-stepping framework for hp finite element methods.

---

# Core Idea

This project explores a structured FEM formulation:

* **hp-refined triangular finite elements**
* **Modified Dubiner modal basis**
* **Hierarchical vertex/edge/interior structure**
* **Exact mass matrix construction**
* **Operator factorization:** `M → T → L`
* **Decoupled DOF updates for explicit time integration**

---

# Project Structure

```text
hpcgtfem/
│
├── pyproject.toml          # Project configuration and packaging
├── requirements.txt        # External dependencies
├── README.md               # Project overview
├── LICENSE                 # MIT license
├── .gitignore              # Git ignore rules
│
├── src/                    # Core library (source of truth)
│   └── localtfem/
│       ├── __init__.py
│       └── modified_dubiner/
│           ├── __init__.py
│           ├── jacobi.py       # Jacobi polynomial utilities
│           ├── basis.py        # Modified Dubiner basis construction
│           ├── mass.py         # Exact FEM mass matrix assembly
│           ├── transform.py    # Basis transformations (T operator)
│           ├── pseudo_mass.py  # Lower-triangular pseudo-mass operator (L)
│           ├── mesh.py         # Triangular mesh generation
│           └── heat.py         # Heat equation solver prototype
│
├── tests/                  # Unit tests for mathematical correctness
├── notebooks/              # Symbolic exploration and derivations
├── docs/                   # Theory and documentation
├── scripts/                # Reproducibility scripts
├── examples/               # Minimal runnable examples
└── data/                   # Reference meshes and datasets (optional)
```

---

# 🚀 Setup and Installation

This project uses a standard Python **src-layout** together with a virtual environment and an editable installation for development.

## 1. Create a virtual environment

From the project root:

```bash
python -m venv .venv
```

---

## 2. Activate the environment

### Windows (PowerShell)

REMOVE`powershell
.venv\Scripts\Activate.ps1
REMOVE`

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Once activated, your shell prompt should indicate the active virtual environment, for example:

```text
(.venv) C:\Users\jayma\GitHub\hpcgtfem>
```

---

## 3. Install dependencies

Install the external Python packages required by the project:

```
pip install -r requirements.txt
```

This installs packages such as **NumPy** and **SymPy** into the active virtual environment.

---

## 4. Install the package in editable mode

From the project root (the directory containing `pyproject.toml`), run:

```bash
pip install -e .
```

This installs the `localtfem` package in **editable mode**, allowing changes made under `src/localtfem/` to be used immediately without reinstalling.

---

## 5. Verify the installation

Launch Python or a notebook and verify that the package imports correctly:

REMOVE```python
from localtfem.modified_dubiner.jacobi import jacobi_coeffs_float

print(jacobi_coeffs_float(4, 0, 0))
REMOVE```

If this executes successfully, the development environment has been configured correctly.

---

## 6. (Optional) Configure Jupyter

If you intend to work in Jupyter notebooks, install an IPython kernel for the virtual environment:

REMOVE`bash
pip install ipykernel
python -m ipykernel install --user --name localtfem
REMOVE`

Then select the **localtfem** kernel when opening notebooks.

---

## Quick Start

After cloning the repository, the complete setup is:

REMOVE```bash
python -m venv .venv

# Windows PowerShell

.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
REMOVE```

---

# Notes on the Project Layout

* `requirements.txt` lists external dependencies.
* `pyproject.toml` defines the `localtfem` package and build configuration.
* `pip install -e .` links the source tree into the active Python environment for development.
* `src/` is the single source of truth for all library code.

---

# Current Development Status

This project is under active development.

The initial implementation focuses on:

* Jacobi polynomial generation
* Modified Dubiner basis construction
* Exact symbolic and numerical verification
* Mass matrix assembly
* Operator factorization
* Explicit time-stepping algorithms

Additional modules and numerical examples will be added as the implementation progresses.

---

# References

Appleton, M. A., & Helenbrook, B. T. (2019).

*A High-Order Lower-Triangular Pseudo-Mass Matrix for Explicit Time Advancement of hp Triangular Finite Element Methods.*

https://arxiv.org/abs/1906.10774
