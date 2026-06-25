# Modified Dubiner hp-cg TFEM (cg-TFEM Prototype)

This repository implements a finite element framework based on a modified Dubiner modal basis and structured pseudo-mass matrix factorization for continuous Galerkin (CG) methods on triangular elements.

The goal is to supplement the findings from

- Appleton & Helenbrook (2019),  
  *A High-Order Lower-Triangular Pseudo-Mass Matrix for Explicit Time Advancement of hp Triangular Finite Element Methods*  
  https://arxiv.org/abs/1906.10774

and develop a decoupled DOF explicit time-stepping framework for hp finite element methods.

---

# Core Idea

This project explores a structured FEM formulation:

- **hp-refined triangular finite elements**
- **Modified Dubiner modal basis**
- **Hierarchical vertex/edge/interior structure**
- **Exact mass matrix construction**
- **Operator factorization: M → T → L**
- **Decoupled DOF updates for explicit time integration**

---

# Project Structure

```text
modified_dubiner/
│
├── pyproject.toml          # Project configuration and dependencies
├── README.md               # Project overview (this file)
├── LICENSE                 # MIT license
├── .gitignore              # Git ignore rules
├── .env.example            # Environment configuration template
│
├── src/                    # Core library (source of truth)
│   └── modified_dubiner/
│       ├── jacobi.py       # Jacobi polynomial utilities
│       ├── basis.py        # Modified Dubiner basis construction
│       ├── mass.py         # Exact FEM mass matrix assembly
│       ├── transform.py    # Basis transformations (T operator)
│       ├── pseudo_mass.py  # Lower-triangular / decoupled operator (L)
│       ├── mesh.py         # Triangular mesh generation
│       └── heat.py         # Heat equation solver prototype
│
├── tests/                  # Unit tests for mathematical correctness
│
├── notebooks/              # Symbolic exploration and derivations
│   ├── jacobi_exploration.ipynb
│   ├── dubiner_basis.ipynb
│   ├── mass_matrix.ipynb
│   └── heat_equation_demo.ipynb
│
├── docs/                   # Documentation (Sphinx / LaTeX / theory)
│   ├── basis_derivation.md
│   ├── operator_factorization.md
│   └── numerical_results.md
│
├── scripts/                # Reproducibility scripts for experiments
│   ├── run_heat_test.py
│   ├── reproduce_paper_figures.py
│   └── benchmark_solver.py
│
├── examples/               # Minimal runnable examples
│   ├── single_element_demo.py
│   └── small_mesh_demo.py
│
└── data/                   # Reference meshes and datasets (optional)
