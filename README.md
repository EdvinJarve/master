# Master thesis work - Solving Cardiac Electrophysiology Models Using Physically Informed Neural Networks

This master thesis work contains the implementation and results of monodomain and bidomain simulations using FEM and PINNs.

## Description

### Monodomain Simulations

The `monodomain` directory contains the scripts for monodomain simulations:
- **mono_case1.py**: Script for the monodomain simulation case with analytic solution for testing the solver.
- **mono_case2.py**: Script for the monodomain simulation case with an applied corner current.
- **mono_case3.py**: Script for the third monodomain simulation case with an applied corner current with a strong nonisotropic conductivity tensor.
### Bidomain Simulations

The `bidomain` directory contains the scripts for bidomain simulations:
- **bi_case1.py**: Script for the bidomain simulation case with analytic solution for testing the solver.
- **bi_case2.py**: Script for the bidomain simulation case with an applied corner current.
- **bi_case3.py**: Script for the bidomain simulation case with an applied corner current with a strong nonisotropic conductivity tensor.
### Bidomain Simulations

### Results

The `monodomain_results` and `bidomain_results` directories contain the results for each case

### Utilities

The `utils` directory contains utility scripts:
- **heart_solver_fem.py**: Functions and classes for solving heart models using FEM.
- **heart_solver_pinns.py**: Functions and classes for solving heart models using PINNs.

## How to Run the Simulations

To run the bidomain simulations use:
```bash
python bidomain/bi_case1.py
python bidomain/bi_case2.py
python bidomain/bi_case3.py
```

To run the monodomain simulations use:
```bash
python monodomain/mono_case1.py
python monodomain/mono_case2.py
python monodomain/mono_case3.py
```
