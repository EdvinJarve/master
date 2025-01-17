# Master thesis work - Solving Cardiac Electrophysiology Models Using Physically Informed Neural Networks

This master thesis work contains the implementation and results of monodomain and bidomain simulations using FEM and PINNs.

## Description

### Monodomain Simulations

The `monodomain_simulations` directory contains the scripts for monodomain simulations:
- **analytical_case.py**: Script for the monodomain simulation case with analytic solution for testing the solver.
- **corner_current.py**: Script for the monodomain simulation case with an applied corner current.
- **corner_current_nonisotropic.py**: Script for the third monodomain simulation case with an applied corner current with a strong nonisotropic conductivity tensor.
- **fitzhugh_nagumo.py**: Script for the fourth monodomain simulation where the current is goverened by the FitzHugh-Nagumo ODE.

### Bidomain Simulations

The `bidomain` directory contains the scripts for bidomain simulations (currently not working):
- **bi_case1.py**: Script for the bidomain simulation case with analytic solution for testing the solver.
- **bi_case2.py**: Script for the bidomain simulation case with an applied corner current.
- **bi_case3.py**: Script for the bidomain simulation case with an applied corner current with a strong nonisotropic conductivity tensor.

### Results

The `monodomain_results` and `bidomain_results` directories contain the results for each case. Preliminary results:
- **Analytical case**: Both **FEM** and **PINNs** produce similar results, with an absolute error on the order of \(10^{-2}\).
- **Corner Current**: Yet to converge properly
- **FitzHugh-Nagumo**:  
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
