import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
import time
from dolfinx import fem, mesh, plot, default_scalar_type; from dolfinx.fem import FunctionSpace
from mpi4py import MPI
import ufl
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile

"""
This code seeks to solve the monodomain model for 2D mesh, which physically represent cardiac tissue. 
The ionic model is chosen to be the FitzHugh model, which is a coupled ODE for the ionic current.
To solve this system of equations, operator splitting is used to simplify the equations and thus the implementation
Specifically, the monodomain model, which is given by:

K ∇⋅(M_i ∇v) = χ (C_m dv/dt + I_ion)

is devided into

χ C_m dv/dt = K ∇⋅(M_i ∇v) - χ I_ion = L_1 + L_2

where using Gudanov operator splitting scheme, we solve the equations as proposed by Qu and Garfinkel by:

1. --------- Solve the ODE from t_n to t_(n+1)-------------

dV/dt = c_1 / v^2_amp (V - v_rest)(V - v_th)(v_peak - V)
        -c_2 / v_amp (V - v_rest)W + I_app

dW/dt = b(V - v_rest -c_3 w)

save the solutons for v(t_n+1) and w(t_n+1) as v_(n+1) and w_(n+1)

2. ------- Solve the PDE ---------------

dv/dt = K ∇⋅(M_i ∇v) + I_app, with inital condition v_(n+1)

save the resulting soluton as v_(n+1) and w_(n+1)

3. ---------- Solve the ODE-----------

Solve the PDE in 1. with inital conditions as v_(n+1) and w_(n+1)

4. ------- Repeat 2 until condition

"""

# Model parameters from book for the FitzHugh Model 
a = 0.13; b = 0.013; c_1 = 0.26; c_2 = 0.1; c_3 = 1.0; i_app_val = 0.05 

v_rest = -85  # resting potential [mV]
v_peak = 40   # peak potential (parameter from book)[mV]
v_amp = v_peak - v_rest  # [mV]
v_th = v_rest + a * v_amp  # [mV]
I_app_val = v_amp * i_app_val  

# Function to provide time-dependent applied current
def I_app(t):
    return I_app_val if 50 <= t <= 60 else 0

