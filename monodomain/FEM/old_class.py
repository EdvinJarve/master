class MonodomainSolverOperatorSpltting:
    def __init__(self, mesh_type, mesh_args, element_degree, T, dt):
        self.comm = MPI.COMM_WORLD
        if mesh_type == "unit_square":
            self.domain = mesh.create_unit_square(self.comm, *mesh_args)
        elif mesh_type == "unit_cube":
            self.domain = mesh.create_unit_cube(self.comm, *mesh_args)
        # Add more mesh types as needed
        else:
            raise ValueError("Unsupported mesh type")
        
        self.V = fem.functionspace(self.domain, ("Lagrange", element_degree))
        
        self.v_n = fem.Function(self.V)
        self.s_n = fem.Function(self.V)  # Assuming s is defined on the same space

        self.mesh_args = mesh_args
        self.T = T
        self.dt = dt
        self.steps = int(T/dt)


    # Define the system of ODEs
    # The system of ODES below is simply hardcoded to reproduse analytical results to check the 
    # credibility of the solver
    def system_of_odes(self, t, y):
        v, s = y
        dvdt = -s
        dsdt = v
        return [dvdt, dsdt]
    
    # Function for the stimulus
    # Also here the expression is hardcoded
    def stimulus_expression(self, t, domain):
        x = ufl.SpatialCoordinate(domain)
        return 8 * ufl.pi**2 * ufl.cos(2 * ufl.pi * x[0]) * ufl.cos(2 * ufl.pi * x[1]) * ufl.sin(t)

    def numpy_to_fenicsx(self, array):
        """Convert a 2D numpy array to a FEniCSx function on a given function space V."""
        # Ensure the array is flat
        array_flat = array.flatten()
        # Create a FEniCSx function
        u = fem.Function(self.V)
        # Set values from NumPy array to FEniCSx function
        u.vector.setArray(array_flat)
        # Scatter data to update ghost values if running in parallel
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return u


    def fenicsx_to_numpy(self, fenics_function):
        """Convert a FEniCSx function to a 2D numpy array matching the original array shape."""
        # Extract the flat array directly from the FEniCSx function

        x_points, y_points = self.mesh_args

        array_flat = fenics_function.x.array
        array_flat = np.reshape(array_flat, (x_points + 1, y_points + 1))
        return array_flat


    def solve_ODE(self, S0, v0, t_start, t_end):
        """
        Solves the ODE system for each mesh point with varying initial v and s values.

        S0: Initial value of S for the entire mesh.
        v0: Initial value of v for the entire mesh.
        t_start: Start time of the time-evolution.
        t_end: End time of the time-evolution.
        x_points: Mesh points in the x direction.
        y_points: Mesh points in the y direction.

        Returns:
        - v_solutions: A 2D array of the final v values for each mesh point.
        - s_solutions: A 2D array of the final s values for each mesh point.
        """
        # Initialize arrays to store the final solutions

        x_points, y_points = self.mesh_args
        v_solutions = np.zeros((x_points+1, y_points+1))
        s_solutions = np.zeros((x_points+1, y_points + 1))

        # Solve for each mesh point
        for i in range(x_points):
            for j in range(y_points):
                # Initial condition for this point
                y0 = [v0[i, j], S0[i, j]]

                # Solve the system of ODEs for this point
                solution = solve_ivp(self.system_of_odes, [t_start, t_end], y0, method='RK45')

                # Store the final solution for this point
                v_solutions[i, j] = solution.y[0, -1]
                s_solutions[i, j] = solution.y[1, -1]

        return v_solutions, s_solutions

    def solve_PDE(self, t_start, t_end, v_init):
        # Assume initial condition is given and correctly initialized
        v_n = fem.Function(self.V)
        v_n.interpolate(v_init)  # Set the initial condition

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        t = fem.Constant(self.domain, PETSc.ScalarType(t_start))
        stimulus = self.stimulus_expression(t, self.domain)

        # Define the variational problem
        a = u * v * ufl.dx + self.dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (v_n + self.dt * stimulus) * v * ufl.dx

        # Assemble system
        bilinear_form = fem.form(a)
        linear_form = fem.form(L)

        A = assemble_matrix(bilinear_form)
        A.assemble()
        b = create_vector(linear_form)

        # Setup the solver
        solver = PETSc.KSP().create(self.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # Solve the system for one time step
        assemble_vector(b, linear_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        # Solve the linear system
        solver.solve(b, v_n.vector)
        v_n.x.scatter_forward()

        # Return the updated solution
        return v_n
    
    def solve_monodomain(self, initial_v_flat, initial_s_flat):
        # Assuming self.u_final is a dolfinx.fem.Function initialized to handle the PDE solutions
        i = 0
        while i < self.steps:
            # Solve the ODE for the current step
            v_flat, s_flat = self.solve_ODE(initial_s_flat, initial_v_flat, i * self.dt)
            
            # Convert the 1D numpy array v_flat to a FEniCSx function
            self.u_final.vector.setArray(v_flat)
            self.u_final.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # Solve the PDE for the current step
            self.u_final = self.solve_PDE(i * self.dt, (i + 1) * self.dt, self.u_final)

            # Update initial conditions for the next step
            initial_v_flat = self.u_final.vector.getArray()  # Get the array representation of the function
            initial_s_flat = s_flat

            i += 1

        return initial_v_flat
