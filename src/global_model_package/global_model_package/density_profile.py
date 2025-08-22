import  fenics as fe
from dolfin import *
from mshr import *


R = 0.5  # Radius of the cylinder
L = 2   #length of the cylinder

D_a = 1
A = 0.1
u_B = 1



constant_shift = fe.Constant(0.01)

#Données de précision
Number_of_elemnts = 60 #définit la précision du calcul et augmente le temps de calcul
Degree_of_elements = 1 #définit la précision du calcul

# Define the geometry and mesh
cylinder = Cylinder(fe.Point(0, 0, 0), fe.Point(L, 0, 0), R, R)

mesh = generate_mesh(cylinder, Number_of_elemnts) # Generates a mesh with Number_of_elements divisions
lagrange_vector_space_first_order = fe.FunctionSpace(
        mesh,
        "Lagrange",
        Degree_of_elements,
    )
print("mesh generated")

n_trial = fe.TrialFunction(lagrange_vector_space_first_order) #guess function
v_test = fe.TestFunction(lagrange_vector_space_first_order) #test function v

#definition of the weak form of the problem
weak_form_rhs = - u_B * constant_shift * v_test * fe.ds  
weak_form_lhs = (
    fe.dot(fe.grad(n_trial), fe.grad(v_test)) * fe.dx #consequence of the IPP
    -
    u_B * n_trial * v_test * fe.ds #consequence of the boundary conditions
    -
    A * n_trial * v_test * fe.dx 
)


#solving
n_solution = fe.Function(lagrange_vector_space_first_order)
fe.solve(
    weak_form_lhs == weak_form_rhs,
    n_solution,
)
print(fe.assemble(n_solution * fe.dx))



