from skfem import *
from skfem.helpers import dot, grad
import numpy as np
import matplotlib.pyplot as plt

epsilon_list = [0.1,0.01,0.001]
colour_list = ['b','g','r']

def graded_mesh(N, p=2):
    """Generate a 1D mesh denser near boundaries."""
    x_half = np.linspace(0, 1, N // 2) ** p
    x_right_half = 1 - (np.linspace(0, 1, N // 2) ** p)
    x = np.concatenate((x_half[:-1], x_right_half[::-1]))  # join two halves
    return np.unique(x)  # remove any duplicates

def get_soln(mesh_option, epsilon):

    if mesh_option == 0:
        # Create a 1D mesh
        m = MeshLine(np.linspace(0, 1, 100))  # 100 nodes over [0,1]
    elif mesh_option == 1:
        m = MeshLine(graded_mesh(100, 2))

    e = ElementLineP1()  # P1 = piecewise linear
    basis = Basis(m, e)

    # Bilinear form: epsilon^2 * u'v' + u v
    @BilinearForm
    def bilinear(u, v, w):
        return epsilon**2 * dot(grad(u), grad(v)) + u * v

    # Right-hand side: ∫ v
    @LinearForm
    def rhs(v, w):
        return v

    # Assemble system
    A = bilinear.assemble(basis)
    b = rhs.assemble(basis)

    # Apply Dirichlet boundary conditions u(0) = u(1) = 0
    A, b = enforce(A, b, D=m.boundary_nodes())

    # Solve
    x = solve(A, b)
    return m, x

# Visualization
def visualize():

    for i,epsilon in enumerate(epsilon_list):
        m, x = get_soln(1, epsilon)
        # Plot FEM solution
        plt.plot(m.p[0], x, colour_list[i], label='FEM Solution: Epsilon = {}'.format(epsilon))

    # Plot exact solution
    EPSILON = 0.001
    x_vals = np.linspace(0, 1, 500)
    u_exact = 1 - np.cosh((x_vals - 0.5) / EPSILON) / np.cosh(0.5 / EPSILON)
    plt.plot(x_vals, u_exact, color='orange', label='Exact Solution: Epsilon = {}'.format(EPSILON))

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(r'FEM vs Exact Solution')
    plt.grid(True)
    return plt

if __name__ == "__main__":
    visualize().show()