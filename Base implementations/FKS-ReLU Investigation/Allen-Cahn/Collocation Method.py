from operator import length_hint

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os

# Problem parameters
EPSILON = 0.01
LAMBDA = 1

def ode(x, y):
    u, u_x = y
    u_xx = LAMBDA * u * (u**2 - 1) / EPSILON**2
    return np.vstack([u_x, u_xx])

# Boundary conditions: u(0) = 0, u(1) = 0
def bc(ya, yb):
    return np.array([ya[0] + 1, yb[0] - 1])

# Initial guess (zero function)
x = np.linspace(-1, 1, 401)
a = np.sqrt(LAMBDA / 2) / EPSILON
u_guess = np.tanh(a * x)
y_guess = np.vstack([u_guess, a * (1 - u_guess**2)])

sol = solve_bvp(ode, bc, x, y_guess,
                tol=1e-5, max_nodes=10000)

if not sol.success:
    raise RuntimeError(sol.message)

# Plot the solution
x_plot = np.linspace(-1, 1, 200)
u_numeric = sol.sol(x_plot)[0]
print(sol.x)
ys = [0 for i in range(0,len(sol.x))]
i_s = [i for i in range(0,len(sol.x))]

directory = "Collocation/" + str(EPSILON)
os.makedirs(directory, exist_ok=True)
filename = directory + "/params.npz"
with open(filename, 'wb') as file:
    np.savez(
        filename,
        breaks=sol.sol.x,
        polynomial_coeffs=sol.sol.c,
    )

# Exact solution for comparison (if desired)
a = 1/(EPSILON * np.sqrt(2))
u_exact = np.tanh(a*x_plot)

plt.plot(x_plot, u_numeric, 'r--', label='Numerical (solve_bvp)')
plt.plot(x_plot, u_exact, 'k-', label='Exact')
plt.scatter(sol.x,ys)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(r"$ \varepsilon^2 u'' + \lambda u(1-u^2) = 0$, $\varepsilon$ = {}".format(EPSILON))
plt.legend()
plt.grid(True)
plt.show()

plt.plot(i_s,sol.x)
plt.show()