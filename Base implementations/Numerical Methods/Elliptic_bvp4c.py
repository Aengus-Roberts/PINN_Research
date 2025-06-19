from operator import length_hint

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# Problem parameters
EPSILON = 0.01

def ode(x,y):
    u = y[0]
    u_x = y[1]
    u_xx = (u-1)/EPSILON**2
    return np.vstack([u_x,u_xx])

# Boundary conditions: u(0) = 0, u(1) = 0
def bc(ya, yb):
    return np.array([ya[0], yb[0]])

# Initial guess (zero function)
x = np.linspace(0, 1, 20)
y_guess = np.zeros((2, x.size))  # [u, u']

# Solve the BVP
sol = solve_bvp(ode, bc, x, y_guess)

# Check result
if sol.success:
    print("BVP solved successfully")
else:
    print("BVP solver failed")

# Plot the solution
x_plot = np.linspace(0, 1, 200)
u_numeric = sol.sol(x_plot)[0]
print(sol.x)
ys = [0 for i in range(0,len(sol.x))]
i_s = [i for i in range(0,len(sol.x))]

# Exact solution for comparison (if desired)
u_exact = 1 - np.cosh((x_plot - 0.5)/EPSILON) / np.cosh(0.5/EPSILON)

plt.plot(x_plot, u_numeric, 'r--', label='Numerical (solve_bvp)')
plt.plot(x_plot, u_exact, 'k-', label='Exact')
plt.scatter(sol.x,ys)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(r"$- \varepsilon^2 u'' + u = 1$, $\varepsilon$ = {}".format(EPSILON))
plt.legend()
plt.grid(True)
plt.show()

plt.plot(i_s,sol.x)
plt.show()