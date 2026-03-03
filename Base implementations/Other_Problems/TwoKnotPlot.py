import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

eps = 0.01

def f(s):
    return s**5 * np.exp(-s/eps)

N = 400  # increase if eps small
x1 = np.linspace(0, 1, N)
x2 = np.linspace(0, 1, N)
X1, X2 = np.meshgrid(x1, x2)

S1 = X1
S2 = X2 - X1
S3 = 1 - X2

# domain where f arguments are nonnegative
mask2 = S2 >= 0
mask3 = S3 >= 0

Z1 = f(S1)
Z2 = np.where(mask2, f(S2), np.nan)
Z3 = np.where(mask3, f(S3), np.nan)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z1, alpha=0.5, linewidth=0)
ax.plot_surface(X1, X2, Z2, alpha=0.5, linewidth=0)
ax.plot_surface(X1, X2, Z3, alpha=0.5, linewidth=0)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")
ax.set_title("z=f(x1), z=f(x2-x1), z=f(1-x2)")
plt.show()