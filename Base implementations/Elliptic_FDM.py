import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.0000001
N = 51
h = EPSILON * 0.01

D = np.ndarray(shape=(N+1,N+1), dtype=np.float64)
f = np.ones(shape=N+1, dtype=np.float64)
f *= h**2
f[0] = 0
f[N] = 0

#forming D
for i in range(0,N+1):
    if i ==0:
        D[i,i] = 1
    elif i == 1:
        D[i,i] = 2*(EPSILON**2) + h**2
        D[i,i+1] = -(EPSILON**2)
    elif i == N-1:
        D[i,i] = 2*(EPSILON**2) + h**2
        D[i,i-1] = -(EPSILON**2)
    elif i == N:
        D[i,i] = 1
    else:
        D[i,i] = 2*(EPSILON**2) + h**2
        D[i,i-1] = -(EPSILON**2)
        D[i,i+1] = -(EPSILON**2)

u = np.linalg.solve(D, f)
x = np.linspace(0,1,N+1)

#plotting soln
plt.plot(x,u)
plt.show()
