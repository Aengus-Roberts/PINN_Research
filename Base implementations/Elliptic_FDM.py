import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.00001
N = 51

D = np.ndarray(shape=(N+1,N+1), dtype=np.float64)

powers = [i for i in range(-3,3)]
tens_list = [10**i for i in powers]
h_list = [EPSILON * ten for ten in tens_list]

for k,h in enumerate(h_list):
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
    plt.plot(x,u, label=r'$h = ε\times10^{}$'.format(powers[k]))
plt.title('FDM: -εu" + u = 1, u(0)=u(1)=0, ε = {}'.format(EPSILON))
plt.legend()
plt.show()
