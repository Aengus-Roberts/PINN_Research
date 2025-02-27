import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.001

B = 1/(1-np.exp(-1/EPSILON))

x_range = np.arange(0,1,0.01)

u2 = lambda x: (np.cos(1/EPSILON) - 1)/np.sin(1/EPSILON) * np.sin(x/EPSILON) - np.cos(x/EPSILON) + 1

u1 = lambda x: B*(np.exp(-x/EPSILON) - 1) + x

u1Data = np.array([u1(x) for x in x_range])
u2Data = np.array([u2(x) for x in x_range])

plt.plot(x_range,u1Data)
plt.title("U1 Solution")
plt.show()

plt.plot(x_range,u2Data)
plt.title("U2 Solution")
plt.show()
