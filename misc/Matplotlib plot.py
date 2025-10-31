import numpy as np
import matplotlib.pyplot as plt

epsilon_list = [1,0.5,0.1,0.05,0.01,0.005]
colours = ['red', 'blue', 'green', 'magenta', 'cyan','black',]

x_test = np.linspace(0, 1, 1000).reshape(-1, 1)
for i,epsilon in enumerate(epsilon_list):
    u2 = lambda x: 1 - np.cosh((x - 0.5) / epsilon) / np.cosh(1 / (2 * epsilon))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test, y_true, label=r"$ε = {:.3f}$".format(epsilon_list[i]), color=colours[i])

plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend(loc=(0.4,0.4))
plt.show()