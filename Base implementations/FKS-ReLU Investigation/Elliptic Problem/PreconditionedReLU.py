import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

EPSILON = 0.01

class ReLUKnot(nn.Module):
    def __init__(self, coeffs, knot_points):
        super(ReLUKnot, self).__init__()
        self.coeffs = torch.tensor(coeffs, dtype=torch.float32)
        self.knot_points = torch.tensor(knot_points, dtype=torch.float32)

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K, ) → reshape to (1, K) for broadcasting
        relus = torch.relu(x - self.knot_points.view(1, -1))  # (N, K)
        output = torch.matmul(relus, self.coeffs)
        return output

data = np.load("FKSmodelParams/0.01/Gaussian.npz")
weights = data["coeffs"]
knots  = data["knots"]

"""
#Sorting
sorted_indices = np.argsort(knots)
print(sorted_indices)
weights[:]  = weights[sorted_indices[1:]]
knots[:]  = knots[sorted_indices]
"""

#Calculating Elements of A for c=Aw

#j : [2:-1]
#j-1 : [1:-2]
#j+1 : [3:]

gamma_knot_element = 1/(knots[2:-1]-knots[1:-2]) #1/(kj - kj-1)
beta_knot_element = 1/(knots[1:-2]-knots[2:-1]) + 1/(knots[2:-1] - knots[3:]) #1/(kj-1 - kj) + 1/(kj - kj+1)
alpha_knot_element = 1/(knots[3:]-knots[2:-1]) #1/(kj+1 - kj)

N = len(knots)

A = np.zeros(shape=(N,N-1))

A[0,0] = 1/(knots[1]-knots[0])
A[1,0] = 1/(knots[0]-knots[1]) + 1/(knots[1]-knots[2])
A[1,1] = 1/(knots[2]-knots[1])
A[N-1,N-2] = 1/(knots[-1]-knots[-2])

for j in range(2,N-1):
    idx = j-2
    A[j,j] = alpha_knot_element[idx]
    A[j,j-1] = beta_knot_element[idx]
    A[j,j-2] = gamma_knot_element[idx]

coeffs = A@weights
print(A)
print(coeffs)

model = ReLUKnot(coeffs,knots)


x_test = torch.linspace(0, 1, 500).reshape(-1, 1)
u2 = lambda x: 1 - np.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))
y_true = np.array([u2(x) for x in x_test])
plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

y_pred = model(x_test).detach().numpy()
plt.plot(x_test.numpy(), y_pred, label='Preconditioned ReLU', color='red', linestyle='--')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
title = r"Precontioned ReLU, ε = {:.2f}".format(EPSILON)
plt.title(title)
plt.show()