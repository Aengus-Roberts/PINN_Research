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
        self.knot_points = nn.Parameter(torch.tensor(knot_points, dtype=torch.float32))

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K,) → reshape to (1, K) for broadcasting
        relus = torch.relu(x - self.knot_points.view(1, -1))  # (N, K)
        coeffs = self.coeffs
        output = torch.matmul(relus, coeffs)
        return output

weights,knots = np.load('FKSmodelParams/0.01/Gaussian.npy')

print(weights)
print(knots)

N = len(weights)

A = np.ndarray(shape=(N,N))

for i in range(1,N-1):
