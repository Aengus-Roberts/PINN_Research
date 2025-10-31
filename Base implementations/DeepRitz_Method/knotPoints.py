import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

#y = sum_{k_i} (\alpha_i ReLU(x-k_i))

EPSILON = 0.1

class ReLU3(nn.Module):
    def __init__(self):
        super(ReLU3, self).__init__()

    def forward(self, x):
        return torch.where(x > 0, x**3, torch.zeros_like(x))

# Knot-point ReLU architecture
class KnotReLU(nn.Module):
    def __init__(self, knot_points):
        super(KnotReLU, self).__init__()
        self.knot_points = nn.Parameter(torch.tensor(knot_points, dtype=torch.float32), requires_grad=False)
        self.alpha = nn.Parameter(torch.randn(len(knot_points), dtype=torch.float32))

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K,) → reshape to (1, K) for broadcasting
        relus = torch.relu(x - self.knot_points.view(1, -1))  # (N, K)
        return torch.matmul(relus, self.alpha)  # (N,)


# Energy loss computation using quadrature points
def compute_energy_loss(model, x_quad, w_quad, epsilon):
    """
    Compute energy functional:
        E(y) = ∑_i w_i [ (ε²/2) * (y')² + (1/2) * y² - y ]
    using quadrature points x_quad and weights w_quad.
    """
    x_quad.requires_grad_(True)
    y = model(x_quad).view(-1,1)

    # First derivative y'
    dy = torch.autograd.grad(y, x_quad, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    # Energy integrand
    integrand = (epsilon**2 / 2) * dy**2 + (1/2) * y**2 - y

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x_quad.device))
    u1_pred = model(torch.tensor([[1.0]], device=x_quad.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    # Weighted quadrature sum
    return torch.sum(w_quad * integrand) + bc_loss

def get_knot_points(N=50):
    start_knot_points = np.linspace(0, EPSILON, N)
    mid_knot_points = np.linspace(EPSILON, 1-EPSILON, N)
    end_knot_points = np.linspace(1-EPSILON, 1, N)
    knot_points = np.concatenate((start_knot_points, mid_knot_points, end_knot_points))

    return knot_points

# Quadrature points and weights (e.g., Gauss-Legendre)
def get_quad_points(N=50,type='uniform'):
        if type == 'uniform':
            x_quad = torch.linspace(0, 1, 3*N).unsqueeze(1)
            w_quad = torch.tensor([1/N for i in range(3*N)], dtype=torch.float32).unsqueeze(1)
        elif type == 'gauss':
            x_quad_np, w_quad_np = roots_legendre(N)
            x_quad = torch.tensor((x_quad_np + 1) / 2, dtype=torch.float32).unsqueeze(1)  # map from [-1,1] to [0,1]
            w_quad = torch.tensor(w_quad_np / 2, dtype=torch.float32).unsqueeze(1)        # adjust weights accordingly
        else:
            raise ValueError("Unsupported quadrature method")

        return x_quad, w_quad

def train_model(x_quad, w_quad):
    knot_points = get_knot_points()
    model = KnotReLU(knot_points)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(10000):
        optimizer.zero_grad()
        loss = compute_energy_loss(model, x_quad, w_quad, EPSILON)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

# Plot the results
def create_results(x_test, x_quad, w_quad, color='red', label=''):
    model = train_model(x_quad,w_quad)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')

def main():
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u2 = lambda x: 1 - np.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    x_uniform, w_uniform = get_quad_points(type='uniform')
    x_gauss, w_gauss = get_quad_points(type='gauss')
    create_results(x_test, x_uniform, w_uniform, color='red',label='Uniform')
    create_results(x_test, x_gauss, w_gauss, color='blue',label='Gaussian')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()