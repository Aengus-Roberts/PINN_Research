import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

EPSILON = .01
B = 1 / (1 - np.exp(-1 / EPSILON))


# Define the neural network model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: epsilon*u"(x) + u'(x) - 1
    residual = EPSILON * u_xx + u_x - 1

    # Compute weighted physics loss if weights are provided
    if weights is not None:
        physics_loss = torch.sum(weights * residual ** 2)  # Weighted sum
    else:
        physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]]))
    u1_pred = model(torch.tensor([[1.0]]))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return physics_loss + bc_loss


def gauss_lobatto_nodes_weights(n):
    # Compute the Gauss-Lobatto nodes and weights on the interval [-1,1] for n nodes.
    if n < 2:
        raise ValueError("n must be at least 2.")

    # Endpoints are fixed
    x = np.zeros(n)
    x[0] = -1.0
    x[-1] = 1.0

    if n > 2:
        # The interior nodes are the roots of the derivative of the (n-1)th Legendre polynomial
        P = Legendre.basis(n - 1)
        dP = P.deriv()
        x[1:-1] = np.sort(dP.roots())

    # Compute the weights using the formula
    w = np.zeros(n)
    for i in range(n):
        # Evaluate the (n-1)th Legendre polynomial at x[i]
        P_val = Legendre.basis(n - 1)(x[i])
        w[i] = 2 / (n * (n - 1) * (P_val ** 2))

    return x, w


# Generate training points using different quadrature methods
def generate_training_points(method='uniform', num_points=10):
    if method == 'uniform':
        x_train = np.linspace(0, 1, num_points)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'gauss_legendre':
        nodes, weights = roots_legendre(num_points)
        x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
        weights = weights * (1 / 2)
    elif method == 'gauss_lobatto':
        nodes, weights = gauss_lobatto_nodes_weights(num_points)
        x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
        weights = weights * (1 / 2)
    else:
        raise ValueError("Unsupported quadrature method")
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights):
    # Training the PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train on the first 10% of data
    first_10_percent = int(len(x_train) * 0.1)
    x_train_initial = x_train[:first_10_percent]
    weights_initial = weights[:first_10_percent]
    for epoch in range(2000):
        loss = compute_loss(model, x_train_initial, weights_initial)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Initial Training Epoch {epoch}, Loss: {loss.item():.6f}")

    # Continue training on the full dataset
    for epoch in range(2000):
        loss = compute_loss(model, x_train, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


# Plotting True Result
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
u1 = lambda x: B * (np.exp(-x / EPSILON) - 1) + x
y_true = np.array([u1(x) for x in x_test])
plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')


# Plot the results
def create_results(quadrature, weights, color, label):
    model = train_PINN(quadrature, weights)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')


# Getting Quadratures
uniform, uniform_weights = generate_training_points()
gauss_10, gauss_10_weights = generate_training_points(method='gauss_legendre')
gauss_11, gauss_11_weights = generate_training_points(method='gauss_legendre', num_points=101)
lobatto_10, lobatto_10_weights = generate_training_points(method='gauss_lobatto')
lobatto_11, lobatto_11_weights = generate_training_points(method='gauss_lobatto', num_points=101)

# Plotting Quadratures
create_results(uniform, uniform_weights, 'red', 'PINN: Uniform')
create_results(gauss_10, gauss_10_weights, 'blue', 'PINN: Gauss_10')
create_results(gauss_11, gauss_11_weights, 'orange', 'PINN: Gauss_11')
create_results(lobatto_10, lobatto_10_weights, 'black', 'PINN: Lobatto_10')
create_results(lobatto_11, lobatto_11_weights, 'pink', 'PINN: Lobatto_11')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
title = r"$ε u''(x) + u'(x) = 1$, ε = {:.5f}".format(EPSILON)
plt.title(title)
plt.show()
