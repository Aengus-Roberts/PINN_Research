# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

# global constants
EPSILON = .01
a = 75
om = 2


# PROBLEM:
# -d^2u / dx^2 = u
# u(0)  = 0
# u'(0) = 1

# define PINN neural network using nn.Module class

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 75),
            nn.Tanh(),
            nn.Linear(75, 75),
            nn.Tanh(),
            nn.Linear(75, 1)
        )

    def forward(self, x):
        return self.net(x)


# neural network functions

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]


def forcing_function(x):
    return 2 * a * a * torch.tanh(a * x) / (torch.cosh(a * x)) ** 2


def compute_loss(model: nn.Module, x_interior: torch.Tensor):
    # evaluating u(x), f(x), du/dx, d^2u/dx^2 at interior collocation points
    x_interior.requires_grad_(True)
    u = model(x_interior)
    du = grad(u, x_interior)
    d2u = grad(du, x_interior)
    f = forcing_function(x_interior)

    # evaluating u(x), du/dx at boundary points
    x_boundary = torch.tensor([[0.]], requires_grad=True)
    u_bc = model(x_boundary)
    du_bc = grad(u_bc, x_boundary)

    #Determining Residuals for Loss Function:
    residual = -d2u - (om * om) * u  # ODE residual: -u"(x) - om^2*u
    dirichlet_residual = u_bc - 0. # Dirchtlet residual: u(0) - 0
    neumann_residual = du_bc - 1. # Neumann residual: u'(0) - 1

    #Determining Losses
    interior_loss = torch.mean(residual ** 2)
    ivp_loss = torch.mean(dirichlet_residual ** 2 + neumann_residual ** 2)

    return interior_loss + ivp_loss


# Generating collocation points

def generate_collocation_points(method='uniform', num_points=200):
    if method == 'uniform':
        x_train = np.linspace(0, 1, num_points)
    elif method == 'gauss_legendre':
        x_train = (roots_legendre(num_points)[0] + 1)/2
    else:
        raise ValueError("Unsupported quadrature method")

    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32)


def train_PINN(x_train):
    # Training the PINN

    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Iterate training over epochs

    for epoch in range(4000):
        loss = compute_loss(model, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


# Helper function to plot results

def create_results(x_train, x_test, color='red', label=''):
    model = train_PINN(x_train)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.detach().numpy(), y_pred, label=label, color=color, linestyle='--')


if __name__ == "__main__":
    # Plotting True Result

    x_test = torch.linspace(0, 10, 500).reshape(-1, 1)
    y_true = torch.sin(om * x_test).numpy() / om
    plt.plot(x_test.numpy(), y_true, label=r'True Solution: $u^*(x) = \sin(om*x) / om$', color='green')

    # Getting Collocation Points and weights

    uniform = generate_collocation_points(method='uniform')
    gauss_legendre = generate_collocation_points(method='gauss_legendre')

    # Plotting results

    create_results(uniform,x_test, color='blue', label='Uniform')
    create_results(gauss_legendre,x_test, color='green', label='Gauss-Legendre')

    # Plotting Prettiness

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-u''(x) = om^2 \times u(x)$"
    plt.title(title)
    plt.show()
