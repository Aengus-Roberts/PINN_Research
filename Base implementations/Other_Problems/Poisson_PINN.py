import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

EPSILON = .01


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

def forcing_function(x):
    #return torch.sin(x)
    return 2*torch.tanh(x)/torch.cosh(x)**2
    #return -8*(torch.exp(x) - torch.exp(-x))/(torch.exp(x)+torch.exp(-x))**3

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None, EPSILON=EPSILON):
    x.requires_grad_(True)
    u = model(x)
    #u_x = grad(u, x)
    #u_xx = grad(u_x, x, u)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    f = forcing_function(x)

    # ODE residual: -u"(x) - f(x)
    residual = -u_xx - f

    # Compute weighted physics loss if weights are provided
    if weights is not None:
        physics_loss = torch.sum(weights * residual ** 2)  # Weighted sum
    else:
        physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    # Boundary condition loss: u(-1) = tanh(-1), u(1) = tanh(1)
    u0_pred = model(torch.tensor([-1.0])) - torch.tanh(torch.tensor([-1.0]))
    u1_pred = model(torch.tensor([1.0])) - torch.tanh(torch.tensor([1.0]))
    bc_loss = torch.sqrt(u0_pred.pow(2) + u1_pred.pow(2))

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
        x_train = np.linspace(-1, 1, num_points)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'gauss_legendre':
        x_train, weights = roots_legendre(num_points)
    elif method == 'gauss_lobatto':
        x_train, weights = gauss_lobatto_nodes_weights(num_points)
    else:
        raise ValueError("Unsupported quadrature method")
    #np.random.shuffle(x_train)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights,epsilon=EPSILON):
    # Training the PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Iterate training over epochs
    for epoch in range(4000):
        loss = compute_loss(model, x_train, weights,epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

# Plot the results
def create_results(x_test, weights, color='red', label=''):
    model = train_PINN(x_test, weights)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.detach().numpy(), y_pred, label=label, color=color, linestyle='--')



if __name__ == "__main__":
    # Plotting True Result
    x_test = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y_true = np.array(torch.tanh(x_test))
    plt.plot(x_test.numpy(), y_true, label= r'True Solution: $u^*(x) = \tanh(x)$', color='green')

    # Getting Collocation Points and weights
    uniform, uniform_weights = generate_training_points()
    gauss_10, gauss_10_weights = generate_training_points(method='gauss_legendre')
    lobatto_10, lobatto_10_weights = generate_training_points(method='gauss_lobatto')

    # Plotting Quadratures
    create_results(uniform, uniform_weights, 'red', 'PINN: Uniform')
    create_results(gauss_10, gauss_10_weights, 'blue', 'PINN: Gauss_10')
    #create_results(lobatto_10, lobatto_10_weights, 'black', 'PINN: Lobatto_10')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-u''(x) = f(x)$"
    plt.title(title)
    plt.show()
