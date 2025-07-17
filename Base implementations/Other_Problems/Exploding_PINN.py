import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

GAMMA = 0.99


# Defined PINN via PyTorch Structure, 2 Hidden Layers
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None, gamma=GAMMA):
    x.requires_grad_(True)
    u = model(x)
    u_squared = torch.pow(u, 2)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: -epsilon^2u"(x) + u(x) - 1
    residual = u_x - u_squared

    # Compute weighted physics loss if weights are provided
    if weights is not None:
        physics_loss = torch.sum(weights * residual ** 2)  # Weighted sum
    else:
        physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    #Initial condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device)) - gamma
    ic_loss = u0_pred.pow(2)

    return 100*physics_loss + ic_loss


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
        x_train = (nodes + 1)/2
        weights = weights/2
    elif method == 'gauss_lobatto':
        nodes, weights = gauss_lobatto_nodes_weights(num_points)
        x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
        weights = weights * (1 / 2)
        """
    elif method == 'thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(0, 2 * EPSILON, third_N)
        third_x_train = np.linspace(1 - (2 * EPSILON), 1, third_N)
        middle_x_train = np.linspace(2 * EPSILON, 1 - (2 * EPSILON), num_points - (2 * third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'outside_thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(-EPSILON, 2 * EPSILON, third_N)
        third_x_train = np.linspace(1 - (2 * EPSILON), 1 + EPSILON, third_N)
        middle_x_train = np.linspace(2 * EPSILON, 1 - (2 * EPSILON), num_points - (2 * third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
        """
    elif method == 'sin':
        linear = np.linspace(0, 1, num_points)
        x_train = np.sin(np.pi*linear/2)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == '2n':
        n_list = [n for n in range(0, num_points)]
        t_list = [1 - (1 / 2 ** n) for n in n_list]
        x_train = np.array(t_list)
        weights = np.ones_like(x_train) / num_points  # Equal weights
    else:
        raise ValueError("Unsupported quadrature method")
    # np.random.shuffle(x_train)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights, gamma=GAMMA):
    # Training the PINN
    model = PINN()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    # Continue training on the full dataset
    for epoch in range(20000):
        loss = compute_loss(model, x_train[1:-1], weights[1:-1], gamma)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


# Plot the results
def create_results(quadrature, weights, color='red', label=''):
    model = train_PINN(quadrature, weights)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')


if __name__ == "__main__":
    # Plotting True Result
    gamma = GAMMA
    x_test = torch.linspace(0, 1, 10000).reshape(-1, 1)
    u_true = lambda x: -1/(x-(1/gamma))
    y_true = np.array([u_true(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    # Getting Collocation Points and weights
    uniform, uniform_weights = generate_training_points(num_points=1000)
    gauss_10, gauss_10_weights = generate_training_points(method='gauss_legendre', num_points=1000)
    scaling, scaling_weights = generate_training_points(method='2n', num_points=1000)
    # Plotting Quadratures
    create_results(uniform, uniform_weights, 'red', 'PINN: Uniform')
    create_results(gauss_10, gauss_10_weights, 'blue', 'PINN: Gauss')
    create_results(scaling, scaling_weights, 'black', 'PINN: scaling')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.yscale('log')
    plt.legend()
    title = r"$ \dot u = u^2, \qquad u(0) = \gamma, \gamma = ${:.5f}".format(gamma)
    plt.title(title)
    plt.show()
