import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

EPSILON = .001


# Defined PINN via PyTorch Structure, 2 Hidden Layers
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None, EPSILON=EPSILON):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: -epsilon^2u"(x) + u(x) - 1
    residual = -(EPSILON ** 2) * u_xx + u - 1

    # Compute weighted physics loss if weights are provided
    if weights is not None:
        physics_loss = torch.sum(weights * residual ** 2)  # Weighted sum
    else:
        physics_loss = torch.mean(residual ** 2)  # Uniform weight (default)

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
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
        x_train = (nodes + 1)/2
        weights = weights/2
    elif method == 'gauss_lobatto':
        nodes, weights = gauss_lobatto_nodes_weights(num_points)
        x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
        weights = weights * (1 / 2)
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
    else:
        raise ValueError("Unsupported quadrature method")
    # np.random.shuffle(x_train)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights, model = 0, epsilon=EPSILON, homotopy_method = True):
    # Training the PINN
    if epsilon == 100 or homotopy_method == False:
        model = PINN()
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    EPOCHS = 5000

    # Continue training on the full dataset
    for epoch in range(EPOCHS):
        loss = compute_loss(model, x_train, weights, epsilon)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


# Plot the results
def create_results(x_test,quadrature, weights, color='red', label=''):
    epsilon_list = [100,80,60,40,20,10,8,6,4,2,1,0.8,0.6,0.4,0.2,0.1,0.08,0.06,0.02,0.01,0.008,0.006,0.002,0.001]
    for epsilon in epsilon_list:
        print(f"Epsilon: {epsilon}")
        if epsilon == 100:
            model = train_PINN(quadrature, weights, epsilon=epsilon)
        else:
            non_homotopy_model = train_PINN(quadrature, weights, epsilon=epsilon, homotopy_method=False)
            model = train_PINN(quadrature, weights,model = model, epsilon=epsilon)
            plt.plot(x_test.numpy(), model(x_test).detach().numpy(), label='homotopy method')
            plt.plot(x_test.numpy(), non_homotopy_model(x_test).detach().numpy(), label='regular method')
            plotTrue(x_test,epsilon)
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.legend()
            title = r"$-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(epsilon)
            plt.title(title)
            plt.show()
    return model(x_test).detach().numpy()


def plotTrue(x_test, epsilon=EPSILON):

    u2 = lambda x: 1 - np.cosh((x - 0.5) / epsilon) / np.cosh(1 / (2 * epsilon))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')
    return x_test

if __name__ == "__main__":
    # Plotting True Result
    x_test = torch.linspace(0, 1, 1000).reshape(-1, 1)

    # Getting Collocation Points and weights
    uniform, uniform_weights = generate_training_points(num_points=100)
    gauss_10, gauss_10_weights = generate_training_points(method='gauss_legendre', num_points=100)
    thirds, thirds_weights = generate_training_points(method='thirds', num_points=100)

    # Plotting Quadratures
    uniform_result = create_results(x_test,uniform, uniform_weights, 'red', 'PINN: Uniform')
    gauss_result = create_results(x_test,gauss_10, gauss_10_weights, 'blue', 'PINN: Gauss')
    thirds_result = create_results(x_test,thirds, thirds_weights, 'green', 'PINN: Thirds')

    plotTrue(x_test)
    plt.plot(x_test, uniform_result, label='Uniform Solution', color='red')
    plt.plot(x_test, gauss_result, label='Gauss Solution', color='blue')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()
