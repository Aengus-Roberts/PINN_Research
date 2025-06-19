import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.ma.extras import ndenumerate
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

EPSILON = 1e-3

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

    return physics_loss + 10*bc_loss

def train_PINN(x_train, weights,epsilon=EPSILON):
    # Training the PINN
    model = PINN()
    optimiser = optim.LBFGS(model.parameters(), lr=0.01)

    def closure():
        optimiser.zero_grad()
        loss = compute_loss(model, x_train, weights, epsilon)  # Correct call
        loss.backward()
        return loss

    # Continue training on the full dataset
    for epoch in range(4000):
        optimiser.step(closure)

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}")

    return model

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
    elif method == 'thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(0, 2*EPSILON, third_N)
        third_x_train = np.linspace(1-(2*EPSILON), 1, third_N)
        middle_x_train = np.linspace(2*EPSILON,1-(2*EPSILON),num_points-(2*third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
    elif method == 'outside_thirds':
        third_N = int(np.ceil(num_points / 3))
        first_x_train = np.linspace(-EPSILON, 2*EPSILON, third_N)
        third_x_train = np.linspace(1-(2*EPSILON), 1 + EPSILON, third_N)
        middle_x_train = np.linspace(2*EPSILON,1-(2*EPSILON),num_points-(2*third_N))
        x_train = np.concatenate((first_x_train, middle_x_train, third_x_train))
        weights = np.ones_like(x_train) / num_points  # Equal weights
    else:
        raise ValueError("Unsupported quadrature method")
    #np.random.shuffle(x_train)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)

num_points = 50
num_test_points = 10000
x_test = torch.linspace(0, 1, num_test_points).reshape(-1, 1)

def get_assymetry(model):
    forwards = model(x_test).detach().numpy()
    backwards = forwards[::-1]

    # calculating l2 norm of difference between forward and backward pass of function
    return (np.mean((forwards - backwards) ** 2)) ** 0.5


if __name__ == '__main__':
    # Getting Collocation and Weights
    uniform, uniform_weights = generate_training_points(num_points=num_points)
    gauss_100, gauss_100_weights = generate_training_points(method='gauss_legendre', num_points=num_points)
    gauss_101, gauss_101_weights = generate_training_points(method='gauss_legendre', num_points=101)
    #lobatto_100, lobatto_100_weights = generate_training_points(method='gauss_lobatto', num_points=num_points)
    #lobatto_101, lobatto_101_weights = generate_training_points(method='gauss_lobatto', num_points=101)

    one_to_ten = [i for i in range(1, 10)]
    ten_to_one = one_to_ten[::-1]

    epsilon_list = []

    for j in range(1, 4):
        for i in ten_to_one:
            epsilon_list.append(i * (10 ** -j))

    epsilon_array = np.array(epsilon_list)
    uniform_assymetry_array = np.zeros_like(epsilon_array)
    gauss_100_assymetry_array = np.zeros_like(epsilon_array)
    # gauss_101_assymetry_array = np.zeros_like(epsilon_array)
    #lobatto_100_assymetry_array = np.zeros_like(epsilon_array)
    # lobatto_101_assymetry_array = np.zeros_like(epsilon_array)

    for i, epsilon in ndenumerate(epsilon_array):
        print("Epsilon: ", epsilon)
        # training on uniform collocation
        print("Uniform")
        model = train_PINN(uniform, uniform_weights,epsilon)
        uniform_assymetry_array[i] = get_assymetry(model)
        # training on gauss collocation (100 points)
        print("Gauss")
        model = train_PINN(gauss_100, gauss_100_weights,epsilon)
        gauss_100_assymetry_array[i] = get_assymetry(model)
        # training on lobatto collocation (100 points)
        #print("Lobatto")
        #model = train_PINN(lobatto_100, lobatto_100_weights,epsilon)
        #lobatto_100_assymetry_array[i] = get_assymetry(model)

    filtered_uniform = []
    filtered_epsilon_uniform = []
    filtered_gauss = []
    filtered_epsilon_gauss = []
    for i, var in enumerate(uniform_assymetry_array):
        if var < 1:
            filtered_uniform.append(var)
            filtered_epsilon_uniform.append(epsilon_array[i])
    for i, var in enumerate(gauss_100_assymetry_array):
        if var < 1:
            filtered_gauss.append(var)
            filtered_epsilon_gauss.append(epsilon_array[i])

    plt.scatter(filtered_epsilon_uniform, filtered_uniform, color='red', label='Uniform')
    plt.scatter(filtered_epsilon_gauss, filtered_gauss, color='blue', label='Gauss')
    #plt.scatter(epsilon_array, lobatto_100_assymetry_array, color='green', label='Lobatto')
    print(filtered_epsilon_uniform)
    print(filtered_epsilon_gauss)
    print(filtered_uniform)
    print(filtered_gauss)
    plt.xscale('log')
    plt.legend()
    plt.title(r'Measure of asymmetry in PINN solution $u_\theta$ as a function of $\varepsilon$')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r"$||u_\theta(x) - u_\theta(1-x)||_{L^2}$")
    plt.show()
