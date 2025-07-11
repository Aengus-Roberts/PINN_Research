import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.ma.extras import ndenumerate
from scipy.special import roots_legendre


# Define the neural network model
class PINN(nn.Module):
    def __init__(self, l):
        super(PINN, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(1, 20))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(l):
            layers.append(nn.Linear(20, 20))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(20, 1))

        # Convert the list of layers into a sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Compute derivatives using PyTorch autograd
def compute_loss(model, x, weights=None, epsilon=0.1):
    x.requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ODE residual: -epsilon^2u"(x) + u(x) - 1
    residual = -(epsilon ** 2) * u_xx + u - 1

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


def generate_training_points(num_points=1000):
    nodes, weights = roots_legendre(num_points)
    x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
    weights = weights * (1 / 2)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)

def train_PINN(x_train, weights,epsilon, l):
    # Training the PINN
    model = PINN(l)
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


def get_assymetry(model, x_test):
    forwards = model(x_test).detach().numpy()
    backwards = forwards[::-1]

    # calculating l2 norm of difference between forward and backward pass of function
    return (np.mean((forwards - backwards) ** 2)) ** 0.5


def get_bifurcation(epsilon_array, l, x_test):
    points, weights = generate_training_points()
    assymetry_array = np.zeros_like(epsilon_array)
    for i, epsilon in ndenumerate(epsilon_array):
        print("Epsilon: ", epsilon)
        model = train_PINN(points, weights, epsilon, l)
        assymetry_array[i] = get_assymetry(model, x_test)

    return assymetry_array


if __name__ == "__main__":
    #defining test dataset
    num_test_points = 10000
    x_test = torch.linspace(0, 1, num_test_points).reshape(-1, 1)

    #defining array of epsilon values
    one_to_ten = [i for i in range(1, 10)]
    ten_to_one = one_to_ten[::-1]
    epsilon_list = []
    for j in range(1, 4):
        for i in ten_to_one:
            epsilon_list.append(i * (10 ** -j))
    epsilon_array = np.array(epsilon_list)

    #varying over number of training points
    l_list = [1,2,3,4,5]
    colours = ['r','b','g','c','k']
    for i,l in enumerate (l_list):
        assymetry_array = get_bifurcation(epsilon_array, l, x_test)
        plt.plot(epsilon_array,assymetry_array,color=colours[i],label=f"$l={l}$")

    plt.legend()
    plt.xscale('log')
    plt.xlabel("Epsilon")
    plt.ylabel("Assymetry")
    plt.title("'Bifurcation' curve per number of hidden layers")
    plt.show()

