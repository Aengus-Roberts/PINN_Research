import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy.ma.extras import ndenumerate
from scipy.special import roots_legendre


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
def compute_loss(model, x,loss_weight, weights=None, epsilon=0.1):
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

    return physics_loss + loss_weight*bc_loss


def generate_training_points(num_points=1000):
    nodes, weights = roots_legendre(num_points)
    x_train = (nodes + 1) * (1 / 2)  # Scale to [0,1]
    weights = weights * (1 / 2)
    return torch.tensor(x_train.reshape(-1, 1), dtype=torch.float32), torch.tensor(weights.reshape(-1, 1),
                                                                                   dtype=torch.float32)


def train_PINN(x_train, weights, epsilon, loss_weight):
    # Training the PINN
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(4000):
        loss = compute_loss(model, x_train,loss_weight, weights, epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Full Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


def get_assymetry(model, x_test):
    forwards = model(x_test).detach().numpy()
    backwards = forwards[::-1]

    # calculating l2 norm of difference between forward and backward pass of function
    return (np.mean((forwards - backwards) ** 2)) ** 0.5


def get_bifurcation(epsilon_array, loss_weight, x_test):
    points, weights = generate_training_points()
    assymetry_array = np.zeros_like(epsilon_array)
    for i, epsilon in ndenumerate(epsilon_array):
        print("Epsilon: ", epsilon)
        model = train_PINN(points, weights, epsilon, loss_weight)
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
    loss_weighting = [10,1,0.5,0.1,0.05,0.01]
    colours = ['r','b','g','c','k','m']
    for i,loss_weight in enumerate (loss_weighting):
        assymetry_array = get_bifurcation(epsilon_array, loss_weight, x_test)
        plt.plot(epsilon_array,assymetry_array,color=colours[i],label=f"$loss weight={loss_weight}$")

    plt.legend()
    plt.xscale('log')
    plt.xlabel("Epsilon")
    plt.ylabel("Assymetry")
    plt.title("'Bifurcation' curve per learning rate")
    plt.show()

