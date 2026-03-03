import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import os

EPSILON = 0.1
INNER_EPOCHS = 500
OUTER_EPOCHS = 50
KNOT_NUMBER = 50
QUAD_NUMBER = 200


class FKS(nn.Module):
    def __init__(self, knot_points):
        super(FKS, self).__init__()
        self.coeffs = nn.Parameter(torch.randn(len(knot_points) - 1, dtype=torch.float32))
        self.knot_points = knot_points

    def set_knot_points(self, knot_points):
        self.knot_points = knot_points

    @property
    def ki(self):
        return self.knot_points[1:-1]

    @property
    def kminus(self):
        return self.knot_points[:-2]

    @property
    def kplus(self):
        return self.knot_points[2:]

    def interior_spline(self, x):
        alpha = 1 / (self.ki - self.kminus)
        beta = (self.kplus - self.kminus) / ((self.kplus - self.ki) * (self.ki - self.kminus))
        gamma = 1 / (self.kplus - self.ki)
        xT = x.reshape(-1, 1)

        output = torch.relu(xT - self.kminus) * alpha - torch.relu(xT - self.ki) * beta + torch.relu(
            xT - self.kplus) * gamma
        return output

    def left_spline(self, x):
        k0 = self.knot_points[0]
        k1 = self.knot_points[1]
        return torch.relu(k1 - x) / (k1 - k0)

    def right_spline(self, x):
        kminus = self.knot_points[-2]
        kfinal = self.knot_points[-1]
        return torch.relu(x - kminus) / (kfinal - kminus)

    def forward(self, x):
        # x: shape (N, 1), knot_points: shape (K-1 , ) → reshape to (1, K - 1) for broadcasting
        # FKS: shape (N,K - 1)
        FKS = torch.zeros(len(x), len(self.knot_points) - 1, dtype=torch.float32, device=x.device)
        # FKS[:, 0] = self.left_spline(x).squeeze()  # first column
        FKS[:, -1] = self.right_spline(x).squeeze()  # last column
        FKS[:, :-1] = self.interior_spline(x)
        coeffs = self.coeffs
        output = torch.matmul(FKS, coeffs)
        return output


def compute_energy_loss(model, x, w, epsilon):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = torch.exp(-x/epsilon) * ((epsilon/2)*du**2 - u)


    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return torch.sum(w * integrand) + bc_loss


def compute_equidistribution_loss(model):
    knots = model.knot_points
    half_knots = (knots[0:-1] + knots[1:]) / 2
    half_knots = half_knots.view(-1, 1)
    # half_knots.requires_grad = True
    knot_differences = torch.abs(knots[1:] - knots[:-1])
    u = model(half_knots).view(-1, 1)
    du = torch.autograd.grad(u, half_knots, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    d2u = torch.autograd.grad(du, half_knots, grad_outputs=torch.ones_like(du), create_graph=True)[0]

    u2 = lambda x: 1 - torch.cosh((x - 0.5) / EPSILON) / torch.cosh(torch.tensor(1 / (2 * EPSILON)))
    new_u = u2(half_knots)
    return torch.sum(((new_u - 1) ** 2) * knot_differences ** 5) / EPSILON ** 4



def get_knot_points(distribution, N=KNOT_NUMBER):
    if distribution == "uniform":
        knot_points = torch.linspace(0, 1, N, dtype=torch.float32)

    elif distribution == "thirds":
        N_b = int(np.floor(N / 3))
        N_i = N - 2 * N_b
        start_knot_points = torch.linspace(0, EPSILON, N_b + 1)[:-1]
        mid_knot_points = torch.linspace(EPSILON, 1 - EPSILON, N_i)
        end_knot_points = torch.linspace(1 - EPSILON, 1, N_b + 1)[1:]
        knot_points = torch.cat([start_knot_points, mid_knot_points, end_knot_points])

    return knot_points  # Returns np.array length K


# Quadrature points and weights (e.g., Gauss-Legendre)
def get_quad_points(N=QUAD_NUMBER, type='uniform'):
    if type == 'uniform':
        x_quad = torch.linspace(0, 1, N).unsqueeze(1)
        w_quad = torch.tensor([1 / N for _ in range(N)], dtype=torch.float32).unsqueeze(1)
    elif type == 'gauss':
        x_quad_np, w_quad_np = roots_legendre(N)
        x_quad = torch.tensor((x_quad_np + 1) / 2, dtype=torch.float32).unsqueeze(1)  # map from [-1,1] to [0,1]
        w_quad = torch.tensor(w_quad_np / 2, dtype=torch.float32).unsqueeze(1)  # adjust weights accordingly
    elif type == 'thirds':
        N_b = int(np.floor(N / 3))
        N_i = N - 2 * N_b
        start = torch.linspace(0, EPSILON, N_b + 2)[1:-1]
        mid = torch.linspace(EPSILON, 1 - EPSILON, N_i)
        end = torch.linspace(1 - EPSILON, 1, N_b + 2)[1:-1]
        x = torch.cat((start, mid, end))
        x_quad = x.unsqueeze(1)
        # Trapezoidal Weightings
        w = torch.zeros_like(x)
        h = x[1:] - x[:-1]
        w[0] = 0.5 * h[0]
        w[1:-1] = 0.5 * (h[1:] + h[:-1])
        w[-1] = 0.5 * h[-1]
        w_quad = w.unsqueeze(1)


    else:
        raise ValueError("Unsupported quadrature method")

    return x_quad, w_quad  # returns 2 tensors, length N

def evaluate_equidistribution(model, method = 0):
    RESOLUTION = 100
    #Sampling domain according to previous Knot Point Distribution
    X = []
    segments = []
    knots = model.knot_points
    for i in range(len(knots) - 1):
        seg = torch.linspace(knots[i], knots[i + 1], RESOLUTION + 1, dtype=torch.float32)[:-1]
        segments.append(seg)

    X = torch.cat(segments).view(-1)
    X.requires_grad = True
    u = model(X)
    du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    if method == 0:
        d2u = (du - 1)/EPSILON #Cheating trick by rearranging -eps^2 u" + u = 1 -> u" = (u-1)/eps^2
    elif method == 1:
        du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),create_graph=True, retain_graph=True)[0]
        d2u = torch.autograd.grad(du, X, grad_outputs=torch.ones_like(du),create_graph=True)[0]

    #Determining Monitor Function u"^(2/5)
    Monitor = d2u.abs().pow(2/5).view(-1) + 1e-10

    dx = X[1:] - X[:-1]
    trap = 0.5 * (Monitor[1:] + Monitor[:-1]) * dx
    G = torch.zeros_like(Monitor)
    G[1:] = torch.cumsum(trap, dim=0)
    G_Normalised = G / G[-1]
    return G_Normalised, X

def search_array(G,X,N):
    new_knots = np.empty(N)
    uniform_dist = np.linspace(0, 1, N)
    uniform_marker = 0
    G_marker = 0
    while uniform_marker < N:
        while G[G_marker] < uniform_dist[uniform_marker]:
            G_marker += 1
        # At this point G marker points to the G that is one more than where the knot should be
        new_knots[uniform_marker] = X[G_marker]
        uniform_marker += 1
    return torch.tensor(new_knots)


def get_updated_knots(model):
    cumulative_integral, X = evaluate_equidistribution(model)
    new_knots = search_array(cumulative_integral, X, len(model.knot_points))
    return new_knots


def train_model(x, w):
    knot_points = get_knot_points('uniform')
    model = FKS(knot_points)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = optim.LBFGS([model.coeffs], lr=0.01, max_iter=INNER_EPOCHS)

        def DRM_closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(model, x, w, EPSILON)
            loss.backward()
            return loss

        if parameter == 0:
            optimiser.step(DRM_closure)
        return model


    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        print("Outer Epoch: ", outer_epoch)
        model = trainParam(0)
        new_knot_points = get_updated_knots(model).detach()
        model.set_knot_points(new_knot_points)

    return model


def train_adam(x, w):
    knot_points = get_knot_points('uniform')
    model = FKS(knot_points)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = optim.Adam([model.coeffs], lr=0.01)
        else:
            optimiser = optim.Adam([model.interior_knot_points], lr=0.01)

        for inner_epoch in range(INNER_EPOCHS):
            if parameter == 0:
                loss = compute_energy_loss(model, x, w, EPSILON)
            else:
                loss = compute_equidistribution_loss(model)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if inner_epoch % 500 == 0:
                print(f"Inner Training Epoch {inner_epoch}, Loss: {loss.item():.6f}")

        return model

    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        print("Outer Epoch: ", outer_epoch)
        model = trainParam(0)
        model = trainParam(1)
    return model

# Plot the results
def create_results(x_test, x, w, color='red', label=''):
    model = train_model(x, w)
    y_pred = model(x_test).detach().numpy()
    #plt.plot(x_test.numpy(), np.pow(np.abs((y_pred - 1) / EPSILON**2),2/5), color=color)
    #zeros = np.zeros_like(model.knot_points.detach().numpy())
    #plt.scatter(model.knot_points.detach().numpy(), zeros, color=color)
    #plt.title("meow")
    #plt.show()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')
    zeros = np.zeros_like(model.knot_points.detach().numpy())
    print(model.knot_points.detach().numpy())
    plt.scatter(model.knot_points.detach().numpy(), zeros, color=color)

    directory = "FKSmodelParams/" + str(EPSILON)
    os.makedirs(directory, exist_ok=True)
    filename = directory + "/" + label + ".npz"
    print(len(model.coeffs))
    print(len(model.knot_points))
    with open(filename, 'wb') as file:
        np.savez(file,
                 coeffs=model.coeffs.detach().numpy(),
                 knots=model.knot_points.detach().numpy())


def main():
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    A = np.exp(1/EPSILON)/(1-np.exp(1/EPSILON))
    u2 = lambda x: x + A*(1-np.exp(-1/EPSILON))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    x_uniform, w_uniform = get_quad_points(type='uniform')
    x_gauss, w_gauss = get_quad_points(type='gauss')
    x_thirds, w_thirds = get_quad_points(type='thirds')
    #create_results(x_test, x_uniform, w_uniform, color='red', label='Uniform')
    create_results(x_test, x_gauss, w_gauss, color='blue', label='Gaussian')
    #create_results(x_test, x_thirds, w_thirds, color='orange', label='Thirds')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"Linear Spline, Free Knots, DRM Energy, ε = {:.2f}".format(EPSILON)
    # title = r"Fixed Knots, Fixed Endpoints: $-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()
