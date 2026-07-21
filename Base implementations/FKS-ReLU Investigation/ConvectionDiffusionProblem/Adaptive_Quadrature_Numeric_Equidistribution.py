import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import os

EPSILON = 0.1
INNER_EPOCHS = 100
OUTER_EPOCHS = 100
KNOT_NUMBER = 20


class FKS(nn.Module):
    def __init__(self, knot_points):
        super(FKS, self).__init__()
        self.coeffs = nn.Parameter(torch.ones(len(knot_points) - 1, dtype=torch.float32))
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
        # x: shape (N, 1)
        # FKS: shape (N,K - 1)
        FKS = torch.zeros(len(x), len(self.knot_points) - 1, dtype=torch.float32, device=x.device)
        # FKS[:, 0] = self.left_spline(x).squeeze()  # first column
        # FKS[:, -1] = self.right_spline(x).squeeze()  # last column
        FKS[:, :-1] = self.interior_spline(x)
        coeffs = self.coeffs
        output = torch.matmul(FKS, coeffs)
        return output


def compute_energy_loss(model, x, w, epsilon, alpha):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = torch.exp(-x / alpha*epsilon) * ((epsilon / 2) * du ** 2 - u)

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x.device))
    u1_pred = model(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return torch.sum(w * integrand) + bc_loss


def compute_conditioning_number(model, epsilon=EPSILON):
    knots = model.knot_points.to(dtype=model.coeffs.dtype, device=model.coeffs.device)
    h = knots[1:] - knots[:-1]

    if torch.any(h <= 0):
        raise ValueError("Knot points must be strictly increasing.")

    # Integral of exp(-x / epsilon) over each element [x_i, x_{i+1}]
    weighted_lengths = epsilon * (
            torch.exp(-knots[:-1] / epsilon) - torch.exp(-knots[1:] / epsilon)
    )

    # Element stiffness coefficients:
    # epsilon * int exp(-x / epsilon) dx / h_i^2
    element_values = epsilon * weighted_lengths / h.pow(2)

    # Interior hat functions only: one basis at each interior knot.
    n_basis = len(knots) - 2
    H = torch.zeros((n_basis, n_basis), dtype=model.coeffs.dtype, device=model.coeffs.device)

    for i in range(n_basis):
        # basis i is centred at knot i + 1
        H[i, i] = element_values[i] + element_values[i + 1]

        if i < n_basis - 1:
            H[i, i + 1] = -element_values[i + 1]
            H[i + 1, i] = -element_values[i + 1]

    eigenvals = torch.linalg.eigvalsh(H)
    lambda_min = torch.min(eigenvals)
    lambda_max = torch.max(eigenvals)
    condition_number = lambda_max / lambda_min
    return condition_number


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


def evaluate_equidistribution(model, method=0):
    RESOLUTION = 100
    # Sampling domain according to previous Knot Point Distribution
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
        d2u = (du - 1) / EPSILON  # Cheating trick by rearranging -eps^2 u" + u = 1 -> u" = (u-1)/eps^2
    elif method == 1:
        d2u = torch.autograd.grad(du, X, grad_outputs=torch.ones_like(du), create_graph=True)[0]

    # Determining Monitor Function u"^(2/5)
    Monitor = d2u.abs().pow(2 / 5).view(-1) + 1e-10

    dx = X[1:] - X[:-1]
    trap = 0.5 * (Monitor[1:] + Monitor[:-1]) * dx
    G = torch.zeros_like(Monitor)
    G[1:] = torch.cumsum(trap, dim=0)
    G_Normalised = G / G[-1]
    return G_Normalised, X


def search_array(G, X, N):
    new_knots = np.empty(N)
    uniform_dist = np.linspace(0, 1, N)
    uniform_marker = 0
    G_marker = 0
    while uniform_marker < N:
        while G[G_marker] < uniform_dist[uniform_marker]:
            G_marker += 1
        # At this point G marker points to the G that is one more than where the knot should be
        if X[G_marker] == X[-1]:
            new_knots[uniform_marker] = 1.0
        else:
            new_knots[uniform_marker] = (X[G_marker] + X[G_marker + 1]) / 2
        uniform_marker += 1
    new_knots[0] = 0.0
    return torch.tensor(new_knots)


def get_updated_knots(model):
    cumulative_integral, X = evaluate_equidistribution(model)
    new_knots = search_array(cumulative_integral, X, len(model.knot_points))
    return new_knots


def get_adaptive_quadrature_points(model):
    knots = model.knot_points.detach().numpy()
    base_gauss_points, base_gauss_weights = roots_legendre(2)

    # Element midpoints and half-widths
    mid = 0.5 * (knots[:-1] + knots[1:])
    half = 0.5 * (knots[1:] - knots[:-1])

    # Broadcast to get all quadrature points
    quad_points = mid[:, None] + half[:, None] * base_gauss_points
    quad_weights = half[:, None] * base_gauss_weights

    # Flatten to 1D array
    quad_points = quad_points.ravel()
    quad_weights = quad_weights.ravel()

    return torch.tensor(quad_points).view(-1, 1), torch.tensor(quad_weights).view(-1, 1)


def train_model():
    knot_points = get_knot_points('uniform')
    model = FKS(knot_points)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = optim.LBFGS([model.coeffs], lr=0.01, max_iter=INNER_EPOCHS)
            x_quad, w_quad = get_adaptive_quadrature_points(model)
            # w_quad = torch.ones_like(x_quad)

        def DRM_closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(model, x_quad, w_quad, EPSILON)
            loss.backward()
            return loss

        if parameter == 0:
            optimiser.step(DRM_closure)
        return model

    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        print("Outer Epoch: ", outer_epoch)
        print("Condition Number: " + str(compute_conditioning_number(model).item()))
        new_knot_points = get_updated_knots(model).detach()
        model.set_knot_points(new_knot_points)
        model = trainParam(0)

    return model


# Plot the results
def create_results(x_test, color='red', label=''):
    model = train_model()
    y_pred = model(x_test).detach().numpy()
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
    x_test = torch.linspace(0, 1, 1000).reshape(-1, 1)
    B = 1 / (1 - np.exp(1 / EPSILON))
    u1 = lambda x: x - B * (1 - np.exp(x / EPSILON))
    y_true = np.array([u1(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    create_results(x_test, color='blue', label='Approximation')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = "Linear Spline, Equidistributed Knots \n" r"DRM Energy with Adaptive Quadrature, ε = {:.4f}".format(EPSILON)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    main()
