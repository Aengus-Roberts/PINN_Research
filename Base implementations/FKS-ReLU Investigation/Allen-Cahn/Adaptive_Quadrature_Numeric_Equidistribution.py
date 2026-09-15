import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from scimba_torch.optimizers.ssbroyden import SSBroyden
import os

LAMBDA = 1.0
EPSILON = 0.01
INNER_EPOCHS = 50
OUTER_EPOCHS = 50
KNOT_NUMBER = 50
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

left_bc = (-1.0, -1.0)  # Dirichlet boundary condition at x = -1
right_bc = (1.0, 1.0)  # Dirichlet boundary


class FKS(nn.Module):
    def __init__(self, knot_points):
        super(FKS, self).__init__()
        self.coeffs = nn.Parameter(knot_points)
        self.knot_points = knot_points

    def set_knot_points(self, knot_points):
        self.knot_points = knot_points

    @                              property
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
        FKS = torch.zeros(len(x), len(self.knot_points), dtype=DTYPE, device=x.device)
        FKS[:, 0] = self.left_spline(x).squeeze()  # first column
        FKS[:, -1] = self.right_spline(x).squeeze()  # last column
        FKS[:, 1:-1] = self.interior_spline(x)
        output = torch.matmul(FKS, self.coeffs)
        return output


def compute_energy_loss(model, x, w, epsilon):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = (epsilon**2)/2 * du**2 - LAMBDA*(u**2 / 2 - u**4 / 4)

    # Boundary condition loss: u(0) = u(1) = 0
    xminus = model.coeffs.new_tensor([left_bc[0]], requires_grad=True)
    xplus = model.coeffs.new_tensor([right_bc[0]], requires_grad=True)
    u_minus = model(xminus)
    u_plus = model(xplus)
    du_minus = torch.autograd.grad(u_minus, xminus, grad_outputs=torch.ones_like(u_minus), create_graph=True)[0]
    du_plus = torch.autograd.grad(u_plus, xplus, grad_outputs=torch.ones_like(u_plus), create_graph=True)[0]
    bc_loss = (u_minus - left_bc[1])**2 + (u_plus - right_bc[1])**2


    return torch.sum(w * integrand) + bc_loss


def get_knot_points(distribution, N=KNOT_NUMBER):
    if distribution == "uniform":
        knot_points = torch.linspace(left_bc[0], right_bc[0], N, dtype=DTYPE)

    return knot_points  # Returns np.array length K


def evaluate_equidistribution(model, method=0):
    RESOLUTION = 20
    # Sampling domain according to previous Knot Point Distribution
    segments = []
    knots = model.knot_points
    for i in range(len(knots) - 1):
        seg = torch.linspace(knots[i], knots[i + 1], RESOLUTION + 1, dtype=DTYPE)[:-1]
        segments.append(seg)

    segments.append(knots[-1:].to(dtype=DTYPE))
    X = torch.cat(segments).view(-1)
    X.requires_grad = True
    u = model(X)
    if method == 0:
        d2u = LAMBDA * u * (1-u**2) / (EPSILON**2)
    elif method == 1:
        du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
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
    q = torch.linspace(
        0.0, 1.0, N,
        dtype=X.dtype,
        device=X.device
    )

    new_knots = torch.empty_like(q)

    # Exact domain endpoints
    new_knots[0] = X[0]
    new_knots[-1] = X[-1]

    G_marker = 1

    for i in range(1, N - 1):
        # Find the first j such that G[j] >= q[i]
        while (
            G_marker < len(G) - 1
            and G[G_marker] < q[i]
        ):
            G_marker += 1

        G0 = G[G_marker - 1]
        G1 = G[G_marker]
        X0 = X[G_marker - 1]
        X1 = X[G_marker]

        denominator = (G1 - G0)

        theta = (q[i] - G0) / denominator
        new_knots[i] = X0 + theta * (X1 - X0)

    return new_knots


def get_updated_knots(model):
    cumulative_integral, X = evaluate_equidistribution(model)
    new_knots = search_array(cumulative_integral, X, len(model.knot_points))
    return new_knots


def get_adaptive_quadrature_points(model):
    knots = model.knot_points.detach().numpy()
    base_gauss_points, base_gauss_weights = roots_legendre(3)

    # Element midpoints and half-widths
    mid = 0.5 * (knots[:-1] + knots[1:])
    half = 0.5 * (knots[1:] - knots[:-1])

    # Broadcast to get all quadrature points
    quad_points = mid[:, None] + half[:, None] * base_gauss_points
    quad_weights = half[:, None] * base_gauss_weights

    # Flatten to 1D array
    quad_points = quad_points.ravel()
    quad_weights = quad_weights.ravel()

    return torch.tensor(quad_points).view(-1,1), torch.tensor(quad_weights).view(-1,1)


def train_model():
    knot_points = get_knot_points('uniform')
    model = FKS(knot_points)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = SSBroyden([model.coeffs], lr=0.01, tolerance_grad=1e-10,
        method="ssbroyden")
            x_quad, w_quad = get_adaptive_quadrature_points(model)

        def DRM_closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(model, x_quad, w_quad, EPSILON)
            loss.backward()
            return loss

        if parameter == 0:
            for inner_epoch in range(INNER_EPOCHS):
                optimiser.step(DRM_closure)
        return model

    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        print("Outer Epoch: ", outer_epoch)
        model = trainParam(0)
        new_knots = get_updated_knots(model).detach()
        new_knots = new_knots.to(model.coeffs)

        with torch.no_grad():
            # Evaluate using the old knots
            new_coeffs = model(new_knots).detach()

            # Install the new representation
            model.set_knot_points(new_knots)
            model.coeffs.copy_(new_coeffs)

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
    x_test = torch.linspace(-1, 1, 1000).reshape(-1, 1)
    a = 1/(EPSILON * np.sqrt(2))
    u1 = lambda x: np.tanh(a*x)
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
