import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.special import roots_legendre

INNER_EPOCHS = 100
OUTER_EPOCHS = 10
KNOT_NUMBER = 100
QUAD_NUMBER = 500
TIMESTEPS = 100
T = 1
deltaT = T / TIMESTEPS


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


def get_knot_points(distribution, N=KNOT_NUMBER):
    if distribution == "uniform":
        knot_points = torch.linspace(0, 1, N, dtype=torch.float32)

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
    else:
        raise ValueError("Unsupported quadrature method")

    return x_quad, w_quad  # returns 2 tensors, length N


def compute_energy_loss(model0, model1, x, w):
    x.requires_grad = True
    u = model1(x).view(-1, 1)
    u_0 = model0(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = (deltaT ** 2 / 2) * du ** 2 + (1 / 2) * u ** 2 - u * u_0

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model1(torch.tensor([[0.0]], device=x.device))
    u1_pred = model1(torch.tensor([[1.0]], device=x.device))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    return torch.sum(w * integrand) + bc_loss


def evaluate_equidistribution(model0, model1, method=0):
    RESOLUTION = 100
    # Sampling domain according to previous Knot Point Distribution
    X = []
    segments = []
    knots = model1.knot_points
    for i in range(len(knots) - 1):
        seg = torch.linspace(knots[i], knots[i + 1], RESOLUTION + 1, dtype=torch.float32)[:-1]
        segments.append(seg)

    X = torch.cat(segments).view(-1)
    X.requires_grad = True
    u_0 = model0(X)
    u = model1(X)
    if method == 0:
        d2u = (u - u_0) / deltaT
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


def get_updated_knots(u_0, model):
    cumulative_integral, X = evaluate_equidistribution(u_0, model)
    new_knots = search_array(cumulative_integral, X, len(model.knot_points))
    return new_knots


def train_model(x, w, u_0):
    knot_points = get_knot_points('uniform')
    model = FKS(knot_points)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = optim.LBFGS([model.coeffs], lr=0.01, max_iter=INNER_EPOCHS)

        def DRM_closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(u_0, model, x, w)
            loss.backward()
            return loss

        if parameter == 0:
            optimiser.step(DRM_closure)
        return model

    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        print("Outer Epoch: ", outer_epoch)
        model = trainParam(0)
        new_knot_points = get_updated_knots(u_0, model).detach()
        model.set_knot_points(new_knot_points)

    return model


def main():
    x_uniform, w_uniform = get_quad_points(type='uniform')
    u_0 = lambda x: torch.sin(torch.pi*x)
    timestepped_u = [u_0]

    for t in range(TIMESTEPS):
        print("Timestep: ", t)
        timestepped_u.append(train_model(x_uniform, w_uniform, timestepped_u[-1]))


    x = torch.linspace(0,1,10)
    y = torch.linspace(0,1,TIMESTEPS+1)
    z_list = [timestepped_u[0](x).detach().numpy()]
    for i in range(1, TIMESTEPS+1):
        z_list.append(timestepped_u[i](x.unsqueeze(1)).detach().numpy())
    Z = np.array(z_list)
    X, Y = np.meshgrid(x.detach().numpy(), y.detach().numpy())

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

main()
