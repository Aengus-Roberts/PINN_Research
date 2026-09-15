import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import os

LAMBDA = 1.0
EPSILON = 0.01
INNER_EPOCHS = 25
OUTER_EPOCHS = 50
KNOT_NUMBER = 50
TIMESTEPS = 200
DT = 0.05
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

left_boundary = (0.0,0.0)
right_boundary = (1.0,0.0)



class FKS(nn.Module):
    def __init__(self, knot_points, coeffs):
        super(FKS, self).__init__()
        self.coeffs = nn.Parameter(coeffs.detach().clone())
        self.knot_points = knot_points.detach().clone()

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
        FKS = torch.zeros(len(x), len(self.knot_points), dtype=DTYPE, device=x.device)
        FKS[:, 0] = self.left_spline(x).squeeze()  # first column
        FKS[:, -1] = self.right_spline(x).squeeze()  # last column
        FKS[:, 1:-1] = self.interior_spline(x)
        output = torch.matmul(FKS, self.coeffs)
        return output

def compute_energy_loss(model, x, w, epsilon, u_t):
    x.requires_grad = True
    u = model(x).view(-1, 1)
    u_previous = u_t(x).detach().view(-1, 1)

    du = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]

    integrand = (torch.exp(-x / epsilon) * ((epsilon / 2) * du ** 2 - u) + 0.5 * (u - u_previous) ** 2 / DT)

    # Boundary condition loss: u(0) = u(1) = 0
    xminus = model.coeffs.new_tensor([left_boundary[0]], requires_grad=True)
    xplus = model.coeffs.new_tensor([right_boundary[0]], requires_grad=True)
    u_minus = model(xminus)
    u_plus = model(xplus)
    du_minus = torch.autograd.grad(u_minus, xminus, grad_outputs=torch.ones_like(u_minus), create_graph=True)[0]
    du_plus = torch.autograd.grad(u_plus, xplus, grad_outputs=torch.ones_like(u_plus), create_graph=True)[0]
    bc_loss = (u_minus - left_boundary[1])**2 + (u_plus-right_boundary[1])**2


    return torch.sum(w * integrand) + bc_loss

def evaluate_equidistribution(model, u_t, method=0):
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
    u_t = u_t(X)
    du = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    if method == 0:
        d2u = ((du-1) + ((u - u_t)/DT)) / (EPSILON)
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
    new_knots[0] = left_boundary[0]
    new_knots[-1] = right_boundary[0]

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

def get_updated_knots(model, u_t):
    cumulative_integral, X = evaluate_equidistribution(model, u_t)
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

def discretiseU0(x_0,u_0):
    x_0.requires_grad = True
    du = torch.autograd.grad(u_0, x_0,grad_outputs=torch.ones_like(u_0), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x_0,grad_outputs=torch.ones_like(du), create_graph=True)[0]
    Monitor = d2u.abs().pow(2 / 5).view(-1) + 1e-10

    dx = x_0[1:] - x_0[:-1]
    trap = 0.5 * (Monitor[1:] + Monitor[:-1]) * dx
    G = torch.zeros_like(Monitor)
    G[1:] = torch.cumsum(trap, dim=0)
    G_Normalised = G / G[-1]

    new_knots = search_array(G_Normalised, x_0, KNOT_NUMBER)
    new_weights = torch.sin(torch.pi*new_knots)
    return new_knots, new_weights

def train_timestep(u_t):
    knots, weights = u_t.knot_points, u_t.coeffs
    model = FKS(knots, weights)

    # Inner Training Loop
    def trainParam(parameter):
        if parameter == 0:
            optimiser = optim.LBFGS([model.coeffs], lr=0.01, max_iter=INNER_EPOCHS)
            x_quad, w_quad = get_adaptive_quadrature_points(model)

        def DRM_closure():
            optimiser.zero_grad()
            loss = compute_energy_loss(model, x_quad, w_quad, EPSILON, u_t)
            loss.backward()
            return loss

        if parameter == 0:
            optimiser.step(DRM_closure)
        return model

    # Outer Training Loop
    for outer_epoch in range(OUTER_EPOCHS):
        #print("Outer Epoch: ", outer_epoch)
        model = trainParam(0)
        new_knots = get_updated_knots(model, u_t).detach()
        new_knots = new_knots.to(model.coeffs)

        with torch.no_grad():
            new_coeffs = model(new_knots).detach()

            model.set_knot_points(new_knots)
            model.coeffs.copy_(new_coeffs)

    return model

def plot_solution_history(solved_models, number_of_curves=10):
    x_plot = torch.linspace(
        left_boundary[0],
        right_boundary[0],
        1000,
        dtype=DTYPE
    ).view(-1, 1)

    indices = np.linspace(
        0,
        len(solved_models) - 1,
        number_of_curves,
        dtype=int
    )

    plt.figure()

    with torch.no_grad():
        for n in indices:
            u_plot = solved_models[n](x_plot)

            plt.plot(
                x_plot.squeeze().cpu().numpy(),
                u_plot.squeeze().cpu().numpy(),
                label=f"t = {n * DT:.3f}"
            )

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.show()

def plot_space_time(solved_models):
    x_plot = torch.linspace(
        left_boundary[0],
        right_boundary[0],
        1000,
        dtype=DTYPE
    ).view(-1, 1)

    with torch.no_grad():
        U = torch.stack([
            model(x_plot).squeeze()
            for model in solved_models
        ])

    plt.figure()

    plt.imshow(
        U.cpu().numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            left_boundary[0],
            right_boundary[0],
            0,
            (len(solved_models) - 1) * DT
        ],
        cmap="coolwarm"
    )

    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar(label="u(x,t)")
    plt.show()

def plot_spacetime_3d(
    solved_models,
    final_time=TIMESTEPS*DT,
    x_resolution=400,
    time_resolution=101
):
    final_index = min(
        int(round(final_time / DT)),
        len(solved_models) - 1
    )

    # Downsample model indices for plotting
    time_indices = np.unique(
        np.linspace(
            0,
            final_index,
            min(time_resolution, final_index + 1),
            dtype=int
        )
    )

    x = torch.linspace(
        left_boundary[0],
        right_boundary[0],
        x_resolution,
        dtype=DTYPE
    ).view(-1, 1)

    with torch.no_grad():
        U = torch.stack([
            solved_models[n](x).squeeze()
            for n in time_indices
        ])

    x_values = x.squeeze().cpu().numpy()
    t_values = time_indices * DT

    X_mesh, T_mesh = np.meshgrid(x_values, t_values)
    U_mesh = U.cpu().numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X_mesh,
        T_mesh,
        U_mesh,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, final_time)
    ax.set_title("Convection-Diffusion Problem space-time solution")

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.65,
        label="u(x,t)"
    )

    plt.tight_layout()
    plt.show()

    print("Stored models:", len(solved_models))
    print("Time indices:", time_indices)
    print("Surface shape:", U.shape)

def main():
    X = torch.linspace(left_boundary[0], right_boundary[0], 20 * KNOT_NUMBER, dtype=DTYPE, requires_grad=True)
    U_0 = torch.sin(torch.pi * X)  # Initial condition
    knots_0, weights_0 = discretiseU0(X, U_0)
    #knots_0 = torch.linspace(0, 1, KNOT_NUMBER, dtype=DTYPE)
    #weights_0 = torch.rand(KNOT_NUMBER, dtype=DTYPE)
    solvedModels = []
    solvedModels.append(FKS(knots_0, weights_0))
    for timestep in range(TIMESTEPS):
        print(f"Time Step: {timestep}")
        u_t = solvedModels[-1]
        new_model = train_timestep(u_t)
        solvedModels.append(new_model)

    #plot_solution_history(solvedModels)
    #plot_space_time(solvedModels)
    plot_spacetime_3d(solvedModels)
    return 0

if __name__ == "__main__":
    main()