import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

class PINN(nn.Module):
    def __init__(self, N):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, N),
            nn.Tanh(),
            nn.Linear(N, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_loss(model, x, x_b, x_s, w=None, bc_weight=10.0):
    x = x.clone().detach().requires_grad_(True)

    u = model(x).view(-1, 1)
    du = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    grad_u_sq = torch.sum(du**2, dim=1, keepdim=True)

    # Energy for -Delta u = 1 with homogeneous Dirichlet data:
    # J(u) = int_Omega 1/2 |grad u|^2 - u dx
    integrand = 0.5 * grad_u_sq - u

    if w is None:
        interior_loss = torch.mean(integrand)
    else:
        interior_loss = torch.sum(w * integrand)

    # Boundary condition: u = 0 on the outer square boundary
    u_b = model(x_b).view(-1, 1)
    square_boundary_loss = torch.mean(u_b**2)

    # Boundary condition: u = 0 on the two internal edges of the removed corner
    u_s = model(x_s).view(-1, 1)
    slit_boundary_loss = torch.mean(u_s**2)

    return interior_loss + bc_weight * (square_boundary_loss + slit_boundary_loss)

def sample_interior(N, device="cpu"):
    """Sample points in [-1,1]^2 \ ((0,1] x [-1,0))."""
    points = []
    remaining = N

    while remaining > 0:
        # Oversample, then reject the removed lower-right quadrant.
        candidate = 2 * torch.rand(2 * remaining, 2, device=device) - 1
        keep = ~((candidate[:, 0] > 0.0) & (candidate[:, 1] < 0.0))
        candidate = candidate[keep]
        candidate = candidate[:remaining]
        points.append(candidate)
        remaining -= candidate.shape[0]

    return torch.cat(points, dim=0)


def sample_square_boundary(N, device="cpu"):
    """Sample the exposed outer boundary of the corner domain."""
    n = N // 4

    s_full = 2 * torch.rand(n, 1, device=device) - 1
    s_left = 2 * torch.rand(n, 1, device=device) - 1
    s_pos = torch.rand(n, 1, device=device)
    s_neg = torch.rand(n, 1, device=device) - 1.0

    left = torch.cat([-torch.ones_like(s_left), s_left], dim=1)       # x=-1, y in [-1,1]
    top = torch.cat([s_full, torch.ones_like(s_full)], dim=1)         # y=1, x in [-1,1]
    right_top = torch.cat([torch.ones_like(s_pos), s_pos], dim=1)     # x=1, y in [0,1]
    bottom_left = torch.cat([s_neg, -torch.ones_like(s_neg)], dim=1)  # y=-1, x in [-1,0]

    return torch.cat([left, top, right_top, bottom_left], dim=0)


def sample_slit_boundary(N, device="cpu"):
    """Sample the two internal boundary edges of the removed corner.

    Internal boundary:
        (0,1] x {0}
        {0} x [-1,0)
    """
    n = N // 2

    x_horizontal = torch.rand(n, 1, device=device)
    y_horizontal = torch.zeros_like(x_horizontal)
    horizontal = torch.cat([x_horizontal, y_horizontal], dim=1)

    y_vertical = torch.rand(N - n, 1, device=device) - 1.0
    x_vertical = torch.zeros_like(y_vertical)
    vertical = torch.cat([x_vertical, y_vertical], dim=1)

    return torch.cat([horizontal, vertical], dim=0)


def train_PINN(N, x, x_b, x_s, weights=None, epochs=20000, lr=0.01, bc_weight=10.0):
    loss_list = []
    model = PINN(N).to(x.device)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = compute_loss(model, x, x_b, x_s, weights, bc_weight=bc_weight)
        loss.backward()
        optimiser.step()
        loss_list.append(loss.item())
        if epoch % 500 == 0:
            print(f"Training Epoch {epoch}, Loss: {loss.item():.6f}")

    return model, loss_list

def plot_solution(model, n_grid=200, device="cpu"):
    x = torch.linspace(-1.0, 1.0, n_grid, device=device)
    y = torch.linspace(-1.0, 1.0, n_grid, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    XY = torch.cat([
        X.reshape(-1, 1),
        Y.reshape(-1, 1)
    ], dim=1)

    with torch.no_grad():
        U = model(XY).reshape(n_grid, n_grid).cpu().numpy()

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    # Mask the removed lower-right corner for visual clarity.
    removed_corner_mask = (X_np > 0.0) & (Y_np < 0.0)
    U[removed_corner_mask] = np.nan

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X_np, Y_np, U, levels=50)
    plt.colorbar(contour, label="u(x, y)")
    plt.plot([0, 1], [0, 0], "k-", linewidth=2, label="internal boundary")
    plt.plot([0, 0], [-1, 0], "k-", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN solution for 2D Poisson problem on corner domain")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(N=20, EPOCHS=20000, plot=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_interior = 1000
    n_square_boundary = 200
    n_slit_boundary = 100

    x = sample_interior(n_interior, device=device)
    x_b = sample_square_boundary(n_square_boundary, device=device)
    x_s = sample_slit_boundary(n_slit_boundary, device=device)

    model, loss_list = train_PINN(
        N=N,
        x=x,
        x_b=x_b,
        x_s=x_s,
        weights=None,
        epochs=EPOCHS,
        lr=0.01,
        bc_weight=10.0
    )

    if plot:
        plot_solution(model, n_grid=200, device=device)

        plt.figure(figsize=(8, 4))
        plt.plot(range(len(loss_list)), loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss curve, width={N}")
        plt.tight_layout()
        plt.show()

    return model, loss_list

if __name__ == "__main__":
    main(N=20, EPOCHS=1000, plot=True, seed=0)