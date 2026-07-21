import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

# Full Hessians scale like P^2 in memory and are very expensive.
# Start small; widths above about 20 quickly become impractical for this architecture.
WIDTHS = [5, 10, 20]

class PINN(nn.Module):
    def __init__(self, width) :
        super(PINN, self).__init__()
        self.width = width
        self.net = nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
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

    # Boundary condition: u = 0 on the internal slit [0, 1) x {0}
    u_s = model(x_s).view(-1, 1)
    slit_boundary_loss = torch.mean(u_s**2)

    return interior_loss + bc_weight * (square_boundary_loss + slit_boundary_loss)

def make_functional_model(model):
    named_params = dict(model.named_parameters())
    names = list(named_params.keys())
    shapes = [p.shape for p in named_params.values()]
    sizes = [p.numel() for p in named_params.values()]
    params0 = torch.cat([p.detach().flatten() for p in named_params.values()])

    def unflatten(flat_params):
        chunks = torch.split(flat_params, sizes)
        return {
            name: chunk.reshape(shape)
            for name, chunk, shape in zip(names, chunks, shapes)
        }

    def fmodel(flat_params, points):
        return functional_call(model, unflatten(flat_params), (points,))

    return fmodel, params0


def compute_energy_loss_from_model(model_fn, x, x_b, x_s, w=None, bc_weight=10.0):
    x = x.clone().detach().requires_grad_(True)

    u = model_fn(x).view(-1, 1)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    integrand = 0.5 * torch.sum(du**2, dim=1, keepdim=True) - u
    interior_loss = torch.mean(integrand) if w is None else torch.sum(w * integrand)

    square_boundary_loss = torch.mean(model_fn(x_b).view(-1, 1)**2)
    slit_boundary_loss = torch.mean(model_fn(x_s).view(-1, 1)**2)

    return interior_loss + bc_weight * (square_boundary_loss + slit_boundary_loss)


def compute_condition_number( model, x, x_b, x_s, weights=None, bc_weight=10.0, eig_tol=1e-10, max_hessian_params=5000):
    fmodel, params0 = make_functional_model(model)

    if params0.numel() > max_hessian_params:
        print(f"Skipping Hessian: {params0.numel()} parameters exceeds " f"max_hessian_params={max_hessian_params}.")
        return torch.tensor(float("nan"), device=params0.device)

    def loss_from_params(flat_params):
        return compute_energy_loss_from_model(lambda points: fmodel(flat_params, points), x, x_b, x_s, w=weights, bc_weight=bc_weight)

    hessian = torch.autograd.functional.hessian(loss_from_params, params0, create_graph=False, strict=False, vectorize=False)

    eigs = torch.abs(torch.linalg.eigvalsh(hessian))
    eigs = eigs[torch.isfinite(eigs)]
    eigs = eigs[eigs > eig_tol]

    if eigs.numel() == 0:
        return torch.tensor(float("inf"), device=params0.device)

    return (eigs.max() / eigs.min()).detach()

def sample_interior(N, device="cpu"):
    X = 2 * torch.rand(N, 2, device=device) - 1
    return X


def sample_square_boundary(N, device="cpu"):
    n = N // 4

    s = 2 * torch.rand(n, 1, device=device) - 1

    left   = torch.cat([-torch.ones_like(s), s], dim=1)
    right  = torch.cat([ torch.ones_like(s), s], dim=1)
    bottom = torch.cat([s, -torch.ones_like(s)], dim=1)
    top    = torch.cat([s,  torch.ones_like(s)], dim=1)

    return torch.cat([left, right, bottom, top], dim=0)


def sample_slit_boundary(N, device="cpu"):
    # x in [0,1), y = 0
    x = torch.rand(N, 1, device=device)
    y = torch.zeros_like(x)
    return torch.cat([x, y], dim=1)


def train_PINN(width, x, x_b, x_s, weights=None, epochs=4000, lr=0.01, bc_weight=10.0, condition_data=None, condition_every=500):
    model = PINN(width).to(x.device)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    condition_history = []

    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = compute_loss(model, x, x_b, x_s, weights, bc_weight=bc_weight)
        loss.backward()
        optimiser.step()

        loss_list.append(loss.item())

        if condition_data is not None and epoch % condition_every == 0:
            x_c, x_b_c, x_s_c = condition_data
            condition_number = compute_condition_number(
                model,
                x_c,
                x_b_c,
                x_s_c,
                weights=None,
                bc_weight=bc_weight,
            )
            condition_history.append((epoch, condition_number.item()))

            print(
                f"Width {width}, epoch {epoch}, "
                f"loss={loss.item():.6f}, "
                f"cond={condition_number.item():.6e}"
            )

    return model, loss_list, condition_history

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

    # Mask the internal Dirichlet slit for visual clarity.
    slit_mask = (X_np >= 0.0) & (Y_np > -1.0 / n_grid) & (Y_np < 1.0 / n_grid)
    U[slit_mask] = np.nan

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X_np, Y_np, U, levels=50)
    plt.colorbar(contour, label="u(x, y)")
    plt.plot([0, 1], [0, 0], "k-", linewidth=2, label="internal boundary")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN solution for 2D Poisson problem")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_interior = 10000
    n_square_boundary = 2000
    n_slit_boundary = 1000

    x = sample_interior(n_interior, device=device)
    x_b = sample_square_boundary(n_square_boundary, device=device)
    x_s = sample_slit_boundary(n_slit_boundary, device=device)

    # Hessian diagnostics use a small fixed set, not the full training data.
    condition_data = (
        sample_interior(200, device=device),
        sample_square_boundary(80, device=device),
        sample_slit_boundary(40, device=device),
    )

    all_condition_histories = {}
    all_loss_lists = {}
    trained_models = {}

    for width in WIDTHS:
        print(f"\nTraining network with width {width}")
        model, loss_list, condition_history = train_PINN(
            width=width,
            x=x,
            x_b=x_b,
            x_s=x_s,
            weights=None,
            epochs=1001,
            lr=0.01,
            bc_weight=10.0,
            condition_data=condition_data,
            condition_every=500,
        )

        trained_models[width] = model
        all_loss_lists[width] = loss_list
        all_condition_histories[width] = condition_history

    final_width = WIDTHS[-1]
    plot_solution(trained_models[final_width], n_grid=200, device=device)

    plt.figure()
    for width, loss_list in all_loss_lists.items():
        plt.plot(range(len(loss_list)), loss_list, label=f"width={width}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    for width, condition_history in all_condition_histories.items():
        epochs_logged = [entry[0] for entry in condition_history]
        conditions = [entry[1] for entry in condition_history]
        plt.semilogy(epochs_logged, conditions, marker="o", label=f"width={width}")
    plt.xlabel("Epoch")
    plt.ylabel("Hessian condition number")
    plt.legend()
    plt.tight_layout()
    plt.show()