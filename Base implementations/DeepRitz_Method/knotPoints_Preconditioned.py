import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from numpy.polynomial.legendre import Legendre

#y = a0 + sum_{k_i} (\alpha_i ReLU(x-k_i))


EPSILON = 0.1
EPOCHS = 100000

# -------- Helpers to build preconditioner in FKS (w) space --------
# T maps nodal values w -> ReLU slope-jumps c. Here we use it to form S_w = T^T T.

def build_T(knots: torch.Tensor) -> torch.Tensor:
    """Return (N-1)×N tri-diagonal T with spacings h_i = k_{i+1}-k_i.
    T w = c implements c0 = (w1-w0)/h0 and ci = (wi+1-wi)/hi - (wi-wi-1)/h_{i-1}.
    """
    k = knots
    N = k.numel()
    assert N >= 3
    h = k[1:] - k[:-1]  # (N-1,)
    T = torch.zeros((N-1, N), device=k.device, dtype=k.dtype)
    # first row
    T[0, 0] = -1.0 / h[0]
    T[0, 1] =  1.0 / h[0]
    # interior rows i = 1..N-2
    for i in range(1, N-1):
        T[i, i-1] =  1.0 / h[i-1]
        T[i, i]   = -(1.0 / h[i-1] + 1.0 / h[i])
        T[i, i+1] =  1.0 / h[i]
    return T


def build_Sw_cholesky(knots: torch.Tensor, ridge: float = 1e-8, max_tries: int = 8):
    """Return the Cholesky factor L of S_w = T^T T + ridge*I.
    If numerical semidefiniteness persists, increase ridge geometrically until PD.
    """
    T = build_T(knots)
    Sw = T.T @ T
    N = Sw.shape[0]
    I = torch.eye(N, device=Sw.device, dtype=Sw.dtype)
    delta = ridge
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(Sw + delta * I)
        except RuntimeError:
            delta *= 10.0  # amplify ridge and retry

    # Final attempt with scale-aware ridge
    scale = torch.clamp(torch.mean(torch.diag(Sw)), min=1e-12)
    delta = max(delta, 1e-8 * float(scale))
    return torch.linalg.cholesky(Sw + delta * I)

# -------- FKS preconditioning (general): map nodal FKS weights w -> (a0, c) --------
# No explicit Dirichlet enforcement: w[0] and w[-1] are learnable and BCs are handled by the loss.

def w_to_a_c(knots: torch.Tensor, w_full: torch.Tensor):
    """
    knots: (N,) strictly increasing
    w_full: (N,) FKS nodal weights (all learnable, including endpoints)
    Returns:
      c: (N-1,) ReLU coefficients so that y(x) ≈ sum_i c_i ReLU(x - k_i)
    """
    k = knots

    # spacing reciprocals for interior indices i=1..N-2
    alpha = 1.0 / (k[1:-1] - k[:-2])          # (N-2,)
    gamma = 1.0 / (k[2:]   - k[1:-1])         # (N-2,)

    # c_0 controls left-interval slope; general form uses both w0 and w1
    c0 = (w_full[1] - w_full[0]) / (k[1] - k[0])

    # interior coefficients c_i for i=1..N-2
    w_im1 = w_full[:-2]
    w_i   = w_full[1:-1]
    w_ip1 = w_full[2:]
    c_interior = gamma * (w_ip1 - w_i) - alpha * (w_i - w_im1)

    # final vector length N-1: hinges at k[0]..k[N-2]
    c = torch.cat([c0.view(1), c_interior], dim=0)
    return c


class PrecondKnotReLU(nn.Module):
    """
    ReLU model parameterised in FKS space (preconditioned):
      - Train all nodal weights w[0..N-1] (including endpoints)
      - Map to ReLU scalings c and constant offset a0, then evaluate y(x) = a0 + sum_i c_i ReLU(x - k_i)
      - BCs are learned via loss, not hard-coded
    """
    def __init__(self, knot_points):
        super().__init__()
        knots = torch.as_tensor(knot_points, dtype=torch.float32)
        # ensure strictly increasing & on device of module
        assert torch.all(knots[1:] > knots[:-1]), "knot_points must be strictly increasing"
        self.register_buffer("knot_points", knots)
        N = knots.numel()
        assert N >= 3, "Need at least 3 knots including endpoints"
        # trainable nodal weights (including endpoints). BCs are learned via loss, not fixed here.
        self.w = nn.Parameter(torch.zeros(N, dtype=torch.float32))

    def forward(self, x):
        # Accept (B,1) or (B,) and return (B,1) for shape-compatibility with quadrature code
        if x.dim() == 2 and x.size(1) == 1:
            x_flat = x.squeeze(1)
        else:
            x_flat = x
        k = self.knot_points
        w_full = self.w
        c = w_to_a_c(k, w_full)                 # scalar, (N-1,)
        relus = torch.relu(x_flat[:, None] - k[:-1][None, :])  # (B, N-1)
        y = relus @ c                          # (B,)
        return y.unsqueeze(1)                       # (B,1)


# Energy loss computation using quadrature points
def compute_energy_loss(model, x_quad, w_quad, epsilon):
    """
    Compute energy functional:
        E(y) = ∑_i w_i [ (ε²/2) * (y')² + (1/2) * y² - y ]
    using quadrature points x_quad and weights w_quad.
    """
    x_quad.requires_grad_(True)
    u = model(x_quad)

    # First derivative y'
    du = torch.autograd.grad(u, x_quad, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Energy integrand
    integrand = (epsilon**2 / 2) * du**2 + (1/2) * u**2 - u

    # Boundary condition loss: u(0) = u(1) = 0
    u0_pred = model(torch.tensor([[0.0]], device=x_quad.device, dtype=x_quad.dtype))
    u1_pred = model(torch.tensor([[1.0]], device=x_quad.device, dtype=x_quad.dtype))
    bc_loss = u0_pred.pow(2) + u1_pred.pow(2)

    # Weighted quadrature sum
    return torch.sum(w_quad * integrand) + bc_loss # + phi

def get_knot_points(N=50):
    # Build clustered-but-unique knots: [0, eps) ∪ [eps, 1-eps) ∪ [1-eps, 1]
    start_knot_points = np.linspace(0.0, EPSILON, N, endpoint=False)
    mid_knot_points   = np.linspace(EPSILON, 1.0-EPSILON, N, endpoint=False)
    end_knot_points   = np.linspace(1.0-EPSILON, 1.0, N, endpoint=True)
    knot_points = np.concatenate((start_knot_points, mid_knot_points, end_knot_points))
    # ensure strictly increasing
    knot_points = np.unique(knot_points)
    return knot_points

# Quadrature points and weights (e.g., Gauss-Legendre)
def get_quad_points(N=50,type='uniform'):
        if type == 'uniform':
            M = 3*N
            x_quad = torch.linspace(0, 1, M).unsqueeze(1)
            w_quad = torch.full((M,1), 1.0/M, dtype=torch.float32)
        elif type == 'gauss':
            x_quad_np, w_quad_np = roots_legendre(N)
            x_quad = torch.tensor((x_quad_np + 1) / 2, dtype=torch.float32).unsqueeze(1)  # map from [-1,1] to [0,1]
            w_quad = torch.tensor(w_quad_np / 2, dtype=torch.float32).unsqueeze(1)        # adjust weights accordingly
        else:
            raise ValueError("Unsupported quadrature method")

        return x_quad, w_quad

def train_model(x_quad, w_quad):
    knot_points = get_knot_points()
    model = PrecondKnotReLU(knot_points)
    # Precompute Cholesky factor of S_w = T^T T + δI for gradient preconditioning in w-space
    L = build_Sw_cholesky(model.knot_points, ridge=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = compute_energy_loss(model, x_quad, w_quad, EPSILON)
        loss.backward()

        # Gradient preconditioning in FKS (w) coordinates:
        # solve (T^T T + δI) p = grad_w, then replace grad_w ← p
        with torch.no_grad():
            g = model.w.grad.view(-1, 1)                # (N,1)
            # forward/backward solves with the cached Cholesky L
            y = torch.cholesky_solve(g, L)              # solves (T^T T + δI) y = g
            model.w.grad.copy_(y.view_as(model.w))

        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model

# Plot the results
def create_results(x_test, x_quad, w_quad, color='red', label=''):
    model = train_model(x_quad,w_quad)
    y_pred = model(x_test).detach().numpy()
    plt.plot(x_test.numpy(), y_pred, label=label, color=color, linestyle='--')

def main():
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    u2 = lambda x: 1 - np.cosh((x - 0.5) / EPSILON) / np.cosh(1 / (2 * EPSILON))
    y_true = np.array([u2(x) for x in x_test])
    plt.plot(x_test.numpy(), y_true, label='True Solution', color='green')

    x_uniform, w_uniform = get_quad_points(type='uniform')
    x_gauss, w_gauss = get_quad_points(type='gauss')
    create_results(x_test, x_uniform, w_uniform, color='red',label='Uniform')
    create_results(x_test, x_gauss, w_gauss, color='blue',label='Gaussian')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    title = r"$-ε^2 u''(x) + u(x) = 1$, ε = {:.5f}".format(EPSILON)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()