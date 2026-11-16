import numpy as np
import torch
from torch import nn
import torch.optim as optim
import copy

GAMMA = 10.0
MAX_DT = 0.01
MIN_DT = 1e-18
MAX_T = 10
DT_RESOLUTION = 0.01
TIMESTEPS = 10000
DTYPE = torch.float64
EPOCHS = 2500
LAYERS = 1
WIDTH = 100
POINTS_PER_INTERVAL = 4

torch.set_num_threads(1)

class ReLU3(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(3)


class PINN(nn.Module):
    def __init__(self, width=WIDTH, depth=LAYERS, dtype=DTYPE, params=None):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.append(nn.Linear(1 if i == 0 else width, width))
            layers.append(ReLU3())

        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def effective_knots(model, domain=(-1., 1.), previous=None, next_id=0,
                    grid_points=4097, iterations=50, root_tolerance=None):
    """Return [layer, neuron, track_id, x] rows and the next unused ID.

    Uses the actual model.net activations: ReLU, ReLU cubed or Tanh.
    For Tanh these are preactivation zeros (transition centres), not knots.
    Deeper roots are approximate: increase grid_points to check resolution.
    Sign changes and sampled minima of |preactivation| are searched; narrow
    root pairs may be missed. Zero intervals have no unique root and are omitted.
    Tracking is greedy proximity matching, not guaranteed branch identity.
    This analysis function detaches values and does not support backpropagation.
    """
    left, right = map(float, domain)
    if not np.isfinite([left, right]).all() or left >= right:
        raise ValueError('Require a finite domain with left < right.')
    if grid_points < 3 or iterations < 1:
        raise ValueError('Require grid_points >= 3 and iterations >= 1.')
    if root_tolerance is not None and root_tolerance <= 0:
        raise ValueError('root_tolerance must be positive.')
    modules = list(model.net)
    linear_indices = [i for i, layer in enumerate(modules)
                      if isinstance(layer, nn.Linear)][:-1]
    p = next(model.parameters())
    eps = torch.finfo(p.dtype).eps
    grid = np.linspace(left, right, grid_points)
    roots = []

    def evaluate(x, stop, neurons=None):
        with torch.no_grad():
            h = torch.as_tensor(x, dtype=p.dtype, device=p.device).reshape(-1, 1)
            for module in modules[:stop + 1]:
                h = module(h)
            if neurons is not None:
                ids = torch.as_tensor(neurons, dtype=torch.long, device=p.device)
                h = h[torch.arange(len(ids), device=p.device), ids]
            values = h.detach().cpu().numpy().astype(float)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite hidden preactivation during knot search.')
        return values

    for layer_id, module_index in enumerate(linear_indices):
        layer = modules[module_index]
        if module_index == 0:
            W = layer.weight.detach().cpu().numpy()[:, 0]
            B = (layer.bias.detach().cpu().numpy() if layer.bias is not None
                 else np.zeros(len(W)))
            for neuron in np.flatnonzero(W != 0):
                x = float(-B[neuron] / W[neuron])
                if left <= x <= right:
                    roots.append((layer_id, int(neuron), x))
            grid = np.unique(np.r_[grid, [r[2] for r in roots]])
            continue

        Z = evaluate(grid, module_index)
        scale = np.max(np.abs(Z), axis=0)
        tolerance = ((32 * eps) * scale if root_tolerance is None
                     else np.full_like(scale, root_tolerance))
        found = []

        # Isolated sampled zeros; do not label a flat zero interval as roots.
        for j, neuron in np.argwhere(Z == 0):
            if ((j == 0 or Z[j - 1, neuron] != 0)
                    and (j == len(grid) - 1 or Z[j + 1, neuron] != 0)):
                found.append((int(neuron), grid[j]))

        j, neurons = np.where(((Z[:-1] < 0) & (Z[1:] > 0))
                              | ((Z[:-1] > 0) & (Z[1:] < 0)))
        lo, hi, flo = grid[j].copy(), grid[j + 1].copy(), Z[j, neurons].copy()
        if len(j):
            for _ in range(iterations):
                mid = (lo + hi) / 2
                fm = evaluate(mid, module_index, neurons)
                same = np.signbit(fm) == np.signbit(flo)
                exact = fm == 0
                lo = np.where(same | exact, mid, lo)
                hi = np.where(~same | exact, mid, hi)
                flo = np.where(same, fm, flo)
            found.extend(zip(neurons.tolist(), ((lo + hi) / 2).tolist()))

        # Search valleys for tangential zeros, which have no sign change.
        V = np.abs(Z)
        j, neurons = np.where((V[1:-1] < V[:-2]) & (V[1:-1] < V[2:])
                              & (V[1:-1] > 0)
                              & (np.signbit(Z[:-2]) == np.signbit(Z[2:])))
        lo, hi = grid[j].copy(), grid[j + 2].copy()
        if len(j):
            ratio = (np.sqrt(5) - 1) / 2
            for _ in range(iterations):
                a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
                fa = np.abs(evaluate(a, module_index, neurons))
                fb = np.abs(evaluate(b, module_index, neurons))
                choose_left = fa <= fb
                lo, hi = np.where(choose_left, lo, a), np.where(choose_left, b, hi)
            x = (lo + hi) / 2
            accepted = np.abs(evaluate(x, module_index, neurons)) <= tolerance[neurons]
            found.extend(zip(neurons[accepted].tolist(), x[accepted].tolist()))

        for neuron in sorted({int(n) for n, _ in found}):
            positions = sorted(x for n, x in found if n == neuron)
            last = None
            for x in positions:
                if last is None or x - last > 8 * eps * max(1., abs(x)):
                    roots.append((layer_id, neuron, float(x)))
                    last = x
        grid = np.unique(np.r_[grid, [r[2] for r in roots if r[0] == layer_id]])

    roots = np.array(sorted(set(roots)), dtype=float).reshape(-1, 3)
    knots = np.empty((len(roots), 4))
    knots[:, :2] = roots[:, :2]
    knots[:, 2] = -1
    knots[:, 3] = roots[:, 2]

    if previous is not None and len(previous):
        previous = np.asarray(previous)
        next_id = max(next_id, int(previous[:, 2].max()) + 1)

        for layer_id, neuron in np.unique(knots[:, :2], axis=0):
            current_indices = np.flatnonzero(
                (knots[:, 0] == layer_id) & (knots[:, 1] == neuron)
            )
            old_indices = np.flatnonzero(
                (previous[:, 0] == layer_id) & (previous[:, 1] == neuron)
            )

            distances = np.abs(
                knots[current_indices, 3, None]
                - previous[old_indices, 3][None, :]
            )

            # Greedy one-to-one matching, shortest distance first.
            for _ in range(min(len(current_indices), len(old_indices))):
                i, j = np.unravel_index(
                    np.argmin(distances), distances.shape
                )
                knots[current_indices[i], 2] = previous[old_indices[j], 2]
                distances[i, :] = np.inf
                distances[:, j] = np.inf

    for i in np.flatnonzero(knots[:, 2] < 0):
        knots[i, 2] = next_id
        next_id += 1

    return knots, next_id

def GetCollocationPoints(knots, points_per_interval=POINTS_PER_INTERVAL):
    knots = torch.as_tensor(knots, dtype=torch.float64)
    nodes = np.polynomial.legendre.leggauss(points_per_interval)[0]
    nodes = torch.as_tensor(nodes, dtype=knots.dtype)

    mid = (knots[:-1] + knots[1:]) / 2
    halfwidth = (knots[1:] - knots[:-1]) / 2

    return (mid[:, None] + halfwidth[:, None] * nodes).reshape(-1, 1)

def compute_loss(model, X, U_n, dt):
    X.requires_grad_(True)

    # Defining new network and derivatives
    U = model(X)
    U_x = torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    U_xx = torch.autograd.grad(U_x, X, grad_outputs=torch.ones_like(U), create_graph=True)[0]

    # ODE residual: U_{N+1} - U_N - dt * (U_{N+1}_xx + U_{N+1}^3)
    residual = U - U_n - dt * (U_xx + U ** 3)
    interior_loss = torch.mean(residual ** 2)

    # Boundary Conditions
    boundary = X.new_tensor([[-1.], [1.]])
    bc_loss = model(boundary).square().mean()

    return interior_loss + bc_loss


def Solve_Next_TimeStep(U, X, U_n, dt, lr=1e-3):
    with torch.no_grad():
        old_values = U_n(X).detach()

    X = X.detach().requires_grad_(True)
    optimiser = optim.Adam(U.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=50,
        threshold=1e-4,
        min_lr=1e-7,
    )

    for epoch in range(EPOCHS):
        optimiser.zero_grad()
        X.grad = None

        loss = compute_loss(U, X, old_values, dt)

        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")

        loss.backward()
        optimiser.step()
        scheduler.step(loss.item())

    return U, optimiser.param_groups[0]["lr"]

def SampleSolution(model, N=200):
    if N < 4 or N % 2:
        raise ValueError("N must be even and at least 4.")

    p = (
        next(model.parameters())
        if isinstance(model, nn.Module)
        else torch.empty((), dtype=DTYPE)
    )

    with torch.no_grad():
        endpoints = torch.tensor(
            [[0.], [1.]], dtype=p.dtype, device=p.device
        )
        values = model(endpoints).reshape(-1)
        target = values[0] / 2

        if not (values[0] > 0 and values[1] < target):
            raise ValueError("Require U(0) > 0 and U(1) < U(0)/2.")

        lo, hi = 0., 1.
        for _ in range(60):
            mid = (lo + hi) / 2
            point = torch.tensor([[mid]], dtype=p.dtype, device=p.device)
            if model(point).item() > target.item():
                lo = mid
            else:
                hi = mid
        r = (lo + hi) / 2

        n_inner = N // 2
        n_left = (N - n_inner) // 2
        n_right = N - n_inner - n_left

        # ±r belong to the inner grid; avoid duplicate endpoints.
        left = torch.linspace(
            -1., -r, n_left + 1, dtype=p.dtype, device=p.device
        )[:-1]
        inner = torch.linspace(
            -r, r, n_inner, dtype=p.dtype, device=p.device
        )
        right = torch.linspace(
            r, 1., n_right + 1, dtype=p.dtype, device=p.device
        )[1:]

        x = torch.cat((left, inner, right)).sort().values
        u = model(x[:, None]).reshape(-1)

    return x.cpu().tolist(), u.cpu().tolist()

def main():
    lr = 1e-3
    def U_0(x):
        return GAMMA * torch.cos(torch.pi * x / 2)

    soln_list = [U_0]
    history = []
    knots = np.empty((0, 4))
    next_id = 0
    t = 0.0

    x, u = SampleSolution(U_0)
    peak = abs(GAMMA)
    history.append(dict(t=t, x=x, u=u, peak=peak,
                        knots=torch.from_numpy(knots.copy()),
                        knot_columns=["layer", "neuron", "track_id", "x"],
                        next_knot_id=next_id))
    torch.save(history, "solution_history.pt")

    for step in range(TIMESTEPS):
        if t >= MAX_T:
            break

        dt = min(
            MAX_DT,
            DT_RESOLUTION / peak**2 if peak > 0 else MAX_DT,
            MAX_T - t,
        )
        if dt < MIN_DT or t + dt == t:
            break

        previous = soln_list[-1]
        model = (
            PINN().to(dtype=DTYPE)
            if step == 0
            else copy.deepcopy(previous)
        )

        # Gauss points between the previous snapshot's sample locations.
        X = GetCollocationPoints(np.asarray(x))
        p = next(model.parameters())
        X = X.to(dtype=p.dtype, device=p.device)

        model, lr = Solve_Next_TimeStep(model, X, previous, dt, lr)

        x, u = SampleSolution(model)
        with torch.no_grad():
            centre = model(p.new_tensor([[0.]])).item()
        peak = max(abs(centre), max(abs(value) for value in u))

        if not np.isfinite(peak):
            raise FloatingPointError("Nonfinite solution amplitude.")

        knots, next_id = effective_knots(
            model, previous=knots, next_id=next_id
        )
        t += dt
        soln_list.append(model)
        history.append(dict(t=t, x=x, u=u, peak=peak,
                            knots=torch.from_numpy(knots.copy()),
                            knot_columns=["layer", "neuron", "track_id", "x"],
                            next_knot_id=next_id))

        torch.save(history, "solution_history.pt")
        print(f"{step + 1}: t={t:.8g}, dt={dt:.3e}, peak={peak:.6g}")

    return soln_list, history


if __name__ == "__main__":
    soln_list, history = main()


