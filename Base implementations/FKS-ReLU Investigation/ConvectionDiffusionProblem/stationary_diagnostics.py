"""Diagnostics for the supplied one-dimensional stationary FKS solver.

Reference functions accept and return torch tensors. Timings are wall times;
run on an otherwise idle machine and use the same device for comparisons.
"""

from contextlib import contextmanager
from time import perf_counter
import numpy as np
import torch


def _quadrature(model, order=8):
    knots = model.knot_points.detach().to(model.coeffs)
    if not bool(torch.all(knots[1:] > knots[:-1])):
        raise ValueError("Knots must be strictly increasing.")
    z, w = np.polynomial.legendre.leggauss(order)
    z, w = knots.new_tensor(z), knots.new_tensor(w)
    half = (knots[1:] - knots[:-1]) / 2
    mid = (knots[1:] + knots[:-1]) / 2
    return (mid[:, None] + half[:, None] * z).flatten(), (half[:, None] * w).flatten()


def solution_errors(model, exact, exact_derivative, order=8):
    """Absolute L2 and H1-seminorm errors, integrated on the final mesh.

    Increase order until the reported errors stabilise; a fixed order is not
    guaranteed to resolve a thin reference layer inside a coarse element.
    Derivatives of the numerical solution are evaluated away from its knots.
    """
    x, w = _quadrature(model, order)
    with torch.enable_grad():
        x.requires_grad_(True)
        uh = model(x).reshape(-1)
        duh = torch.autograd.grad(uh.sum(), x)[0]
        u = exact(x).reshape(-1)
        du = exact_derivative(x).reshape(-1)
        return {
            "L2_error": torch.sqrt(torch.sum(w * (uh - u) ** 2)).item(),
            "H1_seminorm_error": torch.sqrt(torch.sum(w * (duh - du) ** 2)).item(),
        }


def boundary_residuals(model, left_value=-1., right_value=1., kind="dirichlet"):
    """Signed endpoint residuals and their Euclidean norm.

    For Neumann data, values denote outward normal derivatives (-u', +u').
    One-sided element slopes avoid the ambiguous ReLU derivative at a knot.
    Multiply by a diffusion coefficient externally if prescribing flux data.
    """
    c = model.coeffs.detach()
    k = model.knot_points.detach().to(c)
    if kind.lower() == "dirichlet":
        values = torch.stack((c[0], c[-1]))
    elif kind.lower() == "neumann":
        values = torch.stack((-(c[1] - c[0]) / (k[1] - k[0]),
                              (c[-1] - c[-2]) / (k[-1] - k[-2])))
    else:
        raise ValueError("kind must be 'dirichlet' or 'neumann'")
    residual = values - c.new_tensor([left_value, right_value])
    return {"left_residual": residual[0].item(),
            "right_residual": residual[1].item(),
            "boundary_residual_norm": torch.linalg.vector_norm(residual).item()}


def coefficient_hessian(model, epsilon, lam=1., boundary_weight=1.):
    """Exact-in-arithmetic Hessian of the stationary Allen–Cahn energy.

    E = integral(epsilon^2*u'^2/2 + lam*(1-u^2)^2/4) dx
        + boundary_weight*((u(a)-g_a)^2 + (u(b)-g_b)^2).

    The mesh is held fixed. This is the raw nodal-coordinate Hessian, with
    Dirichlet penalties, not a Hessian with respect to movable knots.
    Three Gauss points exactly integrate every Hessian entry for linear hats.
    """
    x, w = _quadrature(model, 3)
    c = model.coeffs.detach()
    k = model.knot_points.detach().to(c)
    n = c.numel()
    if k.numel() != n:
        raise ValueError("One coefficient per knot is required.")
    element = torch.searchsorted(k, x, right=True) - 1
    h = k[element + 1] - k[element]
    t = (x - k[element]) / h
    B = c.new_zeros((x.numel(), n))
    D = torch.zeros_like(B)
    rows = torch.arange(x.numel(), device=c.device)
    B[rows, element], B[rows, element + 1] = 1 - t, t
    D[rows, element], D[rows, element + 1] = -1 / h, 1 / h
    u = B @ c
    H = epsilon ** 2 * (D.T @ (w[:, None] * D))
    H += B.T @ ((w * lam * (3 * u ** 2 - 1))[:, None] * B)
    H[0, 0] += 2 * boundary_weight
    H[-1, -1] += 2 * boundary_weight
    H = (H + H.T) / 2
    eig = torch.linalg.eigvalsh(H)
    magnitudes = eig.abs()
    maximum, minimum = magnitudes.max().item(), magnitudes.min().item()
    threshold = n * torch.finfo(c.dtype).eps * maximum
    return {
        "hessian": H.cpu().numpy(),
        "eigenvalues": eig.cpu().numpy(),
        "condition_number_2": maximum / minimum if minimum > 0 else float("inf"),
        "numerically_singular": minimum <= threshold,
        "minimum_eigenvalue": eig[0].item(),
        "maximum_eigenvalue": eig[-1].item(),
        "positive_definite": eig[0].item() > threshold,
    }


def _synchronise(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class CostTracker:
    """Use one tracker per complete solve; includes remeshing in total time.

    Wrap the entire solve in `with cost.measure():` and increment
    `cost.objective_evaluations += 1` whenever the closure evaluates the loss,
    and `cost.gradient_evaluations += 1` whenever it calls loss.backward().
    Optimiser steps are not equivalent to closure/gradient evaluations.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.wall_seconds = 0.
        self.objective_evaluations = 0
        self.gradient_evaluations = 0

    @contextmanager
    def measure(self):
        _synchronise(self.device)
        start = perf_counter()
        try:
            yield self
        finally:
            _synchronise(self.device)
            self.wall_seconds += perf_counter() - start

    def results(self):
        return {"wall_seconds": self.wall_seconds,
                "objective_evaluations": self.objective_evaluations,
                "gradient_evaluations": self.gradient_evaluations}


def convergence_study(train, knot_counts, exact, exact_derivative,
                      seeds=(0, 1, 2), order=16, device="cpu"):
    """Return one row per independent solve.

    `train(n_knots, cost)` must return a trained model and increment the
    supplied cost counters inside its closure. Timing excludes diagnostics.
    Compare error versus n_knots and wall_seconds; the empirical N-rate is
    log(error_old/error_new)/log(N_new/N_old), not an h-convergence order on
    arbitrary adaptive meshes. Keep optimisation settings comparable.
    """
    rows = []
    for seed in seeds:
        previous = None
        for n in knot_counts:
            np.random.seed(seed)
            torch.manual_seed(seed)
            cost = CostTracker(device)
            with cost.measure():
                model = train(n, cost)
            errors = solution_errors(model, exact, exact_derivative, order)
            k = model.knot_points.detach()
            row = {"seed": seed, "n_knots": n,
                   "h_max": (k[1:] - k[:-1]).max().item(),
                   **errors, **cost.results()}
            if previous is not None and n > previous["n_knots"]:
                for name in ("L2_error", "H1_seminorm_error"):
                    old, new = previous[name], row[name]
                    row[name + "_N_rate"] = (
                        np.log(old / new) / np.log(n / previous["n_knots"])
                        if old > 0 and new > 0 else float("nan"))
            rows.append(row)
            previous = row
    return rows


# Saved-solution comparison for -epsilon**2 u\'\' + u = 1.
"""Plot saved elliptic solutions and report absolute L2/H1-seminorm errors.

Run with --help for paths. No training scripts are imported or executed.
For future solve_bvp saves, preserve the actual piecewise polynomial:
    np.savez(filename, breaks=sol.sol.x, polynomial_coeffs=sol.sol.c)
This avoids interpolation error from saving only plotting samples.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
import torch
from scipy.interpolate import CubicSpline, PPoly

DEFAULT_ROOT = Path(__file__).resolve().parent


def reference(x, epsilon):
    # Stable form of x - B*(1-exp(x/epsilon)), B=1/(1-exp(1/epsilon)).
    tail = np.exp((x - 1) / epsilon)
    base = np.exp(-1 / epsilon)
    denominator = -np.expm1(-1 / epsilon)
    return x - (tail - base) / denominator, 1 - tail / (epsilon * denominator)


class SavedHats:
    def __init__(self, path, legacy_left_zero=False):
        with np.load(path, allow_pickle=False) as data:
            self.knots = np.asarray(data['knots'], dtype=float).copy()
            self.coeffs = np.asarray(data['coeffs'], dtype=float).copy()
        if legacy_left_zero:
            if len(self.coeffs) != len(self.knots) - 1:
                raise ValueError('Legacy left-zero basis requires N-1 coefficients.')
            self.coeffs = np.r_[0., self.coeffs]
        if len(self.coeffs) != len(self.knots):
            raise ValueError(f'{path}: {len(self.coeffs)} coefficients but {len(self.knots)} knots. '
                             'The supplied adaptive model requires N coefficients. '
                             'Use --legacy-left-zero ONLY if the saved basis omits the left endpoint hat.')
        if not np.all(np.diff(self.knots) > 0) or not np.allclose(self.knots[[0,-1]], [0,1]):
            raise ValueError('Expected increasing knots covering [0,1].')
        self.slopes = np.diff(self.coeffs) / np.diff(self.knots)

    def evaluate(self, x):
        j = np.clip(np.searchsorted(self.knots, x, side='right') - 1, 0, len(self.knots)-2)
        return self.coeffs[j] + self.slopes[j] * (x-self.knots[j]), self.slopes[j]


class SavedTanh:
    """Architecture inferred from weight shapes; tanh confirmed in VanillaDRM.py.

    Saved weights are evaluated in float64 for diagnostics; this does not
    recover precision lost during float32 training.
    """
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            indices = sorted(int(k.split('.')[1]) for k in data.files if k.endswith('.weight'))
            layers = []
            for i, index in enumerate(indices):
                weight = data[f'net.{index}.weight']
                layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0], dtype=torch.float64))
                if i < len(indices)-1:
                    layers.append(torch.nn.Tanh())
            self.net = torch.nn.Sequential(*layers)
            self.net.load_state_dict({k.removeprefix('net.'): torch.tensor(data[k], dtype=torch.float64)
                                      for k in data.files}, strict=True)
        self.net.eval().requires_grad_(False)

    def evaluate(self, x):
        values, derivatives = [], []
        with torch.enable_grad():
            for chunk in np.array_split(np.asarray(x), max(1, int(np.ceil(len(x)/4096)))):
                t = torch.tensor(chunk[:, None], dtype=torch.float64, requires_grad=True)
                u = self.net(t)
                du = torch.autograd.grad(u.sum(), t)[0]
                values.append(u.detach().numpy().ravel())
                derivatives.append(du.detach().numpy().ravel())
        return np.concatenate(values), np.concatenate(derivatives)


class SavedCollocation:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            if {'breaks', 'polynomial_coeffs'} <= set(data.files):
                coefficients = data['polynomial_coeffs']
                if coefficients.ndim == 3:
                    coefficients = coefficients[:, :, 0]  # u component of solve_bvp
                self.spline = PPoly(coefficients, data['breaks'], extrapolate=False)
                self.label = 'Collocation'
            else:
                self.spline = CubicSpline(data['x'], data['y'], extrapolate=False)
                self.label = 'Collocation (sample reconstruction)'
        self.knots = self.spline.x
        if not np.allclose(self.knots[[0,-1]], [0,1]):
            raise ValueError('Collocation data must cover [0,1].')

    def evaluate(self, x):
        return self.spline(x), self.spline(x, 1)


def quadrature(breaks, order):
    z, w = np.polynomial.legendre.leggauss(order)
    half = np.diff(breaks)/2
    mid = (breaks[1:]+breaks[:-1])/2
    return (mid[:,None]+half[:,None]*z).ravel(), (half[:,None]*w).ravel()


def errors(model, breaks, epsilon, order):
    x, w = quadrature(breaks, order)
    u, du = model.evaluate(x)
    exact, derivative = reference(x, epsilon)
    boundary, _ = model.evaluate(np.array([0.,1.]))
    result = {'L2_error': np.sqrt(np.sum(w*(u-exact)**2)),
              'H1_seminorm_error': np.sqrt(np.sum(w*(du-derivative)**2)),
              'left_residual': boundary[0], 'right_residual': boundary[1],
              'boundary_residual_norm': np.linalg.norm(boundary)}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Non-finite diagnostics: check saved values and domain coverage.')
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare saved convection-diffusion solutions and report L2, H1-seminorm and boundary errors.')
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--epsilon', type=float, default=.1)
    parser.add_argument('--fks', type=Path)
    parser.add_argument('--drm', type=Path)
    parser.add_argument('--collocation', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'convection_diffusion_comparison')
    parser.add_argument('--legacy-left-zero', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error('epsilon must be positive')
    folder = str(args.epsilon)
    fks_path = args.fks or args.root/'FKSmodelParams'/folder/'Approximation.npz'
    drm_path = args.drm or args.root/'VanillaDRM'/folder/'Params.npz'
    if not drm_path.exists() and args.drm is None:
        drm_path = args.root/'VanillaDRM'/folder/'PINN: Gauss.npz'
    coll_path = args.collocation or args.root/'Collocation'/folder/'params.npz'
    hats = SavedHats(fks_path, args.legacy_left_zero)
    coll = SavedCollocation(coll_path)
    models = {'FKS': hats, coll.label: coll, 'Vanilla DRM': SavedTanh(drm_path)}
    # Common independent integration partition resolves layers and all spline breaks.
    breaks = np.unique(np.r_[np.linspace(0,1,max(101,int(np.ceil(10/args.epsilon))+1)),
                             hats.knots, coll.knots])
    rows = []
    for name, model in models.items():
        low, high = errors(model, breaks, args.epsilon, 8), errors(model, breaks, args.epsilon, 16)
        for key in ('L2_error','H1_seminorm_error'):
            if not np.isclose(low[key], high[key], rtol=1e-6, atol=1e-10):
                raise RuntimeError(f'{name}: {key} has not stabilised under quadrature refinement.')
        rows.append({'Method': name, **high})
    columns = list(rows[0])
    widths = [max(len(k), max(len(str(r[k])) if k=='Method' else 13 for r in rows)) for k in columns]
    print(' | '.join(k.ljust(w) for k,w in zip(columns,widths)))
    print('-'*(sum(widths)+3*(len(widths)-1)))
    for row in rows:
        print(' | '.join((str(row[k]) if k=='Method' else f'{row[k]:.6e}').ljust(w)
                         for k,w in zip(columns,widths)))
    print('\nQuadrature check passed: 8 versus 16 points per interval.')
    if 'reconstruction' in coll.label:
        print('Collocation errors include cubic interpolation of the saved 200-point sample;\n'
              'they are NOT errors of the original solve_bvp polynomial.')
    print('Training cost and convergence cannot be recovered from these snapshots.')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'errors.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.unique(np.r_[np.linspace(0,1,10001), hats.knots])
    exact, _ = reference(x,args.epsilon)
    fig, axes = plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
    axes[0].plot(x,exact,'k-',lw=2,label='Analytical solution')
    for (name,model),color,style in zip(models.items(),['tab:blue','tab:orange','tab:green'],['--',':','-.']):
        u,_ = model.evaluate(x)
        axes[0].plot(x,u,color=color,ls=style,label=name)
        axes[1].semilogy(x,np.maximum(np.abs(u-exact),1e-16),color=color,ls=style,label=name)
    axes[0].set_ylabel('u(x)')
    axes[0].set_title(r'$-\varepsilon u^{\prime\prime}+u^{\prime}=1$, $u(0)=u(1)=0$, '+f'ε={args.epsilon:g}')
    axes[0].legend(fontsize=9)
    axes[1].set(xlabel='x',ylabel='Absolute error')
    for ax in axes:
        ax.grid(alpha=.25); ax.set_xlim(0,1)
    fig.savefig(args.output/'comparison.png',dpi=200)
    fig.savefig(args.output/'comparison.pdf')
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f'Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
