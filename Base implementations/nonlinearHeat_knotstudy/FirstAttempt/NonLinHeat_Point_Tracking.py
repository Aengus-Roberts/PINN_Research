"""Deep Ritz solver for u_t = u_xx + u^3, u(-1,t) = u(1,t) = 0.

Implicit diffusion, explicit reaction; u(x,0) = GAMMA*cos(pi*x/2).
Run this file, or call history = solve() and plot_results(history).
ReLU roots and moving quadrature endpoints retain their autograd graphs.
Temporal Tikhonov term: lambda/(2*dt) * sum((k_new-k_old)**2).
The whole variational objective is scaled by dt/max(1, peak_old)**2,
so its implemented contribution is lambda*sum((k_new-k_old)**2)/(2*scale**2).
Roots are matched one-to-one by minimum squared distance within each neuron.
Unmatched births/deaths and identically zero intervals are not penalised.
Matching is a local correspondence heuristic, not a persistent branch identity.
The penalty acts on physical positions; it also resists genuine contraction.
"""
import copy
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import roots_legendre
from scipy.optimize import linear_sum_assignment

GAMMA = 10.0
WIDTH = 10
DEPTH = 4
TIMESTEPS = 10000
MAX_DT = 0.01
MIN_DT = 1e-18
REACTION_LIMIT = 0.01
INITIAL_EPOCHS = 200
EPOCHS = 20
LBFGS_STEPS = 100
LEARNING_RATE = 0.001
SEED = 42
DTYPE = torch.float64
SAVE_RESULTS = True
KNOT_REGULARISATION = 0.0  # Lambda; set to zero to disable temporal regularisation.


class PINN(nn.Module):
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.hidden = nn.ModuleList([
            nn.Linear(1 if i == 0 else width, width, dtype=DTYPE)
            for i in range(depth)
        ])
        self.output = nn.Linear(width, 1, dtype=DTYPE)
        with torch.no_grad():
            self.hidden[0].weight.fill_(1.)
            self.hidden[0].bias.copy_(-torch.linspace(-.95, .95, width, dtype=DTYPE))
            for layer in self.hidden[1:]:
                layer.weight.copy_(torch.eye(width, dtype=DTYPE)
                                   + .03*torch.randn(width, width, dtype=DTYPE))
                layer.bias.uniform_(-.05, .05)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = torch.relu(layer(h))
        return (1-x.square()) * self.output(h)


def unique_sorted(values):
    """Sort endpoints without removing their parameter derivatives."""
    result = []
    for value in sorted(values, key=lambda v: float(v.detach())):
        if not result or float(value.detach()) != float(result[-1].detach()):
            result.append(value)
    return result


def affine_partition(model):
    """Return affine pieces of the raw network, roots and zero intervals.

    A piece is (left, right, slope, intercept). Root records are
    (layer, neuron, location_tensor); a deep neuron may have several roots.
    """
    p = next(model.parameters())
    pieces = [(p.new_tensor(-1.), p.new_tensor(1.), p.new_ones(1), p.new_zeros(1))]
    roots, zero_intervals = {}, []
    for layer_id, layer in enumerate(model.hidden):
        new_pieces = []
        for left, right, a, b in pieces:
            a, b = layer.weight @ a, layer.weight @ b + layer.bias
            cuts = [left, right]
            for neuron in range(len(a)):
                if float(a[neuron].detach()) == 0:
                    if float(b[neuron].detach()) == 0:
                        zero_intervals.append((layer_id, neuron, left, right))
                    continue
                root = -b[neuron]/a[neuron]
                if float(left.detach()) <= float(root.detach()) <= float(right.detach()):
                    roots[(layer_id, neuron, float(root.detach()))] = (layer_id, neuron, root)
                    cuts.append(root)
            cuts = unique_sorted(cuts)
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                active = (a.detach()*(lo.detach()+hi.detach())/2+b.detach() > 0).to(a)
                new_pieces.append((lo, hi, a*active, b*active))
            if len(new_pieces) > 10000:
                raise RuntimeError('More than 10000 affine pieces: reduce network size.')
        pieces = new_pieces
    pieces = [(lo, hi, (model.output.weight @ a).squeeze(),
               (model.output.weight @ b+model.output.bias).squeeze())
              for lo, hi, a, b in pieces]
    return pieces, [roots[key] for key in sorted(roots)], zero_intervals


def quadrature(pieces, previous_cuts=(), order=10):
    """Split at current/previous knots; ten Gauss points integrate degree 18."""
    cuts = [pieces[0][0]] + [piece[1] for piece in pieces]
    cuts += list(previous_cuts)
    cuts += list(torch.linspace(-1, 1, 33, dtype=cuts[0].dtype, device=cuts[0].device))
    edges = torch.stack(unique_sorted(cuts))
    mid, half = (edges[1:]+edges[:-1])/2, (edges[1:]-edges[:-1])/2
    nodes, weights = roots_legendre(order)
    x = (mid[:, None]+half[:, None]*edges.new_tensor(nodes)).reshape(-1, 1)
    w = (half[:, None]*edges.new_tensor(weights)).reshape(-1, 1)
    return x.requires_grad_(True), w


def derivative(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def knot_displacement(roots, previous_knots):
    """Sum squared motion of matched roots, preserving current-root gradients.

    previous_knots is a detached (N, 3) tensor: layer, neuron, position.
    Assignment uses detached coordinates; only the selected distances are
    differentiated. The objective is piecewise smooth as assignments change.
    Returns (squared_motion, matched, unmatched_current, unmatched_previous).
    """
    current, previous = {}, {}
    for layer, neuron, position in roots:
        current.setdefault((layer, neuron), []).append(position)
    for layer, neuron, position in previous_knots.detach():
        previous.setdefault((int(layer), int(neuron)), []).append(position)
    motion = previous_knots.new_zeros(())
    matched = 0
    for key in sorted(current.keys() & previous.keys()):
        new = torch.stack(current[key])
        old = torch.stack(previous[key]).to(new).detach()
        cost = (new[:, None].detach()-old[None, :]).square().cpu().numpy()
        rows, columns = linear_sum_assignment(cost)
        motion = motion + (new[rows]-old[columns]).square().sum()
        matched += len(rows)
    return motion, matched, len(roots)-matched, len(previous_knots)-matched


def timestep_loss(model, previous, previous_cuts, dt, scale, previous_knots=None):
    pieces, roots, _ = affine_partition(model)
    x, w = quadrature(pieces, previous_cuts)
    u, old = model(x), previous(x)
    # Complete the square and scale to keep the objective usable near blow-up.
    # Do not detach old: its spatial dependence matters at moving quadrature nodes.
    physical = (w*(dt*(derivative(u, x)/scale).square()
                   + ((u-old-dt*old.pow(3))/scale).square())).sum()/2
    if KNOT_REGULARISATION == 0 or previous_knots is None:
        return physical
    motion, _, _, _ = knot_displacement(roots, previous_knots)
    # Apply the same dt/scale² scaling as the physical Ritz functional.
    return physical + KNOT_REGULARISATION*motion/(2*scale**2)


def train(model, objective, epochs):
    adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    def closure():
        model.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss.')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient.')
        return loss
    for _ in range(epochs):
        closure()
        adam.step()
    if LBFGS_STEPS:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_STEPS,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn='strong_wolfe')
        lbfgs.step(closure)
    loss = closure()
    gradient = max(float(p.grad.abs().max()) for p in model.parameters() if p.grad is not None)
    return float(loss.detach()), gradient


@torch.no_grad()
def peak_amplitude(pieces):
    """Maximum of |(1-x²)(a*x+b)|, including interior stationary points."""
    peak, location = 0., -1.
    for left, right, a, b in pieces:
        lo, hi, a, b = map(float, (left, right, a, b))
        if not np.isfinite([lo, hi, a, b]).all():
            raise FloatingPointError('Nonfinite network coefficients.')
        candidates = [lo, hi] + [float(z.real) for z in np.roots([-3*a, -2*b, a])
                                 if np.isreal(z) and lo < z.real < hi]
        for x in candidates:
            value = abs((1-x*x)*(a*x+b))
            if value > peak:
                peak, location = value, x
    return peak, location


def snapshot(model):
    with torch.no_grad():
        pieces, roots, zero_intervals = affine_partition(model)
        root_tensor = torch.tensor([(i, j, float(k)) for i, j, k in roots], dtype=DTYPE).reshape(-1, 3)
        zero_tensor = torch.tensor([(i, j, float(a), float(b)) for i, j, a, b in zero_intervals], dtype=DTYPE).reshape(-1, 4)
        output = []
        for before, after in zip(pieces[:-1], pieces[1:]):
            x, jump = float(after[0]), float(after[2]-before[2])
            if abs(jump) > 1e-10:
                output.append((x, jump, (1-x*x)*jump))
        cuts = torch.stack([pieces[0][0]]+[p[1] for p in pieces]).detach()
        return root_tensor, zero_tensor, torch.tensor(output, dtype=DTYPE).reshape(-1, 3), cuts, peak_amplitude(pieces)


def solve():
    if not np.isfinite(KNOT_REGULARISATION) or KNOT_REGULARISATION < 0:
        raise ValueError('KNOT_REGULARISATION must be finite and nonnegative.')
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = PINN()
    initial = lambda x: GAMMA*torch.cos(torch.pi*x/2)
    x_plot = torch.linspace(-1, 1, 1001, dtype=DTYPE).reshape(-1, 1)
    def initial_loss():
        x, w = quadrature(affine_partition(model)[0])
        u = model(x)
        target_derivative = -GAMMA*torch.pi/2*torch.sin(torch.pi*x/2)
        return (w*((u-initial(x)).square()
                   + (derivative(u, x)-target_derivative).square())).sum()
    loss, gradient = train(model, initial_loss, INITIAL_EPOCHS)
    times, solutions, knots, zeros, output_knots, diagnostics = [], [], [], [], [], []
    def record(time, dt, old_peak, loss, gradient):
        roots, flat, output, cuts, (peak, peak_x) = snapshot(model)
        if dt > 0:
            current = [(int(row[0]), int(row[1]), row[2]) for row in roots]
            motion, matched, born, lost = knot_displacement(current, knots[-1])
            tikhonov = float(motion)/(2*dt)
            scaled_penalty = KNOT_REGULARISATION*float(motion)/(2*max(1., old_peak)**2)
        else:
            tikhonov, scaled_penalty, matched, born, lost = 0., 0., 0, 0, 0
        times.append(time)
        solutions.append(model(x_plot).detach())
        knots.append(roots)
        zeros.append(flat)
        output_knots.append(output)
        diagnostics.append((len(times)-1, time, loss, gradient, peak, dt,
                            dt*old_peak**2, (1/peak)**2 if peak else np.nan, peak_x,
                            tikhonov, scaled_penalty, matched, born, lost))
        return cuts, peak
    cuts, peak = record(0., 0., abs(GAMMA), loss, gradient)
    status = 'Completed requested steps.'
    try:
        for step in range(1, TIMESTEPS+1):
            previous = initial if step == 1 else copy.deepcopy(model).requires_grad_(False)
            previous_cuts = () if step == 1 else cuts
            old_peak = abs(GAMMA) if step == 1 else peak
            dt = MAX_DT if old_peak == 0 else min(MAX_DT, (REACTION_LIMIT/old_peak)/old_peak)
            if dt < MIN_DT or times[-1]+dt == times[-1]:
                status = 'Stopped at minimum timestep; this is not a certified blow-up time.'
                break
            previous_knots = knots[-1].detach()
            loss, gradient = train(model, lambda: timestep_loss(
                model, previous, previous_cuts, dt, max(1., old_peak), previous_knots), EPOCHS)
            proposed_peak = peak_amplitude(affine_partition(model)[0])[0]
            if dt*proposed_peak**2 > 2*REACTION_LIMIT:
                raise RuntimeError('Step grew too far; reduce REACTION_LIMIT or improve training.')
            cuts, peak = record(times[-1]+dt, dt, old_peak, loss, gradient)
            print(f'{step}: t={times[-1]:.10g}, dt={dt:.3e}, peak={peak:.6g}, gradient={gradient:.2e}, '
                  f'knot penalty={diagnostics[-1][10]:.2e}, unmatched={diagnostics[-1][12:14]}', flush=True)
    except (KeyboardInterrupt, FloatingPointError, RuntimeError) as error:
        status = f'Stopped: {type(error).__name__}: {error}'
    history = dict(x=x_plot.squeeze(), times=torch.tensor(times, dtype=DTYPE),
                   u=torch.stack(solutions).squeeze(-1), knots=knots, zero_intervals=zeros,
                   output_knots=output_knots, diagnostics=torch.tensor(diagnostics, dtype=DTYPE),
                   exact_u0=initial(x_plot).detach().squeeze(), status=status,
                   knot_regularisation=KNOT_REGULARISATION)
    print(status)
    return history


def plot_results(history):
    import matplotlib.pyplot as plt
    x, times, u = history['x'], history['times'], history['u']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i in np.unique(np.linspace(0, len(times)-1, min(8, len(times)), dtype=int)):
        axes[0].plot(x, u[i], label=f't={times[i]:.6g}')
    axes[0].set(xlabel='x', ylabel='u(x,t)')
    axes[0].legend(fontsize=7)
    for layer in range(DEPTH):
        for time, roots in zip(times, history['knots']):
            k = roots[roots[:, 0] == layer, 2]
            axes[1].scatter(k, torch.full_like(k, time), s=3, color=f'C{layer % 10}')
        axes[1].scatter([], [], s=10, color=f'C{layer % 10}', label=f'Layer {layer}')
    axes[1].set(xlabel='Knot location x', ylabel='t', xlim=(-1, 1))
    axes[1].legend()
    axes[2].plot(times, history['diagnostics'][:, 4])
    if torch.any(history['diagnostics'][:, 4] > 0):
        axes[2].set_yscale('log')
    axes[2].set(xlabel='t', ylabel='Peak |u|')
    fig.tight_layout()
    return fig


def save_results(history):
    """Optional export, compatible with the existing saved-results plotting script."""
    out = Path(__file__).resolve().parent / ('nonlinear_heat_results_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    torch.save(history, out/'history.pt')
    np.savez_compressed(out/'solution_history.npz', **{k: history[k].numpy() for k in ('x', 'times', 'u', 'exact_u0')})
    for key, filename, columns in (
        ('knots', 'hidden_roots.csv', 'layer,neuron,x'),
        ('zero_intervals', 'zero_intervals.csv', 'layer,neuron,left,right'),
        ('output_knots', 'output_knots.csv', 'x,raw_slope_jump,u_slope_jump'),
    ):
        with (out/filename).open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'time']+columns.split(','))
            for step, (time, values) in enumerate(zip(history['times'], history[key])):
                for row in values.tolist():
                    if key != 'output_knots':
                        row[:2] = [int(v) for v in row[:2]]
                    writer.writerow([step, float(time)]+row)
    np.savetxt(out/'diagnostics.csv', history['diagnostics'].numpy(), delimiter=',',
               header='step,time,objective,parameter_grad_inf,max_abs_u,dt,dt_old_peak_squared,inverse_peak_squared,peak_x,knot_tikhonov,knot_penalty_scaled,knots_matched,knots_unmatched_current,knots_unmatched_previous', comments='')
    (out/'status.txt').write_text(history['status'])
    (out/'regularisation.txt').write_text(
        f"KNOT_REGULARISATION = {history['knot_regularisation']}\n"
        'Temporal penalty on physical knot locations; minimum-distance matching within each neuron.\n'
        'Unmatched roots and zero intervals are excluded.\n')
    return out


if __name__ == '__main__':
    history = solve()
    figure = plot_results(history)
    if SAVE_RESULTS:
        folder = save_results(history)
        figure.savefig(folder/'results.png', dpi=180)
        print(f'Saved: {folder}')
    import matplotlib.pyplot as plt
    plt.show()
