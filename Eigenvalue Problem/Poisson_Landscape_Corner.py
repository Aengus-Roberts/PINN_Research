from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DOMAIN_AREA = 3.0


class PINN(nn.Module):
    """Fully connected trial function used by the Deep Ritz method."""

    def __init__(self, width):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(7):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProjectedModel(nn.Module):
    """A normalised model with projections onto earlier modes removed."""

    def __init__(self, base_model, previous_modes, coefficients, scale):
        super().__init__()
        self.base_model = base_model
        self.previous_modes = nn.ModuleList(previous_modes)
        self.register_buffer("coefficients", coefficients.detach().clone())
        self.register_buffer(
            "scale",
            torch.as_tensor(scale, dtype=coefficients.dtype, device=coefficients.device),
        )

    def forward(self, x):
        value = self.base_model(x)
        for coefficient, mode in zip(self.coefficients, self.previous_modes):
            value = value - coefficient * mode(x)
        return self.scale * value


@dataclass
class IterationConfig:
    max_modes: int = 4
    max_inverse_iterations: int = 12
    min_inverse_iterations: int = 2
    eigenvalue_tolerance: float = 1.0e-3
    function_tolerance: float = 1.0e-3
    max_relative_residual: float = 5.0e-2
    max_boundary_error: float = 5.0e-2
    max_orthogonality_error: float = 2.0e-2
    min_deflated_landscape_ratio: float = 1.0e-3
    min_projection_norm: float = 1.0e-10


@dataclass
class ValidationMetrics:
    eigenvalue: float
    relative_pde_residual: float
    relative_boundary_error: float
    orthogonality_error: float


@dataclass
class EigenpairResult:
    mode_number: int
    model: nn.Module
    eigenvalue: float
    inverse_iterations: int
    converged: bool
    accepted: bool
    rejection_reason: str | None
    relative_eigenvalue_change: float
    function_change: float
    relative_pde_residual: float
    relative_boundary_error: float
    orthogonality_error: float
    deflated_landscape_ratio: float
    history: list[dict[str, float]] = field(default_factory=list)


def compute_loss(
    model,
    x,
    x_b,
    x_s,
    source_values=None,
    w=None,
    bc_weight=10.0,
):
    """Deep Ritz energy for ``-Delta u = source`` with soft Dirichlet data."""
    x_differentiable = x.detach().clone().requires_grad_(True)
    u = model(x_differentiable).view(-1, 1)
    gradient = torch.autograd.grad(
        u,
        x_differentiable,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    if source_values is None:
        source_values = torch.ones_like(u)
    else:
        source_values = source_values.detach().reshape_as(u)

    integrand = 0.5 * torch.sum(gradient**2, dim=1, keepdim=True) - source_values * u
    if w is None:
        interior_loss = torch.mean(integrand)
    else:
        interior_loss = torch.sum(w.reshape_as(integrand) * integrand)

    square_boundary_loss = torch.mean(model(x_b).view(-1, 1) ** 2)
    slit_boundary_loss = torch.mean(model(x_s).view(-1, 1) ** 2)
    return interior_loss + bc_weight * (square_boundary_loss + slit_boundary_loss)


def sample_interior(number_of_points, device="cpu"):
    r"""Sample uniformly from ``[-1,1]^2 \ ((0,1] x [-1,0))``."""
    points = []
    remaining = number_of_points

    while remaining > 0:
        candidate = 2 * torch.rand(2 * remaining, 2, device=device) - 1
        keep = ~((candidate[:, 0] > 0.0) & (candidate[:, 1] < 0.0))
        candidate = candidate[keep][:remaining]
        points.append(candidate)
        remaining -= candidate.shape[0]

    return torch.cat(points, dim=0)


def sample_square_boundary(number_of_points, device="cpu"):
    """Sample the four exposed pieces of the outer boundary."""
    # The edge lengths are 2, 2, 1 and 1.  Proportional counts therefore
    # approximate the arclength measure used by the boundary penalty.
    edge_lengths = torch.tensor([2.0, 2.0, 1.0, 1.0])
    exact_counts = number_of_points * edge_lengths / edge_lengths.sum()
    counts = torch.floor(exact_counts).to(torch.int64).tolist()
    remainders = torch.argsort(exact_counts - torch.floor(exact_counts), descending=True)
    for index in remainders[: number_of_points - sum(counts)]:
        counts[index.item()] += 1

    s_left = 2 * torch.rand(counts[0], 1, device=device) - 1
    s_top = 2 * torch.rand(counts[1], 1, device=device) - 1
    s_right = torch.rand(counts[2], 1, device=device)
    s_bottom = torch.rand(counts[3], 1, device=device) - 1

    left = torch.cat([-torch.ones_like(s_left), s_left], dim=1)
    top = torch.cat([s_top, torch.ones_like(s_top)], dim=1)
    right_top = torch.cat([torch.ones_like(s_right), s_right], dim=1)
    bottom_left = torch.cat([s_bottom, -torch.ones_like(s_bottom)], dim=1)
    return torch.cat([left, top, right_top, bottom_left], dim=0)


def sample_slit_boundary(number_of_points, device="cpu"):
    """Sample ``(0,1] x {0}`` and ``{0} x [-1,0)``."""
    number_horizontal = number_of_points // 2
    x_horizontal = torch.rand(number_horizontal, 1, device=device)
    horizontal = torch.cat([x_horizontal, torch.zeros_like(x_horizontal)], dim=1)

    y_vertical = torch.rand(number_of_points - number_horizontal, 1, device=device) - 1
    vertical = torch.cat([torch.zeros_like(y_vertical), y_vertical], dim=1)
    return torch.cat([horizontal, vertical], dim=0)


def uniform_interior_weights(x):
    """Monte Carlo weights for integrals over the L-shaped domain."""
    return torch.full(
        (x.shape[0], 1),
        DOMAIN_AREA / x.shape[0],
        dtype=x.dtype,
        device=x.device,
    )


def weighted_inner_product(left, right, weights):
    return torch.sum(weights * left.reshape(-1, 1) * right.reshape(-1, 1))


def weighted_norm(values, weights):
    return torch.sqrt(torch.clamp(weighted_inner_product(values, values, weights), min=0.0))


def train_poisson_drm(
    width,
    x,
    x_b,
    x_s,
    source_values=None,
    weights=None,
    epochs=20_000,
    lr=0.01,
    bc_weight=10.0,
    initial_state_dict=None,
    description="DRM solve",
):
    """Solve a Poisson problem with the Deep Ritz energy."""
    model = PINN(width).to(x.device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimiser.zero_grad()
        loss = compute_loss(
            model,
            x,
            x_b,
            x_s,
            source_values=source_values,
            w=weights,
            bc_weight=bc_weight,
        )
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"{description}, epoch {epoch}, loss: {loss.item():.6f}")

    return model, losses


def train_PINN(
    N,
    x,
    x_b,
    x_s,
    weights=None,
    epochs=20_000,
    lr=0.01,
    bc_weight=10.0,
):
    """Backward-compatible landscape solver for ``-Delta u = 1``."""
    return train_poisson_drm(
        width=N,
        x=x,
        x_b=x_b,
        x_s=x_s,
        source_values=None,
        weights=weights,
        epochs=epochs,
        lr=lr,
        bc_weight=bc_weight,
        description="Landscape solve",
    )


def project_and_normalise_model(
    base_model,
    previous_modes,
    x,
    weights,
    reference_values=None,
):
    """Project with the discrete Gram matrix, then orient and normalise."""
    with torch.no_grad():
        raw_values = base_model(x).reshape(-1, 1)
        if previous_modes:
            mode_values = torch.cat(
                [mode(x).reshape(-1, 1) for mode in previous_modes],
                dim=1,
            )
            weighted_modes = weights * mode_values
            gram_matrix = mode_values.T @ weighted_modes
            right_hand_side = mode_values.T @ (weights * raw_values)
            coefficients = torch.linalg.solve(gram_matrix, right_hand_side).reshape(-1)
            projected_values = raw_values - mode_values @ coefficients.reshape(-1, 1)
        else:
            coefficients = torch.empty(0, dtype=x.dtype, device=x.device)
            projected_values = raw_values

        projection_norm = weighted_norm(projected_values, weights)
        if not torch.isfinite(projection_norm) or projection_norm.item() == 0.0:
            raise RuntimeError("Projection produced a zero or non-finite function.")

        orientation = 1.0
        normalised_values = projected_values / projection_norm
        if reference_values is not None:
            correlation = weighted_inner_product(normalised_values, reference_values, weights)
            if correlation.item() < 0.0:
                orientation = -1.0
                normalised_values = -normalised_values

    projected_model = ProjectedModel(
        base_model=base_model,
        previous_modes=previous_modes,
        coefficients=coefficients,
        scale=orientation / projection_norm,
    )
    return projected_model, normalised_values.detach(), projection_norm.item()


def rayleigh_quotient(model, x, weights):
    x_differentiable = x.detach().clone().requires_grad_(True)
    values = model(x_differentiable).reshape(-1, 1)
    gradient = torch.autograd.grad(
        values,
        x_differentiable,
        grad_outputs=torch.ones_like(values),
        create_graph=False,
    )[0]
    numerator = torch.sum(weights * torch.sum(gradient**2, dim=1, keepdim=True))
    denominator = weighted_inner_product(values, values, weights)
    return (numerator / denominator).item()


def inverse_iteration(
    width,
    initial_values,
    previous_modes,
    x,
    x_b,
    x_s,
    weights,
    solve_epochs,
    lr,
    bc_weight,
    config,
):
    """Apply projected inverse iteration using repeated DRM Poisson solves."""
    current_values = initial_values.detach()
    current_values = current_values / weighted_norm(current_values, weights)
    previous_eigenvalue = None
    relative_eigenvalue_change = float("inf")
    function_change = float("inf")
    history = []
    candidate_model = None
    warm_start = None
    converged = False

    for iteration in range(1, config.max_inverse_iterations + 1):
        raw_model, losses = train_poisson_drm(
            width=width,
            x=x,
            x_b=x_b,
            x_s=x_s,
            source_values=current_values,
            weights=None,
            epochs=solve_epochs,
            lr=lr,
            bc_weight=bc_weight,
            initial_state_dict=warm_start,
            description=f"Inverse iteration {iteration}",
        )
        warm_start = {
            name: value.detach().clone()
            for name, value in raw_model.state_dict().items()
        }

        candidate_model, candidate_values, projection_norm = project_and_normalise_model(
            raw_model,
            previous_modes,
            x,
            weights,
            reference_values=current_values,
        )
        if projection_norm < config.min_projection_norm:
            raise RuntimeError("The projected inverse iterate is numerically zero.")

        eigenvalue = rayleigh_quotient(candidate_model, x, weights)
        difference = candidate_values - current_values
        function_change = weighted_norm(difference, weights).item()
        if previous_eigenvalue is not None:
            relative_eigenvalue_change = abs(eigenvalue - previous_eigenvalue) / abs(eigenvalue)

        history.append(
            {
                "iteration": float(iteration),
                "eigenvalue": eigenvalue,
                "relative_eigenvalue_change": relative_eigenvalue_change,
                "function_change": function_change,
                "projection_norm": projection_norm,
                "final_training_loss": losses[-1],
            }
        )
        print(
            f"Inverse iteration {iteration}: lambda={eigenvalue:.8f}, "
            f"delta_lambda={relative_eigenvalue_change:.3e}, "
            f"delta_phi={function_change:.3e}"
        )

        if (
            iteration >= config.min_inverse_iterations
            and relative_eigenvalue_change < config.eigenvalue_tolerance
            and function_change < config.function_tolerance
        ):
            converged = True
            break

        previous_eigenvalue = eigenvalue
        current_values = candidate_values

    return (
        candidate_model,
        iteration,
        converged,
        relative_eigenvalue_change,
        function_change,
        history,
    )


def validation_metrics(model, previous_modes, x, x_b, x_s, weights):
    """Evaluate the Rayleigh quotient and errors on independent points."""
    x_differentiable = x.detach().clone().requires_grad_(True)
    values = model(x_differentiable).reshape(-1, 1)
    gradient = torch.autograd.grad(
        values,
        x_differentiable,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
    )[0]

    energy = torch.sum(weights * torch.sum(gradient**2, dim=1, keepdim=True))
    mass = weighted_inner_product(values, values, weights)
    eigenvalue = energy / mass

    laplacian = torch.zeros_like(values)
    number_of_dimensions = x_differentiable.shape[1]
    for dimension in range(number_of_dimensions):
        second_derivative = torch.autograd.grad(
            gradient[:, dimension : dimension + 1],
            x_differentiable,
            grad_outputs=torch.ones_like(values),
            retain_graph=dimension < number_of_dimensions - 1,
            create_graph=False,
        )[0][:, dimension : dimension + 1]
        laplacian = laplacian + second_derivative

    residual = -laplacian - eigenvalue.detach() * values.detach()
    residual_norm = weighted_norm(residual, weights)
    relative_pde_residual = residual_norm / (
        torch.abs(eigenvalue.detach()) * torch.sqrt(mass.detach())
        + torch.finfo(x.dtype).eps
    )

    with torch.no_grad():
        interior_rms = torch.sqrt(torch.mean(values.detach() ** 2))
        outer_boundary_rms = torch.sqrt(torch.mean(model(x_b) ** 2))
        slit_boundary_rms = torch.sqrt(torch.mean(model(x_s) ** 2))
        boundary_rms = torch.sqrt(
            0.5 * (outer_boundary_rms**2 + slit_boundary_rms**2)
        )
        relative_boundary_error = boundary_rms / (
            interior_rms + torch.finfo(x.dtype).eps
        )

        orthogonality_error = 0.0
        candidate_values = values.detach()
        candidate_norm = weighted_norm(candidate_values, weights)
        for previous_mode in previous_modes:
            previous_values = previous_mode(x).reshape(-1, 1)
            previous_norm = weighted_norm(previous_values, weights)
            correlation = torch.abs(
                weighted_inner_product(candidate_values, previous_values, weights)
                / (candidate_norm * previous_norm)
            ).item()
            orthogonality_error = max(orthogonality_error, correlation)

    return ValidationMetrics(
        eigenvalue=eigenvalue.item(),
        relative_pde_residual=relative_pde_residual.item(),
        relative_boundary_error=relative_boundary_error.item(),
        orthogonality_error=orthogonality_error,
    )


def find_landscape_eigenpairs(
    width,
    x,
    x_b,
    x_s,
    x_validation,
    x_b_validation,
    x_s_validation,
    landscape_epochs,
    solve_epochs,
    lr=0.01,
    bc_weight=10.0,
    config=None,
):
    """Find the eigenmodes visible in the constant-source landscape function."""
    if config is None:
        config = IterationConfig()

    training_weights = uniform_interior_weights(x)
    validation_weights = uniform_interior_weights(x_validation)
    landscape_model, landscape_losses = train_poisson_drm(
        width=width,
        x=x,
        x_b=x_b,
        x_s=x_s,
        source_values=None,
        weights=None,
        epochs=landscape_epochs,
        lr=lr,
        bc_weight=bc_weight,
        description="Landscape solve",
    )

    with torch.no_grad():
        landscape_values = landscape_model(x).reshape(-1, 1)
        landscape_norm = weighted_norm(landscape_values, training_weights).item()

    accepted_modes = []
    results = []
    for mode_number in range(1, config.max_modes + 1):
        try:
            _, deflated_values, deflated_norm = project_and_normalise_model(
                landscape_model,
                accepted_modes,
                x,
                training_weights,
            )
        except RuntimeError as error:
            print(f"Stopping before mode {mode_number}: {error}")
            break
        deflated_ratio = deflated_norm / landscape_norm
        if deflated_ratio < config.min_deflated_landscape_ratio:
            print(
                "Stopping: the remaining landscape has relative norm "
                f"{deflated_ratio:.3e}."
            )
            break

        print(
            f"\nMode {mode_number}: deflated landscape ratio={deflated_ratio:.3e}"
        )
        try:
            (
                candidate_model,
                number_of_iterations,
                converged,
                eigenvalue_change,
                function_change,
                history,
            ) = inverse_iteration(
                width=width,
                initial_values=deflated_values,
                previous_modes=accepted_modes,
                x=x,
                x_b=x_b,
                x_s=x_s,
                weights=training_weights,
                solve_epochs=solve_epochs,
                lr=lr,
                bc_weight=bc_weight,
                config=config,
            )
        except RuntimeError as error:
            print(f"Stopping during mode {mode_number}: {error}")
            break

        metrics = validation_metrics(
            candidate_model,
            accepted_modes,
            x_validation,
            x_b_validation,
            x_s_validation,
            validation_weights,
        )

        rejection_reasons = []
        if not converged:
            rejection_reasons.append("inverse iteration did not converge")
        if metrics.relative_pde_residual > config.max_relative_residual:
            rejection_reasons.append("PDE residual is too large")
        if metrics.relative_boundary_error > config.max_boundary_error:
            rejection_reasons.append("boundary error is too large")
        if metrics.orthogonality_error > config.max_orthogonality_error:
            rejection_reasons.append("orthogonality error is too large")

        accepted = not rejection_reasons
        rejection_reason = "; ".join(rejection_reasons) if rejection_reasons else None
        result = EigenpairResult(
            mode_number=mode_number,
            model=candidate_model,
            eigenvalue=metrics.eigenvalue,
            inverse_iterations=number_of_iterations,
            converged=converged,
            accepted=accepted,
            rejection_reason=rejection_reason,
            relative_eigenvalue_change=eigenvalue_change,
            function_change=function_change,
            relative_pde_residual=metrics.relative_pde_residual,
            relative_boundary_error=metrics.relative_boundary_error,
            orthogonality_error=metrics.orthogonality_error,
            deflated_landscape_ratio=deflated_ratio,
            history=history,
        )
        results.append(result)

        status = "accepted" if accepted else f"rejected ({rejection_reason})"
        print(
            f"Mode {mode_number} {status}: lambda={metrics.eigenvalue:.8f}, "
            f"residual={metrics.relative_pde_residual:.3e}, "
            f"boundary={metrics.relative_boundary_error:.3e}, "
            f"orthogonality={metrics.orthogonality_error:.3e}"
        )
        if not accepted:
            break
        accepted_modes.append(candidate_model)

    return landscape_model, landscape_losses, results


def plot_field(model, title, colour_bar_label, n_grid=200, device="cpu"):
    x = torch.linspace(-1.0, 1.0, n_grid, device=device)
    y = torch.linspace(-1.0, 1.0, n_grid, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    points = torch.cat(
        [x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)],
        dim=1,
    )

    with torch.no_grad():
        values = model(points).reshape(n_grid, n_grid).cpu().numpy()
    x_numpy = x_grid.cpu().numpy()
    y_numpy = y_grid.cpu().numpy()
    values[(x_numpy > 0.0) & (y_numpy < 0.0)] = np.nan

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(x_numpy, y_numpy, values, levels=50)
    plt.colorbar(contour, label=colour_bar_label)
    plt.plot([0, 1], [0, 0], "k-", linewidth=2, label="internal boundary")
    plt.plot([0, 0], [-1, 0], "k-", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_solution(model, n_grid=200, device="cpu"):
    """Backward-compatible landscape plot."""
    plot_field(
        model,
        title="DRM landscape function on the corner domain",
        colour_bar_label="u(x, y)",
        n_grid=n_grid,
        device=device,
    )


def main(
    N=20,
    EPOCHS=20_000,
    ITERATION_EPOCHS=None,
    max_modes=4,
    max_inverse_iterations=12,
    plot=True,
    seed=None,
    config=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ITERATION_EPOCHS is None:
        ITERATION_EPOCHS = EPOCHS

    x = sample_interior(1000, device=device)
    x_b = sample_square_boundary(200, device=device)
    x_s = sample_slit_boundary(100, device=device)
    x_validation = sample_interior(1000, device=device)
    x_b_validation = sample_square_boundary(400, device=device)
    x_s_validation = sample_slit_boundary(200, device=device)

    if config is None:
        config = IterationConfig(
            max_modes=max_modes,
            max_inverse_iterations=max_inverse_iterations,
        )
    landscape_model, landscape_losses, eigenpairs = find_landscape_eigenpairs(
        width=N,
        x=x,
        x_b=x_b,
        x_s=x_s,
        x_validation=x_validation,
        x_b_validation=x_b_validation,
        x_s_validation=x_s_validation,
        landscape_epochs=EPOCHS,
        solve_epochs=ITERATION_EPOCHS,
        lr=0.01,
        bc_weight=10.0,
        config=config,
    )

    if plot:
        plot_solution(landscape_model, n_grid=200, device=device)
        plt.figure(figsize=(8, 4))
        plt.plot(landscape_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Landscape loss, width={N}")
        plt.tight_layout()
        plt.show()

        for result in eigenpairs:
            if result.accepted:
                plot_field(
                    result.model,
                    title=(
                        f"Eigenfunction {result.mode_number}, "
                        f"lambda={result.eigenvalue:.6f}"
                    ),
                    colour_bar_label=f"phi_{result.mode_number}(x, y)",
                    n_grid=200,
                    device=device,
                )

    return landscape_model, eigenpairs, landscape_losses


if __name__ == "__main__":
    main(
        N=10,
        EPOCHS=5000,
        ITERATION_EPOCHS=1000,
        max_modes=4,
        max_inverse_iterations=12,
        plot=True,
        seed=0,
    )
