import torch
from torch.func import functional_call


def make_functional_model(model):
    named_params = dict(model.named_parameters())

    if not named_params:
        raise ValueError("The model has no trainable parameters.")

    names = list(named_params.keys())
    shapes = [parameter.shape for parameter in named_params.values()]
    sizes = [parameter.numel() for parameter in named_params.values()]
    params0 = torch.cat([
        parameter.detach().flatten()
        for parameter in named_params.values()
    ])

    buffers = {
        name: buffer.detach()
        for name, buffer in model.named_buffers()
    }

    def unflatten(flat_params):
        chunks = torch.split(flat_params, sizes)

        return {
            name: chunk.reshape(shape)
            for name, chunk, shape in zip(names, chunks, shapes)
        }

    def fmodel(flat_params, points):
        parameters = unflatten(flat_params)
        model_state = {**buffers, **parameters}

        return functional_call(
            model,
            model_state,
            (points,),
        )

    return fmodel, params0


def compute_hessian_statistics(
    model,
    loss_function,
    loss_args=(),
    loss_kwargs=None,
    relative_eig_tol=1e-8,
    absolute_eig_tol=0.0,
    max_hessian_params=5000,
):
    """
    Compute the full parameter Hessian and its spectral statistics.

    The loss function must have the form

        loss_function(model, *loss_args, **loss_kwargs)

    and must return a scalar tensor.
    """

    if loss_kwargs is None:
        loss_kwargs = {}

    fmodel, params0 = make_functional_model(model)
    number_parameters = params0.numel()

    if number_parameters > max_hessian_params:
        nan = float("nan")

        return {
            "computed": False,
            "skip_reason": (
                f"{number_parameters} parameters exceeds "
                f"max_hessian_params={max_hessian_params}"
            ),
            "number_parameters": number_parameters,
            "condition_number_positive": nan,
            "largest_positive_eigenvalue": nan,
            "smallest_positive_eigenvalue": nan,
            "most_negative_eigenvalue": nan,
            "negative_count": 0,
            "negative_fraction": nan,
            "near_zero_count": 0,
            "near_zero_fraction": nan,
            "effective_rank": nan,
            "nonfinite_count": 0,
            "eigenvalue_tolerance": nan,
            "eigenvalues": torch.empty(0),
        }

    def loss_from_params(flat_params):
        functional_model = lambda points: fmodel(flat_params, points)

        loss = loss_function(
            functional_model,
            *loss_args,
            **loss_kwargs,
        )

        if loss.numel() != 1:
            raise ValueError(
                "The loss function must return a scalar tensor."
            )

        return loss.squeeze()

    hessian = torch.autograd.functional.hessian(
        loss_from_params,
        params0,
        create_graph=False,
        strict=False,
        vectorize=False,
    )

    # Remove small numerical asymmetry.
    hessian = 0.5 * (hessian + hessian.T)

    eigenvalues = torch.linalg.eigvalsh(hessian)
    finite_mask = torch.isfinite(eigenvalues)
    finite_eigenvalues = eigenvalues[finite_mask]

    nonfinite_count = int((~finite_mask).sum().item())

    if finite_eigenvalues.numel() == 0:
        raise RuntimeError("The Hessian has no finite eigenvalues.")

    spectral_scale = finite_eigenvalues.abs().max()

    tolerance = max(
        absolute_eig_tol,
        relative_eig_tol * spectral_scale.item(),
    )

    positive_mask = finite_eigenvalues > tolerance
    negative_mask = finite_eigenvalues < -tolerance
    near_zero_mask = finite_eigenvalues.abs() <= tolerance

    positive_eigenvalues = finite_eigenvalues[positive_mask]
    negative_eigenvalues = finite_eigenvalues[negative_mask]

    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())
    near_zero_count = int(near_zero_mask.sum().item())
    finite_count = finite_eigenvalues.numel()

    if positive_count > 0:
        largest_positive = positive_eigenvalues.max()
        smallest_positive = positive_eigenvalues.min()
        positive_condition_number = (
            largest_positive / smallest_positive
        )
    else:
        largest_positive = torch.tensor(
            float("nan"),
            device=params0.device,
            dtype=params0.dtype,
        )
        smallest_positive = largest_positive.clone()
        positive_condition_number = torch.tensor(
            float("inf"),
            device=params0.device,
            dtype=params0.dtype,
        )

    if negative_count > 0:
        most_negative = negative_eigenvalues.min()
    else:
        most_negative = torch.tensor(
            float("nan"),
            device=params0.device,
            dtype=params0.dtype,
        )

    # Entropy-based effective rank of the resolved spectrum.
    resolved_magnitudes = finite_eigenvalues.abs()[~near_zero_mask]

    if resolved_magnitudes.numel() > 0:
        probabilities = (
            resolved_magnitudes / resolved_magnitudes.sum()
        )
        spectral_entropy = -torch.sum(
            probabilities * torch.log(probabilities)
        )
        effective_rank = torch.exp(spectral_entropy)
    else:
        effective_rank = torch.tensor(
            0.0,
            device=params0.device,
            dtype=params0.dtype,
        )

    return {
        "computed": True,
        "skip_reason": None,
        "number_parameters": number_parameters,
        "condition_number_positive": (
            positive_condition_number.detach().cpu().item()
        ),
        "largest_positive_eigenvalue": (
            largest_positive.detach().cpu().item()
        ),
        "smallest_positive_eigenvalue": (
            smallest_positive.detach().cpu().item()
        ),
        "most_negative_eigenvalue": (
            most_negative.detach().cpu().item()
        ),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "negative_fraction": negative_count / finite_count,
        "near_zero_count": near_zero_count,
        "near_zero_fraction": near_zero_count / finite_count,
        "effective_rank": effective_rank.detach().cpu().item(),
        "nonfinite_count": nonfinite_count,
        "eigenvalue_tolerance": tolerance,
        "eigenvalues": eigenvalues.detach().cpu(),
    }