import argparse
import csv
import importlib.util
import json
import math
import random
import shutil
import sys
import tomllib
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_DIRECTORY = Path(__file__).resolve().parent


def load_module(module_name, module_path):
    specification = importlib.util.spec_from_file_location(module_name, module_path)

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


def get_dtype(dtype_name):
    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    try:
        return dtypes[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from error


def get_activation(activation_name):
    activations = {
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }

    try:
        return activations[activation_name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(activations))
        raise ValueError(
            f"Unknown activation '{activation_name}'. Available: {available}"
        ) from error


def make_optimizer(model, training_configuration):
    optimizer_name = training_configuration.get("optimizer", "adam").lower()
    learning_rate = training_configuration["learning_rate"]
    weight_decay = training_configuration.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=training_configuration.get("momentum", 0.0),
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def unpack_loss(loss_output):
    if not isinstance(loss_output, (tuple, list)) or len(loss_output) != 3:
        raise ValueError(
            "The loss function must return "
            "(total_loss, interior_loss, boundary_loss)."
        )

    total_loss, interior_loss, boundary_loss = loss_output

    for name, value in (
        ("total_loss", total_loss),
        ("interior_loss", interior_loss),
        ("boundary_loss", boundary_loss),
    ):
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor.")

    return total_loss, interior_loss, boundary_loss


def make_scalar_loss(loss_function):
    def scalar_loss(model, *loss_args, **loss_kwargs):
        total_loss, _, _ = unpack_loss(
            loss_function(model, *loss_args, **loss_kwargs)
        )
        return total_loss

    return scalar_loss


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_fieldnames(rows):
    return tuple(dict.fromkeys(
        key
        for row in rows
        for key in row
    ))


def get_sample_sizes(domain_configuration):
    """Return validated (interior, boundary) training sample-size pairs."""
    configured_sizes = domain_configuration.get("sample_sizes")

    if configured_sizes is None:
        try:
            configured_sizes = [{
                "n_interior": domain_configuration["n_interior"],
                "n_boundary": domain_configuration["n_boundary"],
            }]
        except KeyError as error:
            raise ValueError(
                "[domain] must define sample_sizes, or both n_interior "
                "and n_boundary."
            ) from error
    elif (
        "n_interior" in domain_configuration
        or "n_boundary" in domain_configuration
    ):
        raise ValueError(
            "[domain] cannot combine sample_sizes with n_interior or "
            "n_boundary."
        )

    if not isinstance(configured_sizes, list) or not configured_sizes:
        raise ValueError("domain.sample_sizes must be a non-empty list.")

    sample_sizes = []

    for index, sample_size in enumerate(configured_sizes):
        if not isinstance(sample_size, dict):
            raise ValueError(
                f"domain.sample_sizes[{index}] must be a table containing "
                "n_interior and n_boundary."
            )

        missing = {
            "n_interior",
            "n_boundary",
        } - sample_size.keys()

        if missing:
            raise ValueError(
                f"domain.sample_sizes[{index}] is missing {sorted(missing)}."
            )

        n_interior = sample_size["n_interior"]
        n_boundary = sample_size["n_boundary"]

        for name, value in (
            ("n_interior", n_interior),
            ("n_boundary", n_boundary),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"domain.sample_sizes[{index}].{name} must be a "
                    "positive integer."
                )

        sample_sizes.append((n_interior, n_boundary))

    if len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("domain.sample_sizes contains duplicate pairs.")

    return sample_sizes


def save_hessian_statistics(
    statistics,
    run_directory,
    step,
):
    eigenvalues = statistics.pop("eigenvalues")
    eigenvalue_filename = f"eigenvalues_step_{step:06d}.npy"
    statistics_filename = f"hessian_step_{step:06d}.json"

    np.save(
        run_directory / eigenvalue_filename,
        eigenvalues.detach().cpu().numpy(),
    )

    with (run_directory / statistics_filename).open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(json_safe(statistics), output_file, indent=2)

    return statistics, eigenvalue_filename, statistics_filename


def validate_configuration(configuration):
    required_sections = (
        "experiment",
        "architecture",
        "domain",
        "loss",
        "training",
        "analysis",
        "output",
    )

    missing = [
        section
        for section in required_sections
        if section not in configuration
    ]

    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")

    get_sample_sizes(configuration["domain"])

    epochs = configuration["training"]["epochs"]
    hessian_steps = configuration["analysis"]["hessian_steps"]

    invalid_steps = [
        step for step in hessian_steps
        if step < 0 or step > epochs
    ]

    if invalid_steps:
        raise ValueError(
            f"Hessian steps must lie in [0, {epochs}]: {invalid_steps}"
        )


def run_experiment(configuration, configuration_path):
    validate_configuration(configuration)

    experiment_configuration = configuration["experiment"]
    architecture_configuration = configuration["architecture"]
    domain_configuration = configuration["domain"]
    loss_configuration = configuration["loss"]
    training_configuration = configuration["training"]
    analysis_configuration = configuration["analysis"]
    output_configuration = configuration["output"]

    device = get_device(experiment_configuration.get("device", "auto"))
    dtype = get_dtype(experiment_configuration.get("dtype", "float64"))
    torch.set_default_dtype(dtype)
    torch.use_deterministic_algorithms(
        experiment_configuration.get("deterministic", True)
    )

    models_module = load_module(
        "architecture_models",
        PROJECT_DIRECTORY / "Models" / "FullyConnected.py",
    )
    analysis_module = load_module(
        "architecture_analysis",
        PROJECT_DIRECTORY / "Analysis_Functions.py",
    )
    domain_module = load_module(
        "architecture_domain",
        PROJECT_DIRECTORY
        / "2D_Domains"
        / f"{domain_configuration['module']}.py",
    )
    loss_module = load_module(
        "architecture_loss",
        PROJECT_DIRECTORY
        / "Loss_Functions"
        / f"{loss_configuration['module']}.py",
    )

    model_class = models_module.PINN
    sample_interior = getattr(
        domain_module,
        domain_configuration.get("interior_function", "sample_interior"),
    )
    sample_boundary = getattr(
        domain_module,
        domain_configuration["boundary_function"],
    )
    loss_function = getattr(loss_module, loss_configuration["function"])
    scalar_loss_function = make_scalar_loss(loss_function)
    loss_kwargs = dict(loss_configuration.get("kwargs", {}))

    output_root = Path(output_configuration["directory"])
    if not output_root.is_absolute():
        output_root = PROJECT_DIRECTORY / output_root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = experiment_configuration["name"]
    experiment_directory = output_root / f"{experiment_name}_{timestamp}"
    runs_directory = experiment_directory / "runs"
    runs_directory.mkdir(parents=True, exist_ok=False)

    shutil.copy2(
        configuration_path,
        experiment_directory / configuration_path.name,
    )

    metadata = {
        "experiment_name": experiment_name,
        "created": timestamp,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "configuration": configuration,
    }

    with (experiment_directory / "metadata.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(json_safe(metadata), output_file, indent=2)

    widths = architecture_configuration["widths"]
    depths = architecture_configuration["depths"]
    activation_names = architecture_configuration["activations"]
    repetitions = experiment_configuration.get("repetitions", 5)
    base_seed = experiment_configuration.get("base_seed", 0)
    epochs = training_configuration["epochs"]
    hessian_steps = set(analysis_configuration["hessian_steps"])
    sample_sizes = get_sample_sizes(domain_configuration)

    run_summary_rows = []
    hessian_summary_rows = []

    total_runs = (
        len(widths)
        * len(depths)
        * len(activation_names)
        * len(sample_sizes)
        * repetitions
    )
    completed_runs = 0

    for activation_name in activation_names:
        activation = get_activation(activation_name)

        if (
            activation_name.lower() == "relu"
            and loss_configuration["function"] == "PINN_Loss"
        ):
            print(
                "Warning: ReLU has zero second derivative almost everywhere "
                "and is unsuitable for this strong-form PINN loss."
            )

        for depth in depths:
            for width in widths:
                for sample_size, repetition in product(
                    sample_sizes,
                    range(repetitions),
                ):
                    n_interior, n_boundary = sample_size
                    sampling_seed = base_seed + repetition
                    initialization_seed = base_seed + 100000 + repetition
                    run_id = (
                        f"{activation_name.lower()}_w{width}_d{depth}"
                        f"_ni{n_interior}_nb{n_boundary}"
                        f"_repeat{repetition}_seed{sampling_seed}"
                    )
                    run_directory = runs_directory / run_id
                    run_directory.mkdir()

                    set_seed(sampling_seed)
                    x = sample_interior(
                        n_interior,
                        device=device,
                    ).to(dtype=dtype)
                    set_seed(sampling_seed + 10000)
                    x_b = sample_boundary(
                        n_boundary,
                        device=device,
                    ).to(dtype=dtype)
                    set_seed(sampling_seed + 20000)
                    x_analysis = sample_interior(
                        analysis_configuration["n_interior"],
                        device=device,
                    ).to(dtype=dtype)
                    set_seed(sampling_seed + 30000)
                    x_b_analysis = sample_boundary(
                        analysis_configuration["n_boundary"],
                        device=device,
                    ).to(dtype=dtype)

                    set_seed(initialization_seed)
                    model = model_class(
                        width=width,
                        depth=depth,
                        activation=activation,
                    ).to(device=device, dtype=dtype)
                    optimizer = make_optimizer(model, training_configuration)
                    number_parameters = sum(
                        parameter.numel()
                        for parameter in model.parameters()
                    )

                    history = []

                    for step in range(epochs + 1):
                        if step in hessian_steps:
                            was_training = model.training
                            model.eval()
                            statistics = analysis_module.compute_hessian_statistics(
                                model=model,
                                loss_function=scalar_loss_function,
                                loss_args=(x_analysis, x_b_analysis),
                                loss_kwargs=loss_kwargs,
                                relative_eig_tol=analysis_configuration.get(
                                    "relative_eig_tol", 1e-8
                                ),
                                absolute_eig_tol=analysis_configuration.get(
                                    "absolute_eig_tol", 0.0
                                ),
                                max_hessian_params=analysis_configuration.get(
                                    "max_hessian_params", 5000
                                ),
                            )
                            model.train(was_training)

                            statistics, eigenvalue_file, statistics_file = (
                                save_hessian_statistics(
                                    statistics,
                                    run_directory,
                                    step,
                                )
                            )
                            hessian_summary_rows.append({
                                "run_id": run_id,
                                "activation": activation_name,
                                "width": width,
                                "depth": depth,
                                "n_interior": n_interior,
                                "n_boundary": n_boundary,
                                "analysis_n_interior": (
                                    analysis_configuration["n_interior"]
                                ),
                                "analysis_n_boundary": (
                                    analysis_configuration["n_boundary"]
                                ),
                                "repetition": repetition,
                                "sampling_seed": sampling_seed,
                                "initialization_seed": initialization_seed,
                                "step": step,
                                **statistics,
                                "eigenvalue_file": str(
                                    Path("runs") / run_id / eigenvalue_file
                                ),
                                "statistics_file": str(
                                    Path("runs") / run_id / statistics_file
                                ),
                            })

                        optimizer.zero_grad(set_to_none=True)
                        total_loss, interior_loss, boundary_loss = unpack_loss(
                            loss_function(
                                model,
                                x,
                                x_b,
                                **loss_kwargs,
                            )
                        )

                        history.append({
                            "step": step,
                            "total_loss": total_loss.detach().cpu().item(),
                            "interior_loss": interior_loss.detach().cpu().item(),
                            "boundary_loss": boundary_loss.detach().cpu().item(),
                        })

                        if step == epochs:
                            break

                        total_loss.backward()
                        optimizer.step()

                    write_csv(
                        run_directory / "loss_history.csv",
                        history,
                        (
                            "step",
                            "total_loss",
                            "interior_loss",
                            "boundary_loss",
                        ),
                    )

                    if output_configuration.get("save_final_models", False):
                        torch.save(
                            model.state_dict(),
                            run_directory / "final_model.pt",
                        )

                    final_history = history[-1]
                    run_summary_rows.append({
                        "run_id": run_id,
                        "activation": activation_name,
                        "width": width,
                        "depth": depth,
                        "n_interior": n_interior,
                        "n_boundary": n_boundary,
                        "number_parameters": number_parameters,
                        "repetition": repetition,
                        "sampling_seed": sampling_seed,
                        "initialization_seed": initialization_seed,
                        "final_total_loss": final_history["total_loss"],
                        "final_interior_loss": final_history["interior_loss"],
                        "final_boundary_loss": final_history["boundary_loss"],
                    })

                    completed_runs += 1
                    print(
                        f"[{completed_runs}/{total_runs}] {run_id}: "
                        f"loss={final_history['total_loss']:.6e}"
                    )

    write_csv(
        experiment_directory / "run_summary.csv",
        run_summary_rows,
        collect_fieldnames(run_summary_rows),
    )

    if hessian_summary_rows:
        write_csv(
            experiment_directory / "hessian_summary.csv",
            hessian_summary_rows,
            collect_fieldnames(hessian_summary_rows),
        )

    print(f"Results saved to {experiment_directory}")
    return experiment_directory


def main():
    parser = argparse.ArgumentParser(
        description="Run the architecture preconditioning experiment."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=PROJECT_DIRECTORY / "architecture_experiment.toml",
        help="Path to the TOML experiment configuration.",
    )
    arguments = parser.parse_args()
    configuration_path = arguments.config.resolve()

    with configuration_path.open("rb") as configuration_file:
        configuration = tomllib.load(configuration_file)

    run_experiment(configuration, configuration_path)


if __name__ == "__main__":
    main()

meow