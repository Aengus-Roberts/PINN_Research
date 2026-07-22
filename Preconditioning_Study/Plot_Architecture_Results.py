import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIRECTORY = (
    PROJECT_DIRECTORY / "Results" / "Architecture_Preconditioning"
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def as_int(row, key):
    return int(row[key])


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()


def find_latest_experiment(results_directory):
    if not results_directory.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_directory}"
        )

    candidates = [
        path
        for path in results_directory.iterdir()
        if path.is_dir()
        and (path / "run_summary.csv").exists()
        and (path / "hessian_summary.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No completed experiments found in {results_directory}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_experiment(experiment_directory):
    run_summary_path = experiment_directory / "run_summary.csv"
    hessian_summary_path = experiment_directory / "hessian_summary.csv"
    metadata_path = experiment_directory / "metadata.json"

    if not run_summary_path.exists():
        raise FileNotFoundError(run_summary_path)

    if not hessian_summary_path.exists():
        raise FileNotFoundError(hessian_summary_path)

    run_rows = read_csv(run_summary_path)
    hessian_rows = read_csv(hessian_summary_path)

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as input_file:
            metadata = json.load(input_file)

    history_rows = []

    for run in run_rows:
        history_path = (
            experiment_directory
            / "runs"
            / run["run_id"]
            / "loss_history.csv"
        )

        if not history_path.exists():
            print(f"Warning: missing loss history: {history_path}")
            continue

        for history in read_csv(history_path):
            history_rows.append({
                "run_id": run["run_id"],
                "activation": run["activation"],
                "width": as_int(run, "width"),
                "depth": as_int(run, "depth"),
                "repetition": as_int(run, "repetition"),
                "step": as_int(history, "step"),
                "total_loss": as_float(history, "total_loss"),
                "interior_loss": as_float(history, "interior_loss"),
                "boundary_loss": as_float(history, "boundary_loss"),
            })

    return run_rows, hessian_rows, history_rows, metadata


def experiment_label(metadata):
    configuration = metadata.get("configuration", {})
    domain = configuration.get("domain", {}).get("module", "unknown domain")
    loss_configuration = configuration.get("loss", {})
    loss_module = loss_configuration.get("module", "unknown problem")
    loss_function = loss_configuration.get("function", "unknown loss")
    return f"{domain} | {loss_module}.{loss_function}"


def save_figure(figure, output_directory, stem, formats, dpi, show):
    for output_format in formats:
        save_kwargs = {"bbox_inches": "tight"}
        if output_format.lower() == "png":
            save_kwargs["dpi"] = dpi

        figure.savefig(
            output_directory / f"{stem}.{output_format}",
            **save_kwargs,
        )

    if not show:
        plt.close(figure)


def aggregate_matrix(
    rows,
    widths,
    depths,
    value_key,
    transform=None,
):
    grouped_values = defaultdict(list)

    for row in rows:
        value = as_float(row, value_key)

        if not math.isfinite(value):
            continue

        if transform is not None:
            value = transform(value)

        if math.isfinite(value):
            grouped_values[(as_int(row, "depth"), as_int(row, "width"))].append(
                value
            )

    matrix = np.full((len(depths), len(widths)), np.nan)

    for depth_index, depth in enumerate(depths):
        for width_index, width in enumerate(widths):
            values = grouped_values[(depth, width)]

            if values:
                matrix[depth_index, width_index] = np.median(values)

    return matrix


def draw_heatmap(
    axis,
    matrix,
    widths,
    depths,
    title,
    cmap,
    vmin,
    vmax,
    value_format,
    annotate,
):
    colour_map = plt.get_cmap(cmap).copy()
    colour_map.set_bad("lightgray")
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        origin="lower",
        aspect="auto",
        cmap=colour_map,
        vmin=vmin,
        vmax=vmax,
    )

    axis.set_xticks(range(len(widths)), labels=widths)
    axis.set_yticks(range(len(depths)), labels=depths)
    axis.set_xlabel("Width")
    axis.set_ylabel("Depth")
    axis.set_title(title)

    if annotate and matrix.size <= 64:
        for depth_index in range(len(depths)):
            for width_index in range(len(widths)):
                value = matrix[depth_index, width_index]

                if math.isfinite(value):
                    axis.text(
                        width_index,
                        depth_index,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    return image


def finite_limits(matrices):
    finite_arrays = [
        matrix[np.isfinite(matrix)]
        for matrix in matrices
        if np.any(np.isfinite(matrix))
    ]

    if not finite_arrays:
        return None

    values = np.concatenate(finite_arrays)

    minimum = float(values.min())
    maximum = float(values.max())

    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
        minimum -= padding
        maximum += padding

    return minimum, maximum


def plot_condition_heatmaps(
    hessian_rows,
    activations,
    widths,
    depths,
    initial_step,
    final_step,
    label,
    output_directory,
    formats,
    dpi,
    annotate,
    show,
):
    for activation in activations:
        initial_rows = [
            row for row in hessian_rows
            if row["activation"] == activation
            and as_int(row, "step") == initial_step
        ]
        final_rows = [
            row for row in hessian_rows
            if row["activation"] == activation
            and as_int(row, "step") == final_step
        ]

        log_condition = lambda value: (
            math.log10(value) if value > 0.0 else float("nan")
        )
        initial_matrix = aggregate_matrix(
            initial_rows,
            widths,
            depths,
            "condition_number_positive",
            transform=log_condition,
        )
        final_matrix = aggregate_matrix(
            final_rows,
            widths,
            depths,
            "condition_number_positive",
            transform=log_condition,
        )
        limits = finite_limits((initial_matrix, final_matrix))

        if limits is None:
            print(f"Warning: no finite condition numbers for {activation}")
            continue

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4.5),
            layout="constrained",
        )
        image = draw_heatmap(
            axes[0],
            initial_matrix,
            widths,
            depths,
            f"Initialisation (step {initial_step})",
            "viridis",
            *limits,
            ".2f",
            annotate,
        )
        draw_heatmap(
            axes[1],
            final_matrix,
            widths,
            depths,
            f"Final checkpoint (step {final_step})",
            "viridis",
            *limits,
            ".2f",
            annotate,
        )
        figure.colorbar(
            image,
            ax=axes,
            label=r"Median $\log_{10}(\kappa_+)$",
        )
        figure.suptitle(f"Positive-subspace conditioning | {activation} | {label}")
        save_figure(
            figure,
            output_directory,
            f"01_condition_heatmaps_{safe_name(activation)}",
            formats,
            dpi,
            show,
        )


def plot_curvature_fraction_heatmaps(
    hessian_rows,
    activations,
    widths,
    depths,
    final_step,
    label,
    output_directory,
    formats,
    dpi,
    annotate,
    show,
):
    for activation in activations:
        final_rows = [
            row for row in hessian_rows
            if row["activation"] == activation
            and as_int(row, "step") == final_step
        ]
        negative_matrix = aggregate_matrix(
            final_rows,
            widths,
            depths,
            "negative_fraction",
        )
        near_zero_matrix = aggregate_matrix(
            final_rows,
            widths,
            depths,
            "near_zero_fraction",
        )

        if not (
            np.any(np.isfinite(negative_matrix))
            or np.any(np.isfinite(near_zero_matrix))
        ):
            print(f"Warning: no curvature fractions for {activation}")
            continue

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4.5),
            layout="constrained",
        )
        image = draw_heatmap(
            axes[0],
            negative_matrix,
            widths,
            depths,
            "Negative-eigenvalue fraction",
            "magma",
            0.0,
            1.0,
            ".2f",
            annotate,
        )
        draw_heatmap(
            axes[1],
            near_zero_matrix,
            widths,
            depths,
            "Near-zero-eigenvalue fraction",
            "magma",
            0.0,
            1.0,
            ".2f",
            annotate,
        )
        figure.colorbar(image, ax=axes, label="Median fraction")
        figure.suptitle(
            f"Final Hessian curvature (step {final_step}) | "
            f"{activation} | {label}"
        )
        save_figure(
            figure,
            output_directory,
            f"02_curvature_fractions_{safe_name(activation)}",
            formats,
            dpi,
            show,
        )


def loss_quantiles(history_rows, activation, depth, width, metric):
    grouped_values = defaultdict(list)

    for row in history_rows:
        if (
            row["activation"] == activation
            and row["depth"] == depth
            and row["width"] == width
        ):
            value = row[metric]
            if math.isfinite(value):
                grouped_values[row["step"]].append(value)

    steps = sorted(grouped_values)

    if not steps:
        return None

    return (
        np.asarray(steps),
        np.asarray([np.median(grouped_values[step]) for step in steps]),
        np.asarray([np.quantile(grouped_values[step], 0.25) for step in steps]),
        np.asarray([np.quantile(grouped_values[step], 0.75) for step in steps]),
    )


def set_loss_scale(axis, plotted_values):
    finite_values = np.asarray(plotted_values)
    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        return

    if finite_values.min() > 0.0:
        axis.set_yscale("log")
    else:
        scale = max(float(np.max(np.abs(finite_values))), 1e-12)
        axis.set_yscale("symlog", linthresh=scale * 1e-6)


def plot_loss_histories(
    history_rows,
    activations,
    widths,
    depths,
    label,
    output_directory,
    formats,
    dpi,
    show,
):
    metrics = (
        ("total_loss", "Total loss"),
        ("interior_loss", "Interior loss"),
        ("boundary_loss", "Boundary loss"),
    )
    colour_map = plt.get_cmap("viridis")
    colours = {
        width: colour_map(index / max(len(widths) - 1, 1))
        for index, width in enumerate(widths)
    }

    for activation in activations:
        figure, axes = plt.subplots(
            len(metrics),
            len(depths),
            figsize=(4.2 * len(depths), 9.5),
            squeeze=False,
            layout="constrained",
        )

        for column, depth in enumerate(depths):
            axes[0, column].set_title(f"Depth {depth}")

            for row_index, (metric, metric_label) in enumerate(metrics):
                axis = axes[row_index, column]
                plotted_values = []

                for width in widths:
                    quantiles = loss_quantiles(
                        history_rows,
                        activation,
                        depth,
                        width,
                        metric,
                    )

                    if quantiles is None:
                        continue

                    steps, median, lower, upper = quantiles
                    plotted_values.extend(lower)
                    plotted_values.extend(upper)
                    axis.plot(
                        steps,
                        median,
                        color=colours[width],
                        label=f"width={width}",
                    )
                    axis.fill_between(
                        steps,
                        lower,
                        upper,
                        color=colours[width],
                        alpha=0.18,
                        linewidth=0.0,
                    )

                set_loss_scale(axis, plotted_values)
                axis.grid(True, which="both", alpha=0.2)

                if column == 0:
                    axis.set_ylabel(metric_label)

                if row_index == len(metrics) - 1:
                    axis.set_xlabel("Training step")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="outside right upper")

        figure.suptitle(
            f"Median loss histories with interquartile bands | "
            f"{activation} | {label}"
        )
        save_figure(
            figure,
            output_directory,
            f"03_loss_histories_{safe_name(activation)}",
            formats,
            dpi,
            show,
        )


def plot_condition_against_parameters(
    hessian_rows,
    activations,
    widths,
    depths,
    final_step,
    label,
    output_directory,
    formats,
    dpi,
    show,
):
    marker_cycle = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    width_markers = {
        width: marker_cycle[index % len(marker_cycle)]
        for index, width in enumerate(widths)
    }
    colour_map = plt.get_cmap("plasma")
    depth_colours = {
        depth: colour_map(index / max(len(depths) - 1, 1))
        for index, depth in enumerate(depths)
    }

    for activation in activations:
        final_rows = [
            row for row in hessian_rows
            if row["activation"] == activation
            and as_int(row, "step") == final_step
        ]
        figure, axis = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
        plotted = False

        for depth in depths:
            for width in widths:
                architecture_rows = [
                    row for row in final_rows
                    if as_int(row, "depth") == depth
                    and as_int(row, "width") == width
                ]
                parameter_counts = np.asarray([
                    as_float(row, "number_parameters")
                    for row in architecture_rows
                ])
                conditions = np.asarray([
                    as_float(row, "condition_number_positive")
                    for row in architecture_rows
                ])
                valid = (
                    np.isfinite(parameter_counts)
                    & np.isfinite(conditions)
                    & (parameter_counts > 0.0)
                    & (conditions > 0.0)
                )

                if not np.any(valid):
                    continue

                parameter_counts = parameter_counts[valid]
                conditions = conditions[valid]
                plotted = True
                axis.scatter(
                    parameter_counts,
                    conditions,
                    color=depth_colours[depth],
                    marker=width_markers[width],
                    alpha=0.35,
                    s=35,
                    linewidths=0.0,
                )
                axis.scatter(
                    np.median(parameter_counts),
                    np.median(conditions),
                    color=depth_colours[depth],
                    marker=width_markers[width],
                    edgecolors="black",
                    linewidths=0.8,
                    s=90,
                )

        if not plotted:
            plt.close(figure)
            print(f"Warning: no final condition data for {activation}")
            continue

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of trainable parameters")
        axis.set_ylabel(r"Positive-subspace condition number $\kappa_+$")
        axis.grid(True, which="both", alpha=0.2)
        axis.set_title(
            f"Final conditioning versus parameter count "
            f"(step {final_step})\n{activation} | {label}"
        )

        depth_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=depth_colours[depth],
                markeredgecolor="none",
                label=str(depth),
            )
            for depth in depths
        ]
        width_handles = [
            Line2D(
                [0],
                [0],
                marker=width_markers[width],
                linestyle="none",
                color="black",
                markerfacecolor="none",
                label=str(width),
            )
            for width in widths
        ]
        depth_legend = axis.legend(
            handles=depth_handles,
            title="Depth",
            loc="upper left",
        )
        axis.add_artist(depth_legend)
        axis.legend(
            handles=width_handles,
            title="Width",
            loc="lower right",
        )

        save_figure(
            figure,
            output_directory,
            f"04_condition_vs_parameters_{safe_name(activation)}",
            formats,
            dpi,
            show,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from an architecture preconditioning experiment."
    )
    parser.add_argument(
        "experiment_directory",
        nargs="?",
        type=Path,
        help=(
            "Experiment directory containing run_summary.csv. "
            "If omitted, the latest experiment is used."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Plot output directory. Defaults to <experiment>/Plots.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-annotations", action="store_true")
    parser.add_argument("--show", action="store_true")
    arguments = parser.parse_args()

    if arguments.experiment_directory is None:
        experiment_directory = find_latest_experiment(
            DEFAULT_RESULTS_DIRECTORY
        )
    else:
        experiment_directory = arguments.experiment_directory.resolve()

    output_directory = (
        arguments.output_directory.resolve()
        if arguments.output_directory is not None
        else experiment_directory / "Plots"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    run_rows, hessian_rows, history_rows, metadata = load_experiment(
        experiment_directory
    )

    activations = sorted({row["activation"] for row in run_rows})
    widths = sorted({as_int(row, "width") for row in run_rows})
    depths = sorted({as_int(row, "depth") for row in run_rows})
    hessian_steps = sorted({as_int(row, "step") for row in hessian_rows})

    if not hessian_steps:
        raise ValueError("No Hessian checkpoints were found.")

    initial_step = hessian_steps[0]
    final_step = hessian_steps[-1]
    label = experiment_label(metadata)
    plotting_arguments = (
        output_directory,
        arguments.formats,
        arguments.dpi,
    )

    plot_condition_heatmaps(
        hessian_rows,
        activations,
        widths,
        depths,
        initial_step,
        final_step,
        label,
        *plotting_arguments,
        not arguments.no_annotations,
        arguments.show,
    )
    plot_curvature_fraction_heatmaps(
        hessian_rows,
        activations,
        widths,
        depths,
        final_step,
        label,
        *plotting_arguments,
        not arguments.no_annotations,
        arguments.show,
    )
    plot_loss_histories(
        history_rows,
        activations,
        widths,
        depths,
        label,
        *plotting_arguments,
        arguments.show,
    )
    plot_condition_against_parameters(
        hessian_rows,
        activations,
        widths,
        depths,
        final_step,
        label,
        *plotting_arguments,
        arguments.show,
    )

    print(f"Plots saved to {output_directory}")

    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()
