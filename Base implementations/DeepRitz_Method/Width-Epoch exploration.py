import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODULE_PATH = Path("2D-CornerDomain-Poisson.py")


def load_drm_module(path):
    spec = importlib.util.spec_from_file_location("corner_poisson", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_spikes(
    losses,
    window=75,
    min_epoch=150,
    min_abs_jump=0.10,
    min_baseline_rise=0.10,
    mad_threshold=12.0,
    cooldown=100,
):
    """
    Detect large upward instability events in the loss curve.

    This deliberately ignores small oscillations. It also treats a cluster of
    nearby high-loss points as one spike event.
    """
    losses = np.asarray(losses, dtype=float)
    spikes = []

    last_spike_epoch = -cooldown - 1

    start_epoch = max(window, min_epoch)
    for epoch in range(start_epoch, len(losses)):
        local = losses[epoch - window:epoch]
        local_median = np.median(local)
        local_mad = np.median(np.abs(local - local_median))
        robust_scale = max(1.4826 * local_mad, 1e-12)

        current_loss = losses[epoch]
        previous_loss = losses[epoch - 1]

        jump = current_loss - previous_loss
        rise_above_baseline = current_loss - local_median
        robust_z_score = rise_above_baseline / robust_scale

        # Only detect upward loss explosions.
        if jump < min_abs_jump:
            continue

        # The point must also be meaningfully above its recent baseline.
        if rise_above_baseline < min_baseline_rise:
            continue

        # Reject ordinary noisy fluctuations around the median.
        if robust_z_score < mad_threshold:
            continue

        # Do not count the same explosion repeatedly.
        if epoch - last_spike_epoch < cooldown:
            continue

        spikes.append({
            "epoch": epoch,
            "loss": float(current_loss),
            "baseline": float(local_median),
            "rise_above_baseline": float(rise_above_baseline),
            "robust_z_score": float(robust_z_score),
            "previous_loss": float(previous_loss),
            "jump": float(jump),
        })

        last_spike_epoch = epoch

    return spikes


# --- Plotting function for width vs. spike epoch ---
def plot_width_vs_spike_epoch(spikes_df, summary_df, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(8, 5))

    if not spikes_df.empty:
        plt.scatter(
            spikes_df["width"],
            spikes_df["epoch"],
            marker="x",
            label="Detected spike",
        )

    no_spike_df = summary_df[summary_df["num_spikes"] == 0]
    if not no_spike_df.empty:
        plt.scatter(
            no_spike_df["width"],
            [summary_df["max_loss_epoch"].max()] * len(no_spike_df),
            marker="o",
            label="No spike detected",
        )

    plt.xscale("log", base=2)
    plt.xlabel("Network width N")
    plt.ylabel("Epoch")
    plt.title("Detected loss spikes by network width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "width_vs_spike_epoch.png", dpi=200)
    plt.show()


def run_experiment(
    widths,
    epochs,
    seeds=(0,),
    output_dir="width_epoch_spike_results",
    window=75,
    min_abs_jump=0.10,
    min_baseline_rise=0.10,
    mad_threshold=12.0,
    cooldown=100,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    drm = load_drm_module(MODULE_PATH)

    summary_rows = []
    spike_rows = []

    for width in widths:
        for seed in seeds:
            print(f"\nTraining width={width}, seed={seed}")

            _, loss_list = drm.main(
                N=width,
                EPOCHS=epochs,
                plot=False,
                seed=seed,
            )

            losses = np.asarray(loss_list, dtype=float)

            spikes = detect_spikes(
                losses,
                window=window,
                min_abs_jump=min_abs_jump,
                min_baseline_rise=min_baseline_rise,
                mad_threshold=mad_threshold,
                cooldown=cooldown,
            )

            for spike in spikes:
                spike_rows.append({
                    "width": width,
                    "seed": seed,
                    **spike,
                })

            summary_rows.append({
                "width": width,
                "seed": seed,
                "num_spikes": len(spikes),
                "spike_epochs": [s["epoch"] for s in spikes],
                "first_spike_epoch": spikes[0]["epoch"] if spikes else None,
                "max_loss_epoch": int(np.argmax(losses)),
                "max_loss": float(np.max(losses)),
                "final_loss": float(losses[-1]),
                "min_loss": float(np.min(losses)),
            })

            loss_df = pd.DataFrame({
                "epoch": np.arange(len(losses)),
                "loss": losses,
                "width": width,
                "seed": seed,
            })

            loss_df.to_csv(
                output_dir / f"loss_width_{width}_seed_{seed}.csv",
                index=False,
            )

            plt.figure(figsize=(8, 4))
            plt.plot(losses)

            if spikes:
                spike_epochs = [s["epoch"] for s in spikes]
                spike_losses = [s["loss"] for s in spikes]
                plt.scatter(spike_epochs, spike_losses, marker="x")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Width={width}, seed={seed}")
            plt.tight_layout()
            plt.savefig(output_dir / f"loss_width_{width}_seed_{seed}.png", dpi=200)
            plt.close()

    summary_df = pd.DataFrame(summary_rows)
    spikes_df = pd.DataFrame(spike_rows)

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    spikes_df.to_csv(output_dir / "detected_spikes.csv", index=False)

    plot_width_vs_spike_epoch(
        spikes_df=spikes_df,
        summary_df=summary_df,
        output_dir=output_dir,
    )

    return summary_df, spikes_df


if __name__ == "__main__":
    widths = [2,3,4,6,8,12,16,24,32,48,64, 96, 128, 192, 256]

    summary, spikes = run_experiment(
        widths=widths,
        epochs=4000,
        seeds=(0, 1, 2, 3),
        window=75,
        min_abs_jump=0.10,
        min_baseline_rise=0.10,
        mad_threshold=12.0,
        cooldown=100,
    )

    print("\nSummary")
    print(summary)

    print("\nSpikes")
    print(spikes)