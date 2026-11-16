"""Plot saved knot tracks and peak amplitude; run without rerunning the solver."""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

DATA_FILE = Path(__file__).with_name("solution_history.pt")
BLOWUP_TIME = None  # None uses the final recorded time, only as an approximation.
TRACK_IDS = None    # For example [3, 7, 12]; None plots every track.
SOLUTION_SNAPSHOTS = 8  # Equally spaced saved steps, including first and last.

history = torch.load(DATA_FILE, map_location="cpu", weights_only=True)
times = np.array([s["t"] for s in history])
peaks = np.array([s["peak"] for s in history])  # Saved estimate of max |u|.
T = times[-1] if BLOWUP_TIME is None else BLOWUP_TIME
if not np.isfinite(T) or T <= 0:
    raise ValueError("Set BLOWUP_TIME to a finite positive time.")

plt.scatter(times, [1/x**2 for x in peaks])
plt.show()




log_time = np.full(len(times), np.nan)
valid = np.isfinite(times) & (times < T)
log_time[valid] = np.log(T - times[valid])
log_peak = np.full(len(times), np.nan)
valid = np.isfinite(peaks) & (peaks > 0)
log_peak[valid] = np.log(peaks[valid])

# Missing snapshots remain NaN so separate pieces are not joined across gaps.
tracks = {}
for step, snapshot in enumerate(history):
    for layer, neuron, track_id, x in snapshot["knots"].tolist():
        key = (int(layer), int(neuron), int(track_id))
        if TRACK_IDS is not None and key[2] not in TRACK_IDS:
            continue
        if key not in tracks:
            tracks[key] = np.full(len(times), np.nan)
        tracks[key][step] = x

fig, axes = plt.subplots(1, 3, figsize=(17, 9))
for number, ((layer, neuron, track_id), positions) in enumerate(sorted(tracks.items())):
    log_x = np.full(len(times), np.nan)
    valid = np.isfinite(positions) & (positions != 0)
    log_x[valid] = np.log(np.abs(positions[valid]))
    label = f"ID {track_id}: L{layer}, N{neuron}"
    style = dict(color=plt.get_cmap("turbo")(number / max(1, len(tracks) - 1)),
                 linewidth=1, marker=".", markersize=2, label=label)
    axes[0].plot(log_x, log_time, **style)
    axes[1].plot(log_x, log_peak, **style)


axes[0].set(xlabel=r"$\log|x|$", ylabel=r"$\log(T-t)$", title="Knot tracks")
axes[1].set(xlabel=r"$\log(\max|u|)$", ylabel=r"$\log|x|$",
            title="Knot position versus peak amplitude")
axes[2].plot(times, log_peak)
axes[2].set(xlabel="t", ylabel=r"$\log(\max|u|)$", title="Peak amplitude")
for ax in axes:
    ax.grid(alpha=.25)

note = "final recorded time; approximation" if BLOWUP_TIME is None else "supplied"
fig.suptitle(f"T = {T:.12g} ({note})\nNatural logs; saved peak estimates")
if tracks:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7)
fig.tight_layout(rect=(0, min(.65, .035 * np.ceil(len(tracks) / 5) + .02), 1, .93))
output = DATA_FILE.with_name("knot_tracking_plots.png")
fig.savefig(output, dpi=180)
print(f"Saved: {output}")
print("First panel excludes t >= T; knot panels exclude x = 0. Missing tracks break lines.")
print("Tracking IDs use proximity matching; they do not guarantee branch identity.")
# Each saved timestep has its own adaptive spatial grid.
solution_fig, solution_axes = plt.subplots(1, 2, figsize=(13, 5))
steps = np.unique(np.linspace(0, len(history) - 1,
                              min(SOLUTION_SNAPSHOTS, len(history)), dtype=int))
for number, step in enumerate(steps):
    snapshot = history[step]
    x = np.asarray(snapshot["x"], dtype=float)
    u = np.asarray(snapshot["u"], dtype=float)
    order = np.argsort(x)
    x, u = x[order], u[order]
    u[~np.isfinite(u)] = np.nan
    log_u = np.full_like(u, np.nan)
    positive = np.isfinite(u) & (u > 0)
    log_u[positive] = np.log(u[positive])
    style = dict(color=plt.get_cmap("viridis")(number / max(1, len(steps) - 1)),
                 label=f"Step {step}: t={snapshot['t']:.12g}")
    solution_axes[0].plot(x, u, **style)
    solution_axes[1].plot(x, log_u, **style)
solution_axes[0].set(xlabel="x", ylabel="u(x,t)", title="Solution evolution")
solution_axes[1].set(xlabel="x", ylabel=r"$\log u(x,t)$",
                     title="Log solution evolution (u > 0)")
for ax in solution_axes:
    ax.grid(alpha=.25)
solution_axes[0].legend(fontsize=7)
solution_fig.tight_layout()
solution_output = DATA_FILE.with_name("solution_evolution.png")
solution_fig.savefig(solution_output, dpi=180)
print(f"Saved: {solution_output}")

plt.show()
