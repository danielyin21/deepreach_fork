import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def gather_path_sets(data: dict, requested_keys: Iterable[str]) -> List[Tuple[str, List[np.ndarray]]]:
    """Collect path lists from the loaded data."""
    path_sets = []
    if requested_keys:
        keys = requested_keys
    else:
        keys = [
            key for key, value in data.items()
            if isinstance(value, (list, tuple)) and value
            and isinstance(value[0], np.ndarray) and value[0].ndim == 2 and value[0].shape[1] == 3
        ]
    for key in keys:
        if key not in data:
            raise KeyError(f"Path key '{key}' not found in file.")
        value = data[key]
        if not isinstance(value, (list, tuple)) or not value:
            raise TypeError(f"Expected '{key}' to be a non-empty list/tuple of arrays.")
        sample = value[0]
        if not (isinstance(sample, np.ndarray) and sample.ndim == 2 and sample.shape[1] == 3):
            raise ValueError(f"Entries under '{key}' must be arrays of shape (N, 3).")
        paths = [np.asarray(p) for p in value]
        path_sets.append((key, paths))
    return path_sets


def make_plot(ax, path_sets: List[Tuple[str, List[np.ndarray]]], obs_center, obs_radius, goal_point,
              goal_tol, init_center, init_half_extents):
    """Render all path sets on a single axis."""
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['steelblue'])
    if not colors:
        colors = ['steelblue']

    circle = plt.Circle(obs_center, obs_radius, color='k', alpha=0.12, ec='k')
    ax.add_patch(circle)

    cx, cy = init_center
    hx, hy = init_half_extents
    init_box = plt.Rectangle(
        (cx - hx, cy - hy),
        2 * hx,
        2 * hy,
        facecolor='lightgreen',
        alpha=0.25,
        edgecolor='forestgreen',
        linewidth=1.5,
        zorder=10,
    )
    ax.add_patch(init_box)

    goal_circle = plt.Circle(goal_point, goal_tol, color='gold', alpha=0.25, ec='goldenrod', lw=1.5)
    ax.add_patch(goal_circle)
    ax.text(goal_point[0] + goal_tol + 0.02, goal_point[1] + 0.02, "Goal tol", fontsize=9)

    for idx, (name, paths) in enumerate(path_sets):
        color = colors[idx % len(colors)]
        for path in paths:
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=0.9)
        ax.plot([], [], color=color, label=name)

    if len(path_sets) > 1:
        ax.legend(loc="upper right")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_title('Dubins reach-avoid rollouts')


def main():
    pt_path = os.environ.get("ROLL_PLOT_PT_PATH", "deep_reach/runs/dubins_avoid_v3/testing_rollouts/rollouts_xy_data_test.pt")
    out_path = os.environ.get("ROLL_PLOT_OUT_PATH", None)
    raw_path_keys = os.environ.get("ROLL_PLOT_PATH_KEYS", "")
    path_keys = [key.strip() for key in raw_path_keys.split(",") if key.strip()]

    data = torch.load(pt_path, map_location='cpu')
    if not isinstance(data, dict):
        raise TypeError("Loaded object must be a dictionary.")

    required = ["goal_point", "goal_tol", "obs_center", "obs_radius", "init_center", "init_half_extents"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in file.")

    path_sets = gather_path_sets(data, path_keys)
    if not path_sets:
        raise ValueError("No path sets found to plot. Use --path-key to specify keys explicitly.")

    fig, ax = plt.subplots(figsize=(6, 6))
    make_plot(
        ax,
        path_sets,
        obs_center=tuple(map(float, data["obs_center"])),
        obs_radius=float(data["obs_radius"]),
        goal_point=tuple(map(float, data["goal_point"])),
        goal_tol=float(data["goal_tol"]),
        init_center=tuple(map(float, data["init_center"])),
        init_half_extents=tuple(map(float, data["init_half_extents"])),
    )

    ax.set_xlim(-1.2, 1.0)
    ax.set_ylim(-1.2, 1.0)

    if out_path is None:
        base = os.path.splitext(os.path.basename(pt_path))[0]
        out_path = os.path.join(os.path.dirname(pt_path), f"{base}_plot.pdf")

    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
