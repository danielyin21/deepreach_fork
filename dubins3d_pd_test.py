import math
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


DT = 0.1
T_HOR = 10.0
N_STEPS = int(T_HOR / DT)
GOAL_POINT = (0.8, 0.0)
VEL = 0.2
OMEGA_LIMIT = 3.6

Kp, Ki, Kd = 2.5, 0.5, 0.5

SEED = 0
N_ROLLOUTS = 20
GOAL_TOL = 0.03

OUT_DIR = Path("deep_reach/runs/dubins_avoid_v0/testing_rollouts_pid")

torch.set_default_dtype(torch.float32)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def dubins_step(state: torch.Tensor, u: torch.Tensor, v: float, dt: float = DT) -> torch.Tensor:
    """Integrate Dubins dynamics forward one Euler step."""
    x, y, theta = state.tolist()
    w = float(u[0])

    x = x + dt * v * math.cos(theta)
    y = y + dt * v * math.sin(theta)
    theta = wrap_to_pi(theta + dt * w)
    return torch.tensor([x, y, theta], dtype=torch.float32, device=state.device)


def nominal_pid_heading(state: torch.Tensor,
                        goal_xy: tuple[float, float],
                        last_error: float,
                        integral_error: float,
                        dt: float = DT) -> tuple[torch.Tensor, float, float]:
    """Heading PD (optionally PI) controller toward the goal."""
    x, y, theta = state.tolist()
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    desired_theta = math.atan2(dy, dx)
    error = wrap_to_pi(desired_theta - theta)
    derivative = (error - last_error) / dt
    integral_error = integral_error + error * dt
    omega = Kp * error + Ki * integral_error + Kd * derivative
    u = torch.tensor([float(np.clip(omega, -OMEGA_LIMIT, OMEGA_LIMIT))], dtype=torch.float32, device=state.device)
    return u, error, integral_error


def sample_initial_state(rng: np.random.Generator) -> torch.Tensor:
    """Match the training sampler but without obstacle rejection."""
    x0 = rng.uniform(-1.0, -0.5)
    y0 = rng.uniform(-1.0, 1.0)
    theta0 = rng.uniform(-math.pi / 2, math.pi / 2)
    return torch.tensor([x0, y0, theta0], dtype=torch.float32, device="cpu")


def run_rollout(controller_seed: int) -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng(controller_seed)
    torch.manual_seed(controller_seed)
    np.random.seed(controller_seed)

    state = sample_initial_state(rng)
    last_e, e_int = 0.0, 0.0

    traj = [state.detach().cpu().numpy()]
    success = False
    for _ in range(N_STEPS):
        u, last_e, e_int = nominal_pid_heading(state, GOAL_POINT, last_e, e_int)
        state = dubins_step(state, u, VEL)
        traj.append(state.detach().cpu().numpy())
        goal_dist = math.hypot(state[0].item() - GOAL_POINT[0], state[1].item() - GOAL_POINT[1])
        if goal_dist <= GOAL_TOL:
            success = True
            break

    final_xy = traj[-1][:2]
    return np.stack(traj, axis=0), success


def make_plot(paths: list[np.ndarray], successes: list[bool], goal_point: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    goal_circle = plt.Circle(goal_point, GOAL_TOL, color="gold", alpha=0.25, ec="goldenrod", lw=1.5)
    ax.add_patch(goal_circle)
    ax.text(goal_point[0] + GOAL_TOL + 0.02, goal_point[1] + 0.02, "Goal tol", fontsize=9)

    for path, success in zip(paths, successes):
        color = "steelblue" if success else "indianred"
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5, alpha=0.9)

    ax.set_xlim(-1.2, 1.0)
    ax.set_ylim(-1.2, 1.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "rollouts_xy_pid_only.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[Saved] {pdf_path}")
    plt.close(fig)


def main() -> None:
    paths = []
    successes = []

    for ridx in range(N_ROLLOUTS):
        traj, success = run_rollout(SEED + ridx)
        paths.append(traj)
        successes.append(success)
        status = "SUCCESS" if success else "MISS"
        final_xy = traj[-1][:2]
        print(f"[{status}] rollout={ridx:02d}  final_pos=({final_xy[0]: .3f}, {final_xy[1]: .3f})")

    hits = sum(successes)
    print(f"[Stats] {hits}/{N_ROLLOUTS} within {GOAL_TOL} m of goal")
    make_plot(paths, successes, GOAL_POINT)


if __name__ == "__main__":
    main()
