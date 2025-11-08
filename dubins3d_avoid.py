import os, sys, math, pickle, inspect
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# User-provided loader
# =========================
def load_model_and_dynamics(exp_dir, checkpoint, device):
    from utils import modules
    from dynamics import dynamics as dyn_mod
    import os

    with open(os.path.join(exp_dir, "orig_opt.pickle"), "rb") as f:
        orig_opt = pickle.load(f)

    dclass = getattr(dyn_mod, orig_opt.dynamics_class)
    param_names = [name for name in inspect.signature(dclass).parameters.keys() if name != 'self']
    dargs = {name: getattr(orig_opt, name) for name in param_names}
    dynamics = dclass(**dargs)
    dynamics.deepreach_model = orig_opt.deepreach_model

    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=orig_opt.model,
        mode=orig_opt.model_mode,
        final_layer_factor=1.,
        hidden_features=orig_opt.num_nl,
        num_hidden_layers=orig_opt.num_hl
    ).to(device)

    ckpt_dir = os.path.join(exp_dir, "training", "checkpoints")
    if checkpoint == -1:
        p_final = os.path.join(ckpt_dir, "model_final.pth")
        p_curr  = os.path.join(ckpt_dir, "model_current.pth")
        sd = torch.load(p_final if os.path.exists(p_final) else p_curr, map_location=device)
        model.load_state_dict(sd if isinstance(sd, dict) and "model" not in sd else sd["model"])
    else:
        sd = torch.load(os.path.join(ckpt_dir, f"model_epoch_{checkpoint:04d}.pth"), map_location=device)
        model.load_state_dict(sd["model"])

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dynamics, orig_opt


def make_coords(state_b, in_features, t_scalar: float = 0.0):
    """
    state_b: (B,3) tensor
    returns coords: (B, 1, in_features)
    If in_features == 4, coords = [t, x, y, theta], else [x, y, theta].
    """
    B = state_b.shape[0]
    if in_features == 4:
        t = torch.full((B, 1), float(t_scalar), dtype=state_b.dtype, device=state_b.device)
        coords = torch.cat([t, state_b], dim=1)
    elif in_features == 3:
        coords = state_b
    else:
        raise ValueError(f"Unexpected in_features={in_features}; expected 3 or 4.")
    return coords.unsqueeze(1)


# =========================
# Config — fill in from your message
# =========================
EXPERIMENT_DIR = "deep_reach/runs/dubins_avoid_v3"
CHECKPOINT     = 500000   # -1 to auto-pick final/current
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"

# Match traj_gen_car (scaled env)
DT            = 0.1
T_HOR         = 10.0
N_STEPS       = int(T_HOR / DT)
GOAL_POINT    = (0.8, 0.0)
OBS_CENTER    = (0.0, 0.0)
OBS_RADIUS    = 0.3
BUFFER        = 0.0       # sampling buffer like traj_gen_car uses
SEED          = 0
N_ROLLOUTS    = 20

# Nominal heading PID gains (θ_d - θ)
Kp, Ki, Kd    = 2.5, 0.5, 0.5

# Initial state sampling box (center ± half-extent)
INIT_CENTER       = (-0.75, -0.5)
INIT_HALF_EXTENTS = (0.03, 0.03)

# Shield params
ALPHA         = 0.1       # dV/dt >= -alpha * V (alpha=0 -> nonincreasing V)
GRAD_EPS      = 1e-6
SHIELD_MODE   = "gradient"  # "gradient" uses dotV constraint; "value" switches to avoid-optimal when V <= VALUE_EPS
VALUE_EPS     = 0.1

# Goal tolerance (just for optional success reporting; we always run full horizon)
GOAL_TOL      = 0.05

torch.set_default_dtype(torch.float32)

DEBUG_LOGS = False


def debug_print(*args, **kwargs):
    if DEBUG_LOGS:
        print(*args, **kwargs)


def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

@torch.no_grad()
def model_value(model, dynamics, s, in_features):
    """
    s: (...,3) -> V in real units (...,)
    Uses dynamics.coord_to_input + dynamics.io_to_value
    """
    s_b = s.view(-1, 3)
    coords = make_coords(s_b, in_features)                                 # (B,1,D)
    inp = dynamics.coord_to_input(coords)                                  # normalized
    out = model({'coords': inp})                                           # dict with 'model_in', 'model_out'
    model_in  = out['model_in']                                            # (B,1,D) requires_grad=True (inside model)
    model_out = out['model_out'].squeeze(-1)                               # (B,1)
    V = dynamics.io_to_value(model_in.detach(), model_out.detach())        # (B,1)
    return V.view(-1)                                                      # (B,)


def value_and_grad(model, dynamics, s, in_features, t_scalar: float):
    """
    s: (B,3) WITHOUT requires_grad
    returns: V (B,), dVds (B,3), LfV (B,), LgV (B,)
    """
    B = s.shape[0]
    coords = make_coords(s, in_features, t_scalar=t_scalar)                # (B,1,D)
    inp = dynamics.coord_to_input(coords)
    out = model({'coords': inp})

    model_in  = out['model_in']                                            # (B,1,D)
    model_out = out['model_out'].squeeze(-1)                               # (B,1)

    # Real V
    V = dynamics.io_to_value(model_in.detach(), model_out.detach()).view(B)

    # Real dv = [dV/dt, dV/dx, dV/dy, dV/dθ]
    dv = dynamics.io_to_dv(model_in, model_out).squeeze(1)                 # (B,4)
    dVds = dv[:, 1:]

    th  = s[:, 2]
    LfV = dVds[:, 0]*(VEL*torch.cos(th)) + dVds[:, 1]*(VEL*torch.sin(th))  # f = [v cosθ, v sinθ, 0]
    LgV = dVds[:, 2]                                                       # g = [0,0,1] → dV/dθ
    return V, dVds, LfV, LgV


def shielded_control(model, dynamics, state, u_nom, v, omega_max, t_scalar, alpha=ALPHA, eps=GRAD_EPS,
                     mode=None, value_eps=None):
    """
    Shield controller for single-input Dubins vehicle with |u| <= omega_max.

    mode == "gradient": enforce dotV = LfV + LgV * u >= -alpha * V via projection.
    mode == "value":    if V <= value_eps, apply avoid-optimal control; else use nominal.
    """
    s = state.detach().clone()

    V_t, dVds_t, LfV_t, LgV_t = value_and_grad(model, dynamics, s.unsqueeze(0), IN_FEATURES, t_scalar)
    V = V_t[0]
    dVds = dVds_t[0]
    LfV = LfV_t[0]
    LgV = LgV_t[0]

    mode = (mode or SHIELD_MODE).lower()
    if mode not in {"gradient", "value"}:
        raise ValueError(f"Unsupported shield mode '{mode}'. Expected 'gradient' or 'value'.")
    value_eps = VALUE_EPS if value_eps is None else value_eps

    rhs = -alpha * V - LfV                         # want LgV * u >= rhs

    # Always check feasibility with the bounded nominal
    u_nom_sat = torch.clamp(u_nom[0], -omega_max, omega_max)

    lhs_nom   = LgV * u_nom_sat
    dotV_nom  = LfV + lhs_nom
    if mode == "value":
        violates_flag = bool((V <= value_eps).item())
        if violates_flag:
            u_fallback = dynamics.optimal_control(s.unsqueeze(0), dVds.unsqueeze(0)).squeeze(0)[0]
            u_safe     = torch.clamp(u_fallback, -omega_max, omega_max).unsqueeze(0)
        else:
            u_safe = u_nom_sat.unsqueeze(0)
    else:
        violates_flag = bool((lhs_nom < rhs).item())
        if (abs(LgV) < eps) and violates_flag:
            # Gradient gives no steering authority: go to avoid-optimal
            u_fallback = dynamics.optimal_control(s.unsqueeze(0), dVds.unsqueeze(0)).squeeze(0)[0]
            u_safe     = torch.clamp(u_fallback, -omega_max, omega_max).unsqueeze(0)
        elif violates_flag:
            # Project onto the half-space {u | LgV u >= rhs} under box constraints
            u_bar = rhs / (LgV + (abs(LgV) < eps) * eps)   # safe-side threshold
            if LgV > 0:
                u_proj = torch.maximum(u_nom_sat, u_bar)
            else:
                u_proj = torch.minimum(u_nom_sat, u_bar)
            u_safe = torch.clamp(u_proj, -omega_max, omega_max).unsqueeze(0)
        else:
            u_safe = u_nom_sat.unsqueeze(0)

    # Final safety check after clamping/projection; if still bad, take avoid-optimal
    dotV_safe = (LfV + LgV * u_safe[0])
    if dotV_safe < -alpha * V:
        u_fallback = dynamics.optimal_control(s.unsqueeze(0), dVds.unsqueeze(0)).squeeze(0)[0]
        u_safe     = torch.clamp(u_fallback, -omega_max, omega_max).unsqueeze(0)
        dotV_safe  = (LfV + LgV * u_safe[0])

    # Diagnostics (HJ avoid-optimal for reference; sign matches dynamics)
    u_hj    = float(torch.clamp(dynamics.optimal_control(s.unsqueeze(0), dVds.unsqueeze(0)).squeeze(0)[0],
                                -omega_max, omega_max))
    dotV_hj = float(LfV + LgV * u_hj)

    return u_safe, {
        "V": float(V),
        "LfV": float(LfV),
        "LgV": float(LgV),
        "rhs": float(rhs),
        "dotV_nom": float(dotV_nom),
        "dotV_safe": float(dotV_safe),
        "u_hj": u_hj,
        "dotV_hj": dotV_hj,
        "violates": violates_flag,
        "mode": mode,
        "value_eps": float(value_eps),
    }


def dubins_step(state, u, v, dt=DT):
    """ Euler step; state (3,), u (1,) """
    x, y, th = state.tolist()
    w = float(u[0])
    x  = x  + dt * v * math.cos(th)
    y  = y  + dt * v * math.sin(th)
    th = wrap_to_pi(th + dt * w)
    return torch.tensor([x, y, th], dtype=torch.float32, device=state.device)


def nominal_pid_heading(state, goal, last_e=0.0, e_int=0.0, dt=DT):
    """ PD/PID on heading to drive straight to goal; returns u_nom (1,), new (e, e_int) """
    x, y, th = state.tolist()
    dx, dy = goal[0] - x, goal[1] - y
    th_d   = math.atan2(dy, dx)
    e      = wrap_to_pi(th_d - th)
    de     = (e - last_e) / dt
    e_int  = e_int + e * dt
    u      = Kp * e + Ki * e_int + Kd * de
    return torch.tensor([u], dtype=torch.float32, device=state.device), e, e_int


def sample_initial_state(rng, obs_center=OBS_CENTER, obs_radius=OBS_RADIUS+BUFFER):
    """ Sample within configured box around INIT_CENTER, excluding buffered obstacle; θ ~ U[-π/2, π/2] """
    cx, cy = INIT_CENTER
    hx, hy = INIT_HALF_EXTENTS
    ocx, ocy = obs_center
    while True:
        x0 = rng.uniform(cx - hx, cx + hx)
        y0 = rng.uniform(cy - hy, cy + hy)
        if (x0-ocx)**2 + (y0-ocy)**2 >= (obs_radius**2):
            th0 = rng.uniform(-math.pi/2, math.pi/2)
            return torch.tensor([x0, y0, th0], dtype=torch.float32, device=DEVICE)


def make_plot(ax, paths, obs_center, obs_radius, goal_point):
    """ paths: list of (N,3) arrays """
    # obstacle
    circle = plt.Circle(obs_center, obs_radius, color='k', alpha=0.12, ec='k')
    ax.add_patch(circle)
    cx, cy = INIT_CENTER
    hx, hy = INIT_HALF_EXTENTS
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
    goal_circle = plt.Circle(goal_point, GOAL_TOL, color='gold', alpha=0.25, ec='goldenrod', lw=1.5)
    ax.add_patch(goal_circle)
    ax.text(goal_point[0] + GOAL_TOL + 0.02, goal_point[1] + 0.02, "Goal tol", fontsize=9)

    for path in paths:
        x, y = path[:,0], path[:,1]
        ax.plot(x, y, color='steelblue', linewidth=1.5, alpha=0.9)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_title('Dubins avoid: blue=PID, red=shield active')


def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # Load model/dynamics
    model, dynamics, orig_opt = load_model_and_dynamics(EXPERIMENT_DIR, CHECKPOINT, DEVICE)
    model.to(DEVICE)
    global IN_FEATURES
    IN_FEATURES = int(getattr(dynamics, "input_dim", 4))
    global VEL
    VEL = float(getattr(dynamics, "velocity", 0.2))
    omega_max = float(getattr(dynamics, "omega_max", 3.6))
    TMAX = float(getattr(orig_opt, "tMax", 1.0))

    # Obstacle from dynamics if present; else use traj_gen_car values
    if hasattr(dynamics, "obs_center"):
        oc = getattr(dynamics, "obs_center")
        oc = oc.detach().cpu().numpy().tolist() if torch.is_tensor(oc) else list(oc)
        obs_center = (float(oc[0]), float(oc[1]))
    elif hasattr(dynamics, "obs_center_x") and hasattr(dynamics, "obs_center_y"):
        obs_center = (float(dynamics.obs_center_x), float(dynamics.obs_center_y))
    else:
        obs_center = OBS_CENTER

    collisionR = float(getattr(dynamics, "collisionR", OBS_RADIUS))
    obs_radius = collisionR
    obs_radius = 0.3

    # Rollouts
    paths = []
    successes = []
    statuses = []

    out_dir = os.path.join(EXPERIMENT_DIR, "testing_rollouts")
    os.makedirs(out_dir, exist_ok=True)

    for ridx in range(N_ROLLOUTS):
        s = sample_initial_state(rng, obs_center=obs_center, obs_radius=obs_radius+BUFFER)
        last_e, e_int = 0.0, 0.0

        traj = [s.detach().cpu().numpy()]
        success = False
        collided = False

        for tstep in range(N_STEPS):
            # nominal PID on heading to goal (drives through obstacle)
            u_nom, e, e_int = nominal_pid_heading(s, GOAL_POINT, last_e=last_e, e_int=e_int, dt=DT)
            last_e = e
            t_model = min(tstep * DT, TMAX)
            # apply shield
            u_safe, info = shielded_control(
                model,
                dynamics,
                s,
                u_nom,
                VEL,
                omega_max,
                t_scalar=t_model,
                alpha=ALPHA,
                eps=GRAD_EPS,
                mode=SHIELD_MODE,
                value_eps=VALUE_EPS,
            )

            # Print only when shield changes u
            if abs(float(u_safe[0]) - float(u_nom[0])) > 1e-6 and not np.isnan(info["LgV"]):
                debug_print(f"[Shield] t={tstep*DT:5.2f}  s={s.detach().cpu().numpy()}  "
                            f"V={info['V']:.4f}  LfV={info['LfV']:.4f}  LgV={info['LgV']:.4f}  rhs={info['rhs']:.4f}  "
                            f"u_nom={float(u_nom[0]): .3f}  u_hj={info['u_hj']: .3f}  u_safe={float(u_safe[0]): .3f}  "
                            f"dotV_nom={info['dotV_nom']:.4f}  dotV_safe={info['dotV_safe']:.4f}")

            slack = info["dotV_safe"] + ALPHA*info["V"]     # should be >= 0
            debug_print(f"t={tstep*DT:5.2f}  V={info['V']:+.3f}  dotV_nom={info['dotV_nom']:+.3f}  "
                        f"dotV_safe={info['dotV_safe']:+.3f}  slack={slack:+.3e}")

            # integrate one step
            s = dubins_step(s, u_safe, VEL, dt=DT)

            traj.append(s.detach().cpu().numpy())
            dist_sq = (s[0].item() - obs_center[0])**2 + (s[1].item() - obs_center[1])**2
            if dist_sq <= obs_radius**2:
                collided = True
                debug_print(f"[Collision] rollout={ridx:02d} t={tstep*DT:5.2f} state={s.detach().cpu().numpy()}")
                break
            goal_dist = math.hypot(s[0].item() - GOAL_POINT[0], s[1].item() - GOAL_POINT[1])
            if goal_dist <= GOAL_TOL:
                success = True
                break

        paths.append(np.stack(traj, axis=0))
        successes.append(success if not collided else False)

        final_xy = traj[-1][:2]
        if collided:
            status = "COLLIDE"
        else:
            status = "SUCCESS" if success else "MISS"
        debug_print(f"[{status}] rollout={ridx:02d}  final_pos=({final_xy[0]: .3f}, {final_xy[1]: .3f})")
        statuses.append(status)

    hits = sum(successes)
    print(f"[Stats] {hits}/{N_ROLLOUTS} within {GOAL_TOL} m of goal")
    safe_idxs = [idx for idx, status in enumerate(statuses) if status != "COLLIDE"]
    print(f"[Stats] {len(safe_idxs)}/{N_ROLLOUTS} rollouts avoided collision")
    if safe_idxs:
        debug_print("[Safe Rollouts]")
        for idx in safe_idxs:
            final_xy = paths[idx][-1][:2]
            debug_print(f"  rollout={idx:02d}  status={statuses[idx]}  final_pos=({final_xy[0]: .3f}, {final_xy[1]: .3f})")

    # Plot all rollouts
    fig, ax = plt.subplots(figsize=(6, 6))
    make_plot(ax, paths, obs_center, obs_radius, GOAL_POINT)
    ax.set_xlim(-1.2, 1.0)
    ax.set_ylim(-1.2, 1.0)
    pdf_path = os.path.join(out_dir, "rollouts_xy_test.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')
    debug_print(f"[Saved] {pdf_path}")
    data_path = os.path.join(out_dir, "rollouts_xy_data_test.pt")
    torch.save(
        {
            "paths": paths,
            "statuses": statuses,
            "goal_point": GOAL_POINT,
            "goal_tol": GOAL_TOL,
            "obs_center": obs_center,
            "obs_radius": obs_radius,
            "init_center": INIT_CENTER,
            "init_half_extents": INIT_HALF_EXTENTS,
            "alpha": ALPHA,
            "dt": DT,
            "n_steps": N_STEPS,
        },
        data_path,
    )
    debug_print(f"[Saved] {data_path}")

if __name__ == "__main__":
    main()
