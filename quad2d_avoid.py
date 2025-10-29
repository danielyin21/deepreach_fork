# quad2d_avoid.py  (fixed: lateral sign + tilt-compensated hover)
import os, math, pickle, random, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

# ======================== USER SETTINGS ========================
EXPERIMENT_DIR = "deep_reach/runs/quad2d_reachavoid_avoid"  # <- your run dir
CHECKPOINT = 100000                                         # -1 for model_current / model_final
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DT = 0.05
HORIZON = 5.0
NUM_ROLLOUTS = 10
SEED = 100

# Start-state sampling
ONLY_START_IN_SAFE_SET = False
OVERSAMPLE_FACTOR = 20
INIT_RANGES_OVERRIDE = {  # (x, z, th, vx, vz, om)
    "x":  (0.0, 1.5),
    "z":  (0.0, 5.0),
    "th": (0.0, 0.0),
    "vx": (0.0, 0.0),
    "vz": (0.0, 0.0),
    "om": (0.0, 0.0),
}

# PID gains (tune as you like)
K_X   = 0.6      # maps horizontal position error -> desired tilt magnitude
K_VX  = 0.0
K_Z   = 3.0      # vertical P (N per meter via thrust)
K_VZ  = 2.0      # vertical D
K_TH  = 5.0      # angle P (rad)
K_OM  = 1.5      # angle D (rad/s)

# Safety shield parameter: require dotV <= -alpha * max(V,0)
ALPHA = 0.10
FD_EPS = 1e-4

# ==================== LOAD MODEL & DYNAMICS ====================
def load_model_and_dynamics(exp_dir, checkpoint, device):
    from utils import modules
    from dynamics import dynamics as dyn_mod

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
        # try final/current automatically
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

# ==================== VALUE & GRAD HELPERS =====================
@torch.no_grad()
def value_at(model, dynamics, t, states):
    coords = torch.cat([torch.full((states.shape[0], 1), float(t), device=states.device),
                        states], dim=-1)
    res = model({"coords": dynamics.coord_to_input(coords)})
    V = dynamics.io_to_value(res["model_in"], res["model_out"].squeeze(-1))
    return V  # [N]

def dvds_at(model, dynamics, t, states):
    coords = torch.cat([torch.full((states.shape[0], 1), float(t), device=states.device),
                        states], dim=-1)
    res = model({"coords": dynamics.coord_to_input(coords)})
    dv = dynamics.io_to_dv(res["model_in"], res["model_out"].squeeze(-1))
    return dv[..., 1:]  # drop dv/dt

# ================== INITIAL STATE SAMPLING =====================
def sample_uniform(dynamics, n, rngs=None, device="cpu"):
    base = dynamics.state_test_range()
    keys = ["x","z","th","vx","vz","om"]
    base = [[*row] for row in base]
    for i,k in enumerate(keys):
        if rngs and (k in rngs): base[i] = [float(rngs[k][0]), float(rngs[k][1])]
    base = torch.tensor(base, dtype=torch.float32, device=device)
    lo, hi = base[:,0], base[:,1]
    return lo + torch.rand(n, dynamics.state_dim, device=device)*(hi-lo)

def choose_initial_states(model, dynamics, n, device):
    if not ONLY_START_IN_SAFE_SET:
        return sample_uniform(dynamics, n, INIT_RANGES_OVERRIDE, device)
    m = n * OVERSAMPLE_FACTOR
    pool = sample_uniform(dynamics, m, INIT_RANGES_OVERRIDE, device)
    V0 = value_at(model, dynamics, 0.0, pool)
    keep = (V0 <= 0.0)
    kept = pool[keep]
    if kept.shape[0] >= n:
        return kept[:n]
    extra = sample_uniform(dynamics, n - kept.shape[0], INIT_RANGES_OVERRIDE, device)
    return torch.cat([kept, extra], dim=0)

# ================== CONTROLLERS (PID & SHIELD) =================
def clamp_u(dynamics, u):
    u1_min, u1_max = dynamics.u1_min, dynamics.u1_max
    u2_min, u2_max = dynamics.u2_min, dynamics.u2_max
    u[..., 0] = u[..., 0].clamp(u1_min, u1_max)
    u[..., 1] = u[..., 1].clamp(u2_min, u2_max)
    return u

def wrap_angle(a):
    return ((a + math.pi) % (2*math.pi)) - math.pi

def pid_nominal(dynamics, s):
    """
    Goal-seeking PID with correct lateral sign and tilt-compensated hover.
    Using quad dynamics:
      dvx = -(sin th)/m * u1
      dvz =  g         + (cos th)/m * u1
    """
    m, Iyy = dynamics.m, dynamics.Iyy
    g = dynamics.g
    gx, gz = dynamics.goal_x, dynamics.goal_z

    th = s[..., 2:3]; vx = s[..., 3:4]; vz = s[..., 4:5]; om = s[..., 5:6]
    x  = s[..., 0:1]; z  = s[..., 1:2]

    # --- Vertical: hover feedforward / cos(th) + PD about gz ---
    z_err = gz - z
    vz_err = -vz
    # compensate for tilt so vertical component of thrust equals -m*g at steady state
    u1_hover = m * (-g) / torch.clamp(torch.cos(th), min=0.2)
    u1 = u1_hover + K_Z * z_err + K_VZ * vz_err

    # --- Lateral: to move RIGHT (increase x), need NEGATIVE theta (see dvx sign) ---
    x_err = gx - x
    vx_err = -vx
    th_des = -(K_X * x_err + K_VX * vx_err)            # <-- sign fixed
    th_des = th_des.clamp(-0.35, 0.35)                 # limit tilt to ~±20°

    # Angle PD (wrap error to [-pi,pi])
    th_err = wrap_angle(th_des - th)
    om_err = -om
    u2 = Iyy * (K_TH * th_err + K_OM * om_err)

    u = torch.cat([u1, u2], dim=-1)
    return clamp_u(dynamics, u)

def dotV(model, dynamics, t, s, u, dvds=None):
    if dvds is None:
        dvds = dvds_at(model, dynamics, t, s)
    f = dynamics.dsdt(s, u, None)
    return (dvds * f).sum(dim=-1)

def shielded_control(model, dynamics, t, s):
    dv = dvds_at(model, dynamics, t, s)
    V_here = value_at(model, dynamics, t, s)[..., 0:1]
    rhs = (-ALPHA * torch.clamp(V_here, min=0.0)).item()

    u_pid = pid_nominal(dynamics, s.clone())
    dV_pid = dotV(model, dynamics, t, s, u_pid.clone(), dv).item()
    if dV_pid <= rhs:
        return u_pid

    # HJ avoid-optimal control
    u_hj = clamp_u(dynamics, dynamics.optimal_control(s, dv).clone())

    # Linearize dotV(u) ≈ b + a^T u
    def dotV_scalar(u_): return dotV(model, dynamics, t, s, u_.clone(), dv).item()
    u_zero = torch.zeros_like(u_pid)
    b = dotV_scalar(u_zero)
    a = torch.zeros(2, device=s.device)
    du1 = torch.tensor([[FD_EPS, 0.0]], device=s.device)
    du2 = torch.tensor([[0.0, FD_EPS]], device=s.device)
    a[0] = (dotV_scalar(du1) - b) / FD_EPS
    a[1] = (dotV_scalar(du2) - b) / FD_EPS

    if (a @ u_pid.squeeze(0) + b) <= rhs:
        return u_pid

    num = (a @ u_pid.squeeze(0) + b - rhs).item()
    den = (a @ (u_pid.squeeze(0) - u_hj.squeeze(0))).item()
    lam = 1.0 if abs(den) < 1e-9 else float(num / den)
    lam = float(max(0.0, min(1.0, lam)))
    u = (1.0 - lam) * u_pid + lam * u_hj
    return clamp_u(dynamics, u)

# =========================== ROLLOUT ===========================
def rollout(model, dynamics, x0, dt, horizon):
    T = int(math.ceil(horizon / dt))
    xs = [x0]
    reached, violated = False, False

    for k in range(T):
        t = k * dt
        s = xs[-1].unsqueeze(0)
        u = shielded_control(model, dynamics, t, s)
        ds = dynamics.dsdt(s, u, None)
        x_next = dynamics.equivalent_wrapped_state(s + ds * dt).squeeze(0)
        xs.append(x_next)

        a = dynamics.avoid_fn(x_next.unsqueeze(0)).item()
        r = dynamics.reach_fn(x_next.unsqueeze(0)).item()
        if a <= 0.0:
            violated = True; break
        if r <= 0.0:
            reached = True; break

    traj = torch.stack(xs, dim=0)
    return traj, reached, violated

# ============================ MAIN =============================
def main():
    torch.set_default_dtype(torch.float32)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    model, dynamics, _ = load_model_and_dynamics(EXPERIMENT_DIR, CHECKPOINT, DEVICE)
    x0s = choose_initial_states(model, dynamics, NUM_ROLLOUTS, DEVICE)

    results = []
    for i in range(x0s.shape[0]):
        traj, reached, violated = rollout(model, dynamics, x0s[i], DT, HORIZON)
        results.append((traj.cpu(), reached, violated))

    n_viol = sum(int(v) for _, _, v in results)
    n_succ = sum(int(r) for _, r, v in results if not v)
    n_timeout = NUM_ROLLOUTS - n_succ - n_viol
    print(f"[Rollouts] total={NUM_ROLLOUTS}  success={n_succ}  violate={n_viol}  timeout={n_timeout}")

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    cx, cz, r = dynamics.obs_cx, dynamics.obs_cz, dynamics.obs_r
    th = np.linspace(0, 2*np.pi, 256)
    ax.plot(cx + r*np.cos(th), cz + r*np.sin(th), linewidth=2, label="obstacle")
    gx, gz, gr = dynamics.goal_x, dynamics.goal_z, dynamics.goal_pos_tol
    ax.plot(gx + gr*np.cos(th), gz + gr*np.sin(th), linewidth=2, linestyle="--", label="goal pos tol")
    ax.set_xlim(0.0, 5.0); ax.set_ylim(0.0, 5.0)

    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(results)))
    start_handle = ax.scatter([], [], color="black", marker="o", s=40)
    end_handle   = ax.scatter([], [], color="black", marker="^", s=45)

    for idx, (traj, reached, violated) in enumerate(results):
        x = traj[:, 0].numpy(); z = traj[:, 1].numpy()
        color = colors[idx]
        if violated:
            ax.plot(x, z, linestyle=":", linewidth=1, color=color)
        elif reached:
            ax.plot(x, z, linestyle="-", linewidth=1.5, color=color)
        else:
            ax.plot(x, z, linestyle="--", linewidth=1, color=color)
        ax.scatter(x[0], z[0], color=color, marker="o", s=40, edgecolor="k", linewidth=0.4, alpha=0.9)
        ax.scatter(x[-1], z[-1], color=color, marker="^", s=45, edgecolor="k", linewidth=0.4, alpha=0.9)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([start_handle, end_handle]); labels.extend(["start", "end"])
    ax.legend(handles, labels, loc="best")
    ax.set_title(f"Quad2D avoid-only shielded rollouts (N={NUM_ROLLOUTS})")

    out_dir = os.path.join(EXPERIMENT_DIR, "testing_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "rollouts_xz.pdf")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"[Saved] {fig_path}")

if __name__ == "__main__":
    main()
