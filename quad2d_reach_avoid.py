import os, math, pickle, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import inspect

# ==== USER SETTINGS (edit as needed) ====
EXPERIMENT_DIR = "deep_reach/runs/quad2d_reachavoid_v0"   # folder that contains training/ and orig_opt.pickle
CHECKPOINT = 150000                                # -1 -> model_final.pth, else integer epoch number
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DT = 0.05                                      # simulation step [s]
HORIZON = 5.0                                  # max sim time [s]
NUM_ROLLOUTS = 100                              # how many trajectories to simulate
SEED = 100                                       # reproducibility
ONLY_START_IN_SAFE_SET = True                  # filter starts with V(0,x) <= 0
OVERSAMPLE_FACTOR = 20                         # for safe-start filtering

# Sampling: which state dims to vary when choosing initial states
# (x, z, th, vx, vz, om) — keep non-spatial dims near zero unless you want aggressive starts.
INIT_RANGES_OVERRIDE = {
    "x":  (0.0, 1.5),
    "z":  (0.0, 5.0),
    "th": (0.0, 0.0),
    "vx": (0.0, 0.0),
    "vz": (0.0, 0.0),
    "om": (0.0, 0.0),
}

# ==== LOADS TRAINED MODEL + DYNAMICS (exactly as used in training) ====
def load_model_and_dynamics(exp_dir, checkpoint, device):
    from utils import modules
    from dynamics import dynamics as dyn_mod

    with open(os.path.join(exp_dir, "orig_opt.pickle"), "rb") as f:
        orig_opt = pickle.load(f)

    # Recreate dynamics from training config (arguments pulled from signature)
    dclass = getattr(dyn_mod, orig_opt.dynamics_class)
    param_names = [name for name in inspect.signature(dclass).parameters.keys() if name != 'self']
    dargs = {name: getattr(orig_opt, name) for name in param_names}
    dynamics = dclass(**dargs)
    dynamics.deepreach_model = orig_opt.deepreach_model  # 'exact' for your quad2d class
    # (io_to_dv & coord/value conversions rely on this; see dynamics base class) :contentReference[oaicite:0]{index=0}

    # Network architecture exactly as during training
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type=orig_opt.model,
        mode=orig_opt.model_mode,
        final_layer_factor=1.,
        hidden_features=orig_opt.num_nl,
        num_hidden_layers=orig_opt.num_hl
    ).to(device)

    # Load weights
    ckpt_dir = os.path.join(exp_dir, "training", "checkpoints")
    if checkpoint == -1:
        sd = torch.load(os.path.join(ckpt_dir, "model_final.pth"), map_location=device)
        model.load_state_dict(sd)
    else:
        sd = torch.load(os.path.join(ckpt_dir, f"model_epoch_{checkpoint:04d}.pth"), map_location=device)
        model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, dynamics, orig_opt

# ==== VALUE & GRAD HELPERS (use DeepReach’s normalization utilities) ====
@torch.no_grad()
def value_at(model, dynamics, t, states):
    # coords=[t, state]; then convert to model input and back to real value
    coords = torch.cat([torch.full((states.shape[0], 1), float(t), device=states.device),
                        states], dim=-1)
    res = model({"coords": dynamics.coord_to_input(coords)})
    V = dynamics.io_to_value(res["model_in"], res["model_out"].squeeze(-1))
    return V

def dvds_at(model, dynamics, t, states):
    # need grad for dv/ds; do not wrap in no_grad
    coords = torch.cat([torch.full((states.shape[0], 1), float(t), device=states.device),
                        states], dim=-1)
    res = model({"coords": dynamics.coord_to_input(coords)})
    # io_to_dv returns [dv/dt, dv/ds...] in real units (handles 'exact' case) :contentReference[oaicite:1]{index=1}
    dv = dynamics.io_to_dv(res["model_in"], res["model_out"].squeeze(-1))
    return dv[..., 1:]  # dv/ds

# ==== SAMPLING INITIAL STATES ====
def sample_uniform(dynamics, n, rngs=None, device="cpu"):
    # Use dynamics.state_test_range() as baseline; optionally override a few dims
    base = dynamics.state_test_range()  # [[xmin,xmax], ...] 6D for quad2d :contentReference[oaicite:2]{index=2}
    keys = ["x","z","th","vx","vz","om"]
    for i,k in enumerate(keys):
        if rngs and (k in rngs):
            a,b = rngs[k]
        else:
            a,b = base[i]
        base[i] = [a,b]
    base = torch.tensor(base, dtype=torch.float32, device=device)
    lo, hi = base[:,0], base[:,1]
    return lo + torch.rand(n, dynamics.state_dim, device=device)*(hi-lo)

def choose_initial_states(model, dynamics, n, device):
    if not ONLY_START_IN_SAFE_SET:
        return sample_uniform(dynamics, n, INIT_RANGES_OVERRIDE, device)
    # oversample, keep those with V(0,x) <= 0 (inside BRT safe set)
    m = n * OVERSAMPLE_FACTOR
    pool = sample_uniform(dynamics, m, INIT_RANGES_OVERRIDE, device)
    V0 = value_at(model, dynamics, 0.0, pool)
    keep = (V0 <= 0.0)
    kept = pool[keep]
    if kept.shape[0] >= n:
        return kept[:n]
    # if not enough, top-up with randoms
    extra = sample_uniform(dynamics, n - kept.shape[0], INIT_RANGES_OVERRIDE, device)
    return torch.cat([kept, extra], dim=0)

# ==== SINGLE ROLLOUT ====
def rollout(model, dynamics, x0, dt, horizon):
    T = int(math.ceil(horizon/dt))
    xs = [x0]
    reached, violated = False, False

    # pre-allocate metrics for RA check
    avoid_min = float("+inf")
    for k in range(T):
        t = k*dt
        s = xs[-1].unsqueeze(0)  # [1,6]
        # controller from value gradient: u* = argmin_u <dv/ds, f(x,u)> (implemented in dynamics.optimal_control)
        dvds = dvds_at(model, dynamics, t, s)         # [1,6]
        u = dynamics.optimal_control(s, dvds)         # [1,2] (quad2d-specific)
        ds = dynamics.dsdt(s, u, None)                # [1,6]
        x_next = dynamics.equivalent_wrapped_state(s + ds*dt).squeeze(0)  # wrap angle
        xs.append(x_next)

        # safety / goal checks
        a = dynamics.avoid_fn(x_next.unsqueeze(0)).item()   # >0 = outside obstacle
        r = dynamics.reach_fn(x_next.unsqueeze(0)).item()   # <=0 = in goal tube
        avoid_min = min(avoid_min, a)
        if a <= 0.0:
            violated = True
            break
        if r <= 0.0:
            reached = True
            break

    traj = torch.stack(xs, dim=0)  # [K,6]
    return traj, reached, violated

def main():
    torch.set_default_dtype(torch.float32)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    model, dynamics, _ = load_model_and_dynamics(EXPERIMENT_DIR, CHECKPOINT, DEVICE)

    # choose initial conditions
    x0s = choose_initial_states(model, dynamics, NUM_ROLLOUTS, DEVICE)  # [N,6]

    # run
    results = []
    for i in range(x0s.shape[0]):
        traj, reached, violated = rollout(model, dynamics, x0s[i], DT, HORIZON)
        results.append((traj.cpu(), reached, violated))

    # metrics
    n_succ   = sum(int(r) for _, r, v in results if not v)
    n_viol   = sum(int(v) for _, r, v in results)
    n_timeout = NUM_ROLLOUTS - n_succ - n_viol
    print(f"[Rollouts] total={NUM_ROLLOUTS}  success={n_succ}  violate={n_viol}  timeout={n_timeout}")

    # ==== PLOT all trajectories in x–z ====
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    # obstacle (disk)
    cx, cz, r = dynamics.obs_cx, dynamics.obs_cz, dynamics.obs_r
    th = np.linspace(0, 2*np.pi, 256)
    ax.plot(cx + r*np.cos(th), cz + r*np.sin(th), linewidth=2, label="obstacle")
    # goal tube (just position tolerance circle)
    gx, gz, gr = dynamics.goal_x, dynamics.goal_z, dynamics.goal_pos_tol
    ax.plot(gx + gr*np.cos(th), gz + gr*np.sin(th), linewidth=2, linestyle="--", label="goal pos tol")

    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 5.0)

    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(results)))
    start_handle = ax.scatter([], [], color="black", marker="o", s=40)
    end_handle = ax.scatter([], [], color="black", marker="^", s=45)

    # rollouts: different linestyles by outcome
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
    handles.extend([start_handle, end_handle])
    labels.extend(["start", "end"])
    ax.legend(handles, labels, loc="best")
    ax.set_title(f"Quad2D Reach-Avoid rollouts (N={NUM_ROLLOUTS})")

    out_dir = os.path.join(EXPERIMENT_DIR, "testing_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "rollouts_xz.pdf")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"[Saved] {fig_path}")

if __name__ == "__main__":
    main()
