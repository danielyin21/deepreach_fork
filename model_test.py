# safe_set_quad2d_avoid.py
# Clean safe-set visualization for Quad2DReachAvoid (avoid-only).

import os, pickle, inspect
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ===== USER SETTINGS =====
EXPERIMENT_DIR = "deep_reach/runs/quad2d_reachavoid_avoid"   # set to your run dir
CHECKPOINT     = 100000   # -1 -> model_final.pth; else epoch number (e.g., 5000)
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
NX, NZ         = 300, 300   # grid resolution
TIME_FRACTIONS = [0.0, 0.5, 1.0]  # t = tMin + frac*(tMax - tMin)
# =========================

# --- Load training config ---
with open(os.path.join(EXPERIMENT_DIR, "orig_opt.pickle"), "rb") as f:
    orig_opt = pickle.load(f)

# --- Project imports ---
from dynamics import dynamics as dyn_mod
from utils import modules  # SingleBVPNet

# --- Instantiate dynamics (pass only ctor params that exist in orig_opt) ---
dyn_name = getattr(orig_opt, "dynamics_class", None)
if dyn_name is None:
    raise RuntimeError("orig_opt lacks 'dynamics_class'.")
dyn_cls = getattr(dyn_mod, dyn_name)

sig = inspect.signature(dyn_cls.__init__)
dyn_kwargs = {}
missing_required = []
for name, param in sig.parameters.items():
    if name == "self":
        continue
    if hasattr(orig_opt, name):
        dyn_kwargs[name] = getattr(orig_opt, name)
    elif param.default is inspect._empty:
        missing_required.append(name)
if missing_required:
    raise RuntimeError(f"Missing required ctor args for {dyn_name}: {missing_required}")

dynamics = dyn_cls(**dyn_kwargs)

# Force avoid-only mode if available
if hasattr(dynamics, "set_mode"):
    try: dynamics.set_mode("avoid")
    except Exception: pass
elif hasattr(dynamics, "mode"):
    setattr(dynamics, "mode", "avoid")

# Keep flag consistent if present
if hasattr(orig_opt, "deepreach_model") and hasattr(dynamics, "deepreach_model"):
    dynamics.deepreach_model = orig_opt.deepreach_model

# --- Build model with same hypers as training ---
in_features = getattr(dynamics, "input_dim", None)
if in_features is None:
    state_dim = getattr(dynamics, "state_dim", None)
    if state_dim is None:
        raise RuntimeError("Cannot infer input_dim (no dynamics.input_dim/state_dim).")
    in_features = 1 + state_dim  # [t] + state

model = modules.SingleBVPNet(
    in_features=in_features,
    out_features=1,
    type=getattr(orig_opt, "model", "sine"),
    mode=getattr(orig_opt, "model_mode", "mlp"),
    final_layer_factor=1.0,
    hidden_features=getattr(orig_opt, "num_nl", 256),
    num_hidden_layers=getattr(orig_opt, "num_hl", 4),
).to(DEVICE)
model.eval()
for p in model.parameters(): p.requires_grad_(False)

# --- Load checkpoint (supports final or epoch format) ---
ckpt_dir = os.path.join(EXPERIMENT_DIR, "training", "checkpoints")
if CHECKPOINT == -1:
    ckpt_path = os.path.join(ckpt_dir, "model_final.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing {ckpt_path}. Set CHECKPOINT to an existing epoch.")
    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
else:
    ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{CHECKPOINT:04d}.pth")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

# --- Plot config & ranges ---
def _plot_cfg(dyn):
    if hasattr(dyn, "plot_config"): return dyn.plot_config()
    return {
        "x_axis_idx": 0, "y_axis_idx": 1,
        "state_slices": [0., 0., 0., 0., 0., 0.],
        "state_labels": ["x", "z", "theta", "vx", "vz", "omega"],
    }

def _ranges(dyn):
    if hasattr(dyn, "state_test_range"): return dyn.state_test_range()
    if hasattr(dyn, "state_range"): return dyn.state_range()
    return [(-4, 4), (-1, 5), (-0.6, 0.6), (-2, 2), (-2, 2), (-3, 3)]

pcfg = _plot_cfg(dynamics)
ranges = _ranges(dynamics)
x_min, x_max = ranges[pcfg["x_axis_idx"]]
z_min, z_max = ranges[pcfg["y_axis_idx"]]

xs = torch.linspace(x_min, x_max, NX)
zs = torch.linspace(z_min, z_max, NZ)
X, Z = torch.meshgrid(xs, zs, indexing="xy")

state_dim = getattr(dynamics, "state_dim", len(pcfg["state_slices"]))
base_slice = torch.tensor(pcfg["state_slices"], dtype=torch.float32)
if len(base_slice) != state_dim:
    base_slice = (torch.nn.functional.pad(base_slice, (0, state_dim - len(base_slice)))
                  if len(base_slice) < state_dim else base_slice[:state_dim])

tMin = float(getattr(orig_opt, "tMin", 0.0))
tMax = float(getattr(orig_opt, "tMax", 1.0))
times = [tMin + f*(tMax - tMin) for f in TIME_FRACTIONS]

has_c2i = hasattr(dynamics, "coord_to_input")
has_i2v = hasattr(dynamics, "io_to_value")

@torch.no_grad()
def eval_value_at_time(t_val: float) -> np.ndarray:
    coords = base_slice.repeat(NX * NZ, 1)
    coords[:, pcfg["x_axis_idx"]] = X.reshape(-1)
    coords[:, pcfg["y_axis_idx"]] = Z.reshape(-1)
    t_col = torch.full((coords.shape[0], 1), t_val, dtype=torch.float32)
    full = torch.cat([t_col, coords], dim=-1).to(DEVICE)
    model_in = dynamics.coord_to_input(full) if has_c2i else full
    out = model({"coords": model_in})
    raw = out["model_out"].squeeze(-1)
    V = dynamics.io_to_value(out.get("model_in", model_in), raw) if has_i2v else raw
    return V.detach().cpu().numpy().reshape(NZ, NX)

# --- Plotting ---
fig, axes = plt.subplots(1, len(times), figsize=(6*len(times), 6), constrained_layout=True)
if len(times) == 1: axes = [axes]

safe_fracs = []
for ax, t_val in zip(axes, times):
    V = eval_value_at_time(t_val)
    safe_mask = (V > 0)
    safe_fracs.append(float(safe_mask.mean()))

    im = ax.imshow(V[::-1, :], extent=[x_min, x_max, z_min, z_max], origin="lower", cmap="coolwarm")
    CS = ax.contour(xs.numpy(), zs.numpy(), V, levels=[0.0], colors="k", linewidths=2)
    ax.clabel(CS, inline=True, fmt={0.0: "V=0"}, fontsize=9)

    cx = getattr(dynamics, "obs_cx", None)
    cz = getattr(dynamics, "obs_cz", None)
    rr = getattr(dynamics, "obs_r", None)
    if cx is not None and cz is not None and rr is not None and rr > 0:
        ax.add_patch(Circle((cx, cz), rr, edgecolor="black", facecolor="none", linestyle="--", linewidth=1.5))

    labels = pcfg.get("state_labels", None)
    xlab = labels[pcfg["x_axis_idx"]] if labels else "x"
    zlab = labels[pcfg["y_axis_idx"]] if labels else "z"
    ax.set_xlabel(xlab); ax.set_ylabel(zlab); ax.set_aspect("equal")
    ax.set_title(f"t={t_val:.3f} | safe frac={safe_mask.mean():.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(t,x)")

out_name = f"safe_set_quad2davoid_{'final' if CHECKPOINT==-1 else f'epoch_{CHECKPOINT:04d}'}.pdf"
out_path = os.path.join(EXPERIMENT_DIR, out_name)
plt.savefig(out_path, dpi=200)
print(f"[OK] Saved: {out_path}")
for i, (t_val, frac) in enumerate(zip(times, safe_fracs)):
    print(f"  - t[{i}] = {t_val:.3f}  safe fraction (V>0): {frac:.4f}")
# plt.close(fig)
plt.savefig(out_path, dpi=200)
