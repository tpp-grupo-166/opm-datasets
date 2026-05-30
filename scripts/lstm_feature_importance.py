"""Permutation feature importance for the LSTM+PVT-encoder.

Re-trains the LSTM with the same setup as `train_lstm_multipvt.py` and
measures, for every input feature, how much Volve MAE worsens when the
feature is permuted across sims (keeping each sim's temporal pattern
intact but shuffling which sim's pattern goes with which sim).

Higher MAE delta = feature contributes more to the model's predictions.

Three groups of features:
  - 9 dynamic features (per-timestep cumulatives + rates + ratios)
  - 3 static features (porosity, log10 perm, P_initial)
  - 4 PVT-vector slot groups (Pb, Bo curve, Bg curve, Rs curve)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the data + model from the training script.
from train_lstm_multipvt import (  # type: ignore
    DATASETS, DYNAMIC_COLS, FEATURE_COLS, LSTMPVT, P_GRID_PSI,
    PRESSURE_SCALE, SEED, STATIC_COLS, TARGET, build_features,
    build_seq_tensors, fit_split_stats, masked_mse,
    _pvt_vector_from_levers, volve_pvt_vector,
)


PLOTS = ROOT / "plots_pvt"
PLOTS.mkdir(exist_ok=True)


# Groups for the PVT vector (52-dim layout: 1 Pb_norm + 17 Bo + 17 Bg + 17 Rs).
PVT_GROUPS: list[tuple[str, slice]] = [
    ("PVT.Pb_norm",       slice(0, 1)),
    ("PVT.Bo curve",      slice(1, 18)),
    ("PVT.Bg curve",      slice(18, 35)),
    ("PVT.Rs curve",      slice(35, 52)),
]


# ---------------------------------------------------------------------------
# Data setup — same as train_lstm_multipvt.main(), inlined.
# ---------------------------------------------------------------------------

def setup_data():
    candidates = [
        ("dataset_norne_sched_multipvt_n10.csv",
         "runs_log_norne_sched_multipvt_n10.csv"),
        ("dataset_norne_sched_multipvt_n25.csv",
         "runs_log_norne_sched_multipvt_n25.csv"),
    ]
    ds_path = log_path = None
    for d, l in candidates:
        if (DATASETS / d).exists() and (DATASETS / l).exists():
            ds_path = DATASETS / d; log_path = DATASETS / l
            break
    if ds_path is None:
        raise FileNotFoundError(f"no perturbation dataset found: {candidates}")
    print(f"Loading {ds_path.name}…")

    norne_base = pd.read_csv(DATASETS / "dataset_norne.csv")
    norne_multi = pd.read_csv(ds_path)
    runs_log = pd.read_csv(log_path)
    volve_raw = pd.read_csv(DATASETS / "dataset_volve.csv")

    shift = int(norne_base["sim_id"].max())
    norne_multi = norne_multi.copy()
    norne_multi["sim_id"] = norne_multi["sim_id"] + shift
    train_raw = pd.concat([norne_base, norne_multi], ignore_index=True)
    n_multi = int(norne_multi["sim_id"].nunique())

    # Per-sim PVT vectors.
    base_vec = _pvt_vector_from_levers(0.0, 1.0, 1.0, 1.0, 1.0)
    pvt_lookup: dict[tuple, np.ndarray] = {}
    for sid in norne_base["sim_id"].unique():
        pvt_lookup[("norne", int(sid))] = base_vec
    ok_log = runs_log[runs_log["ok"]].copy()
    for _, row in ok_log.iterrows():
        sid = int(row["sim_id"]) + shift
        pvt_lookup[("norne", sid)] = _pvt_vector_from_levers(
            pb_shift_bar=float(row["pb_shift_bar"]),
            rs_mult=float(row["rs_pb_mult"]),
            bo_slope_mult=float(row["bo_undersat_slope_mult"]),
            oil_density_mult=float(row["oil_density_mult"]),
            bg_mult=float(row.get("bg_mult", 1.0)),
        )
    vol_vec = volve_pvt_vector()
    for sid in volve_raw["sim_id"].unique():
        pvt_lookup[("volve", int(sid))] = vol_vec

    train_feat = build_features(train_raw)
    volve_feat = build_features(volve_raw)

    VAL_SIMS = [shift, shift + n_multi]
    val_df = train_feat[train_feat["sim_id"].isin(VAL_SIMS)].reset_index(drop=True)
    tr_df = train_feat[~train_feat["sim_id"].isin(VAL_SIMS)].reset_index(drop=True)

    mu_dyn, sd_dyn, mu_stat, sd_stat = fit_split_stats(tr_df)
    tr_t = build_seq_tensors(tr_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    va_t = build_seq_tensors(val_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    te_t = build_seq_tensors(volve_feat, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    print(f"  train sims={len(tr_t['sim_ids'])}, val sims={len(va_t['sim_ids'])}, "
          f"eval Volve sims={len(te_t['sim_ids'])}")
    return tr_t, va_t, te_t


# ---------------------------------------------------------------------------
# Train (same hyperparams as train_lstm_multipvt).
# ---------------------------------------------------------------------------

def train_lstm(tr_t, va_t, device):
    def to_torch(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "i", "b"):
                out[k] = torch.from_numpy(v).to(device)
            else:
                out[k] = v
        out["mask_f"] = out["mask"].float()
        return out

    tr = to_torch(tr_t); va = to_torch(va_t)
    model = LSTMPVT(
        n_dyn=len(DYNAMIC_COLS), n_static=len(STATIC_COLS),
        pvt_dim=tr_t["pvt"].shape[1],
        ctx_dim=16, hidden=64, num_layers=2, dropout=0.15,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)

    best_val = float("inf"); best_state = None; bad = 0
    for epoch in range(1, 601):
        model.train()
        pred = model(tr["X_dyn"], tr["X_static"], tr["pvt"])
        loss = masked_mse(pred, tr["delta_target"], tr["mask_f"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            p_va = model(va["X_dyn"], va["X_static"], va["pvt"])
            val_loss = masked_mse(p_va, va["delta_target"], va["mask_f"]).item()
        if val_loss < best_val - 1e-7:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 80:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation + permutation importance.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_volve_mae(model, te_t, device) -> float:
    X_dyn = torch.from_numpy(te_t["X_dyn"]).to(device)
    X_static = torch.from_numpy(te_t["X_static"]).to(device)
    pvt = torch.from_numpy(te_t["pvt"]).to(device)
    delta = model(X_dyn, X_static, pvt).cpu().numpy()
    y_true, y_pred = [], []
    for i in range(len(te_t["sim_ids"])):
        T = int(te_t["mask"][i].sum())
        y_true.append(te_t["per_sim_meta"][i][TARGET].values[:T])
        y_pred.append(te_t["p_init"][i] + delta[i, :T] * PRESSURE_SCALE)
    return mean_absolute_error(np.concatenate(y_true), np.concatenate(y_pred))


def permutation_importance(model, te_t, device, n_repeats: int = 5,
                            tr_pvt: np.ndarray | None = None) -> pd.DataFrame:
    """Permutation importance on Volve. For dynamic/static features we
    shuffle across Volve sims. For the PVT vector slots — where all Volve
    sims share the same canonical PVT — we instead swap with random
    training PVTs (per Volve sim), which is the meaningful test of
    "does the model rely on the PVT context"."""
    rng = np.random.default_rng(42)
    base_mae = evaluate_volve_mae(model, te_t, device)
    print(f"  baseline Volve MAE = {base_mae:.2f} psi")

    n_sims = te_t["X_dyn"].shape[0]
    rows: list[dict] = []

    # Dynamic features — per-feature permutation across sims (keep each sim's
    # temporal pattern; just swap which sim it belongs to).
    for k, name in enumerate(DYNAMIC_COLS):
        deltas = []
        for r in range(n_repeats):
            perm = rng.permutation(n_sims)
            orig = te_t["X_dyn"][:, :, k].copy()
            te_t["X_dyn"][:, :, k] = te_t["X_dyn"][perm, :, k]
            mae = evaluate_volve_mae(model, te_t, device)
            te_t["X_dyn"][:, :, k] = orig
            deltas.append(mae - base_mae)
        rows.append({
            "group": "dynamic", "feature": name,
            "mae_delta": float(np.mean(deltas)),
            "mae_delta_std": float(np.std(deltas)),
        })

    # Static features.
    for k, name in enumerate(STATIC_COLS):
        deltas = []
        for r in range(n_repeats):
            perm = rng.permutation(n_sims)
            orig = te_t["X_static"][:, k].copy()
            te_t["X_static"][:, k] = te_t["X_static"][perm, k]
            mae = evaluate_volve_mae(model, te_t, device)
            te_t["X_static"][:, k] = orig
            deltas.append(mae - base_mae)
        rows.append({
            "group": "static", "feature": name,
            "mae_delta": float(np.mean(deltas)),
            "mae_delta_std": float(np.std(deltas)),
        })

    # PVT-vector slot groups. Volve sims all share the same canonical PVT
    # vector, so permuting within Volve is a no-op. Instead, swap each
    # Volve sim's PVT slot with a random training-set PVT — this measures
    # whether the model relies on the Volve PVT signal vs ignores it.
    if tr_pvt is None:
        for name, sl in PVT_GROUPS:
            rows.append({
                "group": "pvt", "feature": name,
                "mae_delta": float("nan"),
                "mae_delta_std": float("nan"),
            })
    else:
        n_tr = tr_pvt.shape[0]
        for name, sl in PVT_GROUPS:
            deltas = []
            for r in range(n_repeats):
                pick = rng.integers(0, n_tr, size=n_sims)
                orig = te_t["pvt"][:, sl].copy()
                te_t["pvt"][:, sl] = tr_pvt[pick][:, sl]
                mae = evaluate_volve_mae(model, te_t, device)
                te_t["pvt"][:, sl] = orig
                deltas.append(mae - base_mae)
            rows.append({
                "group": "pvt", "feature": name,
                "mae_delta": float(np.mean(deltas)),
                "mae_delta_std": float(np.std(deltas)),
            })

    return pd.DataFrame(rows), base_mae


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------

def plot_importance(imp_df: pd.DataFrame, base_mae: float, out_path: Path) -> None:
    imp_df = imp_df.sort_values("mae_delta", ascending=True).reset_index(drop=True)
    colors_by_group = {"dynamic": "C0", "static": "C1", "pvt": "C2"}
    colors = [colors_by_group[g] for g in imp_df["group"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp_df["feature"], imp_df["mae_delta"],
            xerr=imp_df["mae_delta_std"], color=colors, alpha=0.85)
    ax.set_xlabel("Δ MAE on Volve when permuted across sims [psi]")
    ax.set_title(
        f"LSTM+PVT permutation importance — Volve baseline MAE = {base_mae:.1f} psi"
    )
    ax.grid(alpha=0.3, axis="x")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.85, label="dynamic"),
        plt.Rectangle((0, 0), 1, 1, color="C1", alpha=0.85, label="static"),
        plt.Rectangle((0, 0), 1, 1, color="C2", alpha=0.85, label="PVT vector"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up data…")
    tr_t, va_t, te_t = setup_data()

    print("\nTraining LSTM…")
    model = train_lstm(tr_t, va_t, device)
    print("  done")

    print("\nComputing permutation importance on Volve (5 repeats per feature)…")
    imp_df, base_mae = permutation_importance(
        model, te_t, device, n_repeats=5, tr_pvt=tr_t["pvt"],
    )

    print("\nFeature importances (sorted, Δ MAE in psi):")
    print(imp_df.sort_values("mae_delta", ascending=False).to_string(index=False))

    out_path = PLOTS / "lstm_feature_importance.png"
    plot_importance(imp_df, base_mae, out_path)


if __name__ == "__main__":
    main()
