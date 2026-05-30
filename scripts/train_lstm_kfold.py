"""Leave-one-sim-out (LOSO) cross-validation for LSTM+PVT (epoch 30 recipe).

For each of the 40 Norne sims (30 baseline + 10 sched_multipvt):
  - train on the other 39 sims for exactly 30 epochs (cosine LR T_max=30)
  - evaluate on the held-out Norne sim (in-distribution generalization)
  - evaluate on all 10 Volve sims      (cross-reservoir generalization)

Output:
  predictions/lstm_kfold_results.csv — one row per fold
  plots_pvt/lstm_kfold_distribution.png — boxplots + per-fold scatter
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train_lstm_multipvt import (  # type: ignore
    DATASETS, DYNAMIC_COLS, LSTMPVT, PRESSURE_SCALE, SEED, STATIC_COLS,
    TARGET, _pvt_vector_from_levers, build_features, build_seq_tensors,
    fit_split_stats, masked_mse, volve_pvt_vector,
)

N_EPOCHS = 30


def evaluate_psi(y_true, y_pred) -> dict:
    return {
        "R2":   float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def load_data():
    candidates = [
        ("dataset_norne_sched_multipvt_n10.csv",
         "runs_log_norne_sched_multipvt_n10.csv"),
        ("dataset_norne_sched_multipvt_n25.csv",
         "runs_log_norne_sched_multipvt_n25.csv"),
    ]
    ds_path = log_path = None
    for d, l in candidates:
        if (DATASETS / d).exists() and (DATASETS / l).exists():
            ds_path, log_path = DATASETS / d, DATASETS / l
            break
    if ds_path is None:
        raise FileNotFoundError(f"no perturbation dataset found ({candidates})")
    print(f"Loading datasets ({ds_path.name})…")
    norne_base = pd.read_csv(DATASETS / "dataset_norne.csv")
    norne_multi = pd.read_csv(ds_path)
    runs_log = pd.read_csv(log_path)
    volve_raw = pd.read_csv(DATASETS / "dataset_volve.csv")

    shift = int(norne_base["sim_id"].max())
    norne_multi = norne_multi.copy()
    norne_multi["sim_id"] = norne_multi["sim_id"] + shift
    train_raw = pd.concat([norne_base, norne_multi], ignore_index=True)
    n_multi = int(norne_multi["sim_id"].nunique())

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

    all_sims = sorted(train_feat["sim_id"].unique().tolist())
    print(f"  fold count: {len(all_sims)} Norne sims, Volve eval rows: {len(volve_feat)}")
    return train_feat, volve_feat, pvt_lookup, all_sims, shift, n_multi


def to_torch(d, device):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "i", "b"):
            out[k] = torch.from_numpy(v).to(device)
        else:
            out[k] = v
    out["mask_f"] = out["mask"].float()
    return out


def collect_preds(model, t_dict, t_np):
    with torch.no_grad():
        delta = model(t_dict["X_dyn"], t_dict["X_static"], t_dict["pvt"]).cpu().numpy()
    y_true, y_pred = [], []
    for i in range(len(t_np["sim_ids"])):
        T = int(t_np["mask"][i].sum())
        y_true.append(t_np["per_sim_meta"][i][TARGET].values[:T])
        y_pred.append(t_np["p_init"][i] + delta[i, :T] * PRESSURE_SCALE)
    return np.concatenate(y_true), np.concatenate(y_pred)


def train_one_fold(holdout_sim: int, train_feat, volve_feat, pvt_lookup,
                   device, shift: int):
    """Train one fold; return metrics dict for held-out sim and Volve."""
    torch.manual_seed(SEED); np.random.seed(SEED)

    val_df = train_feat[train_feat["sim_id"] == holdout_sim].reset_index(drop=True)
    tr_df = train_feat[train_feat["sim_id"] != holdout_sim].reset_index(drop=True)

    mu_dyn, sd_dyn, mu_stat, sd_stat = fit_split_stats(tr_df)
    tr_t = build_seq_tensors(tr_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    va_t = build_seq_tensors(val_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    te_t = build_seq_tensors(volve_feat, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    tr = to_torch(tr_t, device); va = to_torch(va_t, device); te = to_torch(te_t, device)

    model = LSTMPVT(
        n_dyn=len(DYNAMIC_COLS), n_static=len(STATIC_COLS),
        pvt_dim=tr_t["pvt"].shape[1],
        ctx_dim=16, hidden=64, num_layers=2, dropout=0.15,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    for _ in range(N_EPOCHS):
        model.train()
        pred = model(tr["X_dyn"], tr["X_static"], tr["pvt"])
        loss = masked_mse(pred, tr["delta_target"], tr["mask_f"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    model.eval()
    y_tr, p_tr = collect_preds(model, tr, tr_t)
    y_va, p_va = collect_preds(model, va, va_t)
    y_te, p_te = collect_preds(model, te, te_t)

    return {
        "holdout_sim": holdout_sim,
        "is_multipvt": holdout_sim > shift,
        "train":     evaluate_psi(y_tr, p_tr),
        "holdout":   evaluate_psi(y_va, p_va),
        "volve":     evaluate_psi(y_te, p_te),
    }


def main():
    train_feat, volve_feat, pvt_lookup, all_sims, shift, n_multi = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    rows = []
    t0 = time.time()
    for k, sim in enumerate(all_sims, 1):
        ts = time.time()
        m = train_one_fold(sim, train_feat, volve_feat, pvt_lookup, device, shift)
        dt = time.time() - ts
        rows.append({
            "fold": k,
            "holdout_sim": sim,
            "kind": "sched_multipvt" if m["is_multipvt"] else "baseline",
            "train_R2": m["train"]["R2"],     "train_MAE": m["train"]["MAE"],
            "holdout_R2": m["holdout"]["R2"], "holdout_MAE": m["holdout"]["MAE"],
            "volve_R2": m["volve"]["R2"],     "volve_MAE": m["volve"]["MAE"],
            "fold_seconds": round(dt, 2),
        })
        print(f"  fold {k:>2}/{len(all_sims)}  sim={sim:>2} ({rows[-1]['kind']:<14})  "
              f"train R²={m['train']['R2']:+.3f}  "
              f"holdout R²={m['holdout']['R2']:+.3f} MAE={m['holdout']['MAE']:.0f}  "
              f"Volve R²={m['volve']['R2']:+.3f} MAE={m['volve']['MAE']:.0f}  "
              f"[{dt:.1f}s]")

    df = pd.DataFrame(rows)
    out_csv = ROOT / "predictions" / "lstm_kfold_results.csv"
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Wrote per-fold results → {out_csv}")

    # ---- aggregate summary ----------------------------------------------
    def stats(s: pd.Series) -> str:
        return (f"median={s.median():+.3f}  mean={s.mean():+.3f}  "
                f"std={s.std():.3f}  [min={s.min():+.3f}, max={s.max():+.3f}]")

    print("\n=== LOSO summary ===")
    print(f"  holdout R² (Norne in-dist, 40 folds)   {stats(df['holdout_R2'])}")
    print(f"  holdout MAE [psi]                       "
          f"median={df['holdout_MAE'].median():.1f}  "
          f"mean={df['holdout_MAE'].mean():.1f}  std={df['holdout_MAE'].std():.1f}  "
          f"[min={df['holdout_MAE'].min():.1f}, max={df['holdout_MAE'].max():.1f}]")
    print(f"  Volve R² (cross-reservoir, 40 folds)   {stats(df['volve_R2'])}")
    print(f"  Volve MAE [psi]                         "
          f"median={df['volve_MAE'].median():.1f}  "
          f"mean={df['volve_MAE'].mean():.1f}  std={df['volve_MAE'].std():.1f}  "
          f"[min={df['volve_MAE'].min():.1f}, max={df['volve_MAE'].max():.1f}]")

    print("\nWorst 5 folds by holdout R²:")
    worst = df.sort_values("holdout_R2").head(5)
    for _, r in worst.iterrows():
        print(f"  fold {int(r['fold']):>2}  sim={int(r['holdout_sim']):>2} ({r['kind']:<14})  "
              f"holdout R²={r['holdout_R2']:+.3f}  MAE={r['holdout_MAE']:.0f}")

    # ---- plots ----------------------------------------------------------
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(df["fold"], df["holdout_R2"],
               c=df["kind"].map({"baseline": "C0", "sched_multipvt": "C3"}),
               s=35, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axhline(df["holdout_R2"].median(), color="C0", lw=1.0, ls="--",
               label=f"median {df['holdout_R2'].median():+.2f}")
    ax.set_xlabel("fold"); ax.set_ylabel("holdout R²")
    ax.set_title("LOSO — held-out Norne sim R² (per fold)")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.scatter(df["fold"], df["volve_R2"],
               c=df["kind"].map({"baseline": "C0", "sched_multipvt": "C3"}),
               s=35, alpha=0.85)
    ax.axhline(0.215, color="gray", lw=0.6, ls=":", label="Naive Volve (+0.215)")
    ax.axhline(df["volve_R2"].median(), color="C2", lw=1.0, ls="--",
               label=f"median {df['volve_R2'].median():+.2f}")
    ax.set_xlabel("fold"); ax.set_ylabel("Volve R²")
    ax.set_title("LOSO — Volve cross-reservoir R² per fold")
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 0]
    bp = ax.boxplot(
        [df["holdout_R2"], df["volve_R2"]],
        labels=["Norne holdout\n(40 folds)", "Volve cross\n(40 folds)"],
        showmeans=True, patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C2"]):
        patch.set_facecolor(color); patch.set_alpha(0.4)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("R²")
    ax.set_title("R² distribution across LOSO folds")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    bp = ax.boxplot(
        [df["holdout_MAE"], df["volve_MAE"]],
        labels=["Norne holdout\n(40 folds)", "Volve cross\n(40 folds)"],
        showmeans=True, patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["C0", "C2"]):
        patch.set_facecolor(color); patch.set_alpha(0.4)
    ax.axhline(197.1, color="gray", lw=0.6, ls=":", label="Naive Volve MAE")
    ax.set_ylabel("MAE [psi]")
    ax.set_title("MAE distribution across LOSO folds")
    ax.grid(alpha=0.3, axis="y"); ax.legend()

    fig.suptitle(
        f"LSTM+PVT epoch-30 LOSO ({len(all_sims)} Norne folds). "
        f"Blue = baseline sim held out, red = sched_multipvt sim held out",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path = ROOT / "plots_pvt" / "lstm_kfold_distribution.png"
    plot_path.parent.mkdir(exist_ok=True)
    fig.savefig(plot_path, dpi=110); plt.close(fig)
    print(f"\nWrote plot → {plot_path}")


if __name__ == "__main__":
    sys.exit(main())
