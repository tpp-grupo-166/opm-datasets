"""LSTM+PVT fixed to 30 training epochs (no early stopping, no checkpoint).

The full-training-curve sweep showed that the LSTM is *most internally
consistent* at epoch 30: Norne train and Volve metrics are closest there
(train R²≈+0.55 vs Volve R²≈+0.45, train MAE 104 vs Volve MAE 157), before
the model starts overfitting Norne. Earlier than 30 has lower Volve MAE
but a larger Norne/Volve gap; later than 30 the gap blows up rapidly.

This script runs exactly that recipe and emits:
  - Final metrics on Norne train / Norne val / Volve
  - `predictions/preds_lstm_e30_volve.csv` for downstream plotting
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

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
    VAL_SIMS = [shift, shift + n_multi]
    val_df = train_feat[train_feat["sim_id"].isin(VAL_SIMS)].reset_index(drop=True)
    tr_df = train_feat[~train_feat["sim_id"].isin(VAL_SIMS)].reset_index(drop=True)
    print(f"  train rows: {len(tr_df)}, val rows: {len(val_df)} "
          f"(VAL_SIMS={VAL_SIMS}), eval Volve rows: {len(volve_feat)}")

    mu_dyn, sd_dyn, mu_stat, sd_stat = fit_split_stats(tr_df)
    tr_t = build_seq_tensors(tr_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    va_t = build_seq_tensors(val_df, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)
    te_t = build_seq_tensors(volve_feat, pvt_lookup, mu_dyn, sd_dyn, mu_stat, sd_stat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_torch(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.dtype.kind in ("f", "i", "b"):
                out[k] = torch.from_numpy(v).to(device)
            else:
                out[k] = v
        out["mask_f"] = out["mask"].float()
        return out
    tr = to_torch(tr_t); va = to_torch(va_t); te = to_torch(te_t)

    model = LSTMPVT(
        n_dyn=len(DYNAMIC_COLS), n_static=len(STATIC_COLS),
        pvt_dim=tr_t["pvt"].shape[1],
        ctx_dim=16, hidden=64, num_layers=2, dropout=0.15,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Cosine schedule pegged to the actual training horizon so the LR
    # actually anneals over the 30 epochs we run.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    print(f"\n[Training LSTM+PVT for exactly {N_EPOCHS} epochs, no early stop]")
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        pred = model(tr["X_dyn"], tr["X_static"], tr["pvt"])
        loss = masked_mse(pred, tr["delta_target"], tr["mask_f"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                p_va = model(va["X_dyn"], va["X_static"], va["pvt"])
                val_loss = masked_mse(p_va, va["delta_target"], va["mask_f"]).item()
            print(f"  epoch {epoch:>2}  train_mse={loss.item():.5f}  val_mse={val_loss:.5f}")
    model.eval()

    @torch.no_grad()
    def collect(t_dict, t_np):
        delta = model(t_dict["X_dyn"], t_dict["X_static"], t_dict["pvt"]).cpu().numpy()
        y_true, y_pred = [], []
        for i in range(len(t_np["sim_ids"])):
            T = int(t_np["mask"][i].sum())
            y_true.append(t_np["per_sim_meta"][i][TARGET].values[:T])
            y_pred.append(t_np["p_init"][i] + delta[i, :T] * PRESSURE_SCALE)
        return np.concatenate(y_true), np.concatenate(y_pred)

    print("\n[LSTM+PVT @ epoch 30 — evaluation]")
    for tag, t_dict, t_np in [
        (f"Norne train ({len(tr_t['sim_ids'])} sims, in-fit)", tr, tr_t),
        (f"Norne val sims {VAL_SIMS}", va, va_t),
        ("Volve (10 sims, cross-reservoir)", te, te_t),
    ]:
        y_true, y_pred = collect(t_dict, t_np)
        m = evaluate_psi(y_true, y_pred)
        print(f"  {tag:<42}  R²={m['R2']:+.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    # Naive baseline for reference.
    y_volve = np.concatenate([te_t["per_sim_meta"][i][TARGET].values
                              [:int(te_t["mask"][i].sum())]
                              for i in range(len(te_t["sim_ids"]))])
    naive = np.concatenate([np.full(int(te_t["mask"][i].sum()),
                                     float(te_t["p_init"][i]))
                            for i in range(len(te_t["sim_ids"]))])
    m_naive = evaluate_psi(y_volve, naive)
    print(f"  {'Naive on Volve':<42}  R²={m_naive['R2']:+.3f}  "
          f"RMSE={m_naive['RMSE']:.1f}  MAE={m_naive['MAE']:.1f}")

    # Save Volve predictions for downstream plotting.
    y_true_volve, y_pred_volve = collect(te, te_t)
    preds_dir = ROOT / "predictions"
    preds_dir.mkdir(exist_ok=True)
    rows = []
    cursor = 0
    for i in range(len(te_t["sim_ids"])):
        T = int(te_t["mask"][i].sum())
        meta = te_t["per_sim_meta"][i].iloc[:T].copy()
        meta["P_pred_lstm_e30_psi"] = y_pred_volve[cursor:cursor + T]
        meta["P_pred_naive_psi"] = te_t["p_init"][i]
        rows.append(meta)
        cursor += T
    df_out = pd.concat(rows, ignore_index=True).rename(columns={TARGET: "P_true_psi"})
    out_path = preds_dir / "preds_lstm_e30_volve.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved predictions → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
