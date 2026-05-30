"""LSTM seq-to-seq on the multi-PVT pilot dataset.

Mirrors `scripts/train_lstm.py` but uses:
  - Training set = baseline 30 + combined sched_multipvt 10 = 40 sims
  - Per-sim PVT vector built from the 5 PVT levers in runs_log (same
    helper as `train_pvt_input_multipvt.py`)
  - Same Δ-target framing and feature split (static / dynamic) as before

We hold out sim 30 (last baseline) and sim 40 (last combined) for early
stopping. Eval = Volve, with per-sim Volve PVT from `pvt_volve.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
SEED = 42
M3_TO_BBL = 6.28981
PRESSURE_SCALE = 5000.0
BAR_TO_PSI = 14.5037738
TARGET = "Presion_Reservorio_psi"

P_GRID_PSI = np.linspace(1500.0, 5500.0, 17)

FEATURE_COLS = [
    "Porosidad", "log10_Permeabilidad_mD", "P_initial_norm",
    "Np_over_PV", "Wp_over_PV", "Winj_over_PV", "GOR_cum",
    "qo_over_PV", "qwinj_over_PV", "WOR_inst", "water_cut_cum", "VRR_simple",
]
STATIC_COLS = ["Porosidad", "log10_Permeabilidad_mD", "P_initial_norm"]
DYNAMIC_COLS = [c for c in FEATURE_COLS if c not in STATIC_COLS]


# ---------------------------------------------------------------------------
# Feature engineering (identical to train_pvt_input_multipvt.py).
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["sim_id"] = df["sim_id"]
    out["reservoir_id"] = df["reservoir_id"]
    out["tiempo_dias"] = df["tiempo_dias"]
    out[TARGET] = df[TARGET]

    pv_bbl = df["Area"] * df["Espesor_Neto_m"] * df["Porosidad"] * M3_TO_BBL
    out["Porosidad"] = df["Porosidad"]
    out["log10_Permeabilidad_mD"] = np.log10(df["Permeabilidad_mD"].clip(lower=1e-3))
    p_first = df.groupby(["reservoir_id", "sim_id"])[TARGET].transform("first")
    out["P_initial_norm"] = p_first / PRESSURE_SCALE
    out["Np_over_PV"] = df["Prod_Acumulada_Petroleo"] / pv_bbl
    out["Wp_over_PV"] = df["Prod_Acumulada_Agua"] / pv_bbl
    out["Winj_over_PV"] = df["Iny_Acumulada_Agua"] / pv_bbl
    with np.errstate(divide="ignore", invalid="ignore"):
        out["GOR_cum"] = np.where(
            df["Prod_Acumulada_Petroleo"] > 0,
            df["Prod_Acumulada_Gas"] / df["Prod_Acumulada_Petroleo"], 0.0)
    out["qo_over_PV"] = df["Caudal_Prod_Petroleo_bbl"] / pv_bbl
    out["qwinj_over_PV"] = df["Caudal_Iny_Agua_bbl"] / pv_bbl
    g = df.groupby(["reservoir_id", "sim_id"])
    dt = g["tiempo_dias"].diff().replace(0, np.nan)
    wp_rate = g["Prod_Acumulada_Agua"].diff() / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        out["WOR_inst"] = np.where(
            df["Caudal_Prod_Petroleo_bbl"] > 0,
            wp_rate / df["Caudal_Prod_Petroleo_bbl"], 0.0)
    total_liq = df["Prod_Acumulada_Petroleo"] + df["Prod_Acumulada_Agua"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["water_cut_cum"] = np.where(
            total_liq > 0, df["Prod_Acumulada_Agua"] / total_liq, 0.0)
        out["VRR_simple"] = np.where(
            total_liq > 0, df["Iny_Acumulada_Agua"] / total_liq, 0.0)
    return out.fillna(0.0)


# ---------------------------------------------------------------------------
# PVT vector — analytical 5-lever Norne perturbation + canonical Volve.
# ---------------------------------------------------------------------------

def _load_base_norne_pvt() -> dict:
    t = json.loads((DATASETS / "pvt_norne.json").read_text())
    return {
        "pb_psi": float(t["pb_psi"]),
        "p_grid": np.asarray(t["p_grid_psi"], dtype=np.float64),
        "bo": np.asarray(t["bo_rb_stb"], dtype=np.float64),
        "bg": np.asarray(t["bg_rb_scf"], dtype=np.float64),
        "rs": np.asarray(t["rs_scf_stb"], dtype=np.float64),
    }


_BASE_NORNE: dict | None = None


def _pvt_vector_from_levers(pb_shift_bar, rs_mult, bo_slope_mult,
                             oil_density_mult, bg_mult=1.0) -> np.ndarray:
    global _BASE_NORNE
    if _BASE_NORNE is None:
        _BASE_NORNE = _load_base_norne_pvt()
    base = _BASE_NORNE
    dp_psi = pb_shift_bar * BAR_TO_PSI
    pb_new = base["pb_psi"] + dp_psi
    p_query = P_GRID_PSI - dp_psi
    bo = np.interp(p_query, base["p_grid"], base["bo"])
    bg = np.interp(p_query, base["p_grid"], base["bg"])
    rs = np.interp(p_query, base["p_grid"], base["rs"])
    rs = rs * rs_mult
    bo_at_pb = float(np.interp(base["pb_psi"], base["p_grid"], base["bo"]))
    undersat = P_GRID_PSI > pb_new
    bo[undersat] = bo_at_pb + bo_slope_mult * (bo[undersat] - bo_at_pb)
    bo = bo * oil_density_mult
    bg = bg * bg_mult
    return np.concatenate([
        [pb_new / PRESSURE_SCALE], bo, bg * 1000.0, rs / 1000.0,
    ]).astype(np.float32)


def volve_pvt_vector() -> np.ndarray:
    t = json.loads((DATASETS / "pvt_volve.json").read_text())
    pb = float(t["pb_psi"]) / PRESSURE_SCALE
    bo = np.asarray(t["bo_rb_stb"])
    bg = np.asarray(t["bg_rb_scf"]) * 1000.0
    rs = np.asarray(t["rs_scf_stb"]) / 1000.0
    return np.concatenate([[pb], bo, bg, rs]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-sim sequence tensors (padded to max length).
# ---------------------------------------------------------------------------

def build_seq_tensors(df_feat: pd.DataFrame, pvt_lookup: dict,
                       mu_dyn, sd_dyn, mu_stat, sd_stat) -> dict:
    sims = list(df_feat.groupby("sim_id", sort=True))
    B = len(sims)
    T_max = max(len(g) for _, g in sims)
    n_dyn = len(DYNAMIC_COLS)
    n_static = len(STATIC_COLS)
    pvt_dim = next(iter(pvt_lookup.values())).shape[0]

    X_dyn = np.zeros((B, T_max, n_dyn), dtype=np.float32)
    X_static = np.zeros((B, n_static), dtype=np.float32)
    pvt = np.zeros((B, pvt_dim), dtype=np.float32)
    delta_target = np.zeros((B, T_max), dtype=np.float32)
    p_init_arr = np.zeros((B,), dtype=np.float32)
    mask = np.zeros((B, T_max), dtype=bool)
    sim_ids = np.zeros((B,), dtype=np.int64)
    per_sim_meta = []
    for i, (sim_id, g) in enumerate(sims):
        T = len(g)
        sim_ids[i] = int(sim_id)
        p_init = float(g["P_initial_norm"].iloc[0]) * PRESSURE_SCALE
        p_init_arr[i] = p_init
        dyn = g[DYNAMIC_COLS].values.astype(np.float32)
        dyn = (dyn - mu_dyn) / sd_dyn
        X_dyn[i, :T] = dyn
        stat = g[STATIC_COLS].iloc[0].values.astype(np.float32)
        stat = (stat - mu_stat) / sd_stat
        X_static[i] = stat
        res_id = g["reservoir_id"].iloc[0]
        pvt[i] = pvt_lookup[(res_id, int(sim_id))]
        p_target = g[TARGET].values.astype(np.float32)
        delta_target[i, :T] = (p_target - p_init) / PRESSURE_SCALE
        mask[i, :T] = True
        per_sim_meta.append(g[["sim_id", "reservoir_id", "tiempo_dias", TARGET]]
                             .reset_index(drop=True))
    return {
        "X_dyn": X_dyn, "X_static": X_static, "pvt": pvt,
        "delta_target": delta_target, "p_init": p_init_arr,
        "mask": mask, "sim_ids": sim_ids, "per_sim_meta": per_sim_meta,
    }


def fit_split_stats(df: pd.DataFrame):
    dyn = df[DYNAMIC_COLS].values.astype(np.float32)
    mu_dyn = dyn.mean(axis=0); sd_dyn = dyn.std(axis=0)
    sd_dyn[sd_dyn < 1e-8] = 1.0
    stat_df = df.groupby("sim_id", sort=True).first().reset_index()
    stat = stat_df[STATIC_COLS].values.astype(np.float32)
    mu_stat = stat.mean(axis=0); sd_stat = stat.std(axis=0)
    sd_stat[sd_stat < 1e-8] = 1.0
    return mu_dyn, sd_dyn, mu_stat, sd_stat


# ---------------------------------------------------------------------------
# Model — LSTM with PVT encoder + static encoder.
# ---------------------------------------------------------------------------

class LSTMPVT(nn.Module):
    def __init__(self, n_dyn: int, n_static: int, pvt_dim: int,
                 ctx_dim: int = 16, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.pvt_encoder = nn.Sequential(
            nn.Linear(pvt_dim, 32), nn.GELU(), nn.Linear(32, ctx_dim))
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 16), nn.GELU(), nn.Linear(16, ctx_dim))
        self.lstm = nn.LSTM(
            input_size=n_dyn + 2 * ctx_dim, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, X_dyn, X_static, pvt):
        B, T, _ = X_dyn.shape
        c_pvt = self.pvt_encoder(pvt)
        c_stat = self.static_encoder(X_static)
        c = torch.cat([c_pvt, c_stat], dim=-1)
        c_t = c.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([X_dyn, c_t], dim=-1)
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


def masked_mse(pred, target, mask_f):
    err = (pred - target) ** 2
    return (err * mask_f).sum() / mask_f.sum().clamp(min=1)


def evaluate_psi(y_true, y_pred):
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    # Same dataset resolution as train_pvt_input_multipvt.py.
    # Prefer n=10 (the MLP+PVT sweet spot) for direct comparison.
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

    print(f"  train combined: {train_raw.shape}, sims=1..{shift+n_multi}, "
          f"Pr {train_raw[TARGET].min():.0f}-{train_raw[TARGET].max():.0f}")
    print(f"  volve eval:     {volve_raw.shape}, "
          f"Pr {volve_raw[TARGET].min():.0f}-{volve_raw[TARGET].max():.0f}")

    print("\nComputing per-sim PVT vectors…")
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

    print("Building features…")
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
    print(f"  T_max — train={tr_t['X_dyn'].shape[1]}, "
          f"val={va_t['X_dyn'].shape[1]}, test={te_t['X_dyn'].shape[1]}")

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
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)

    n_epochs = 600; patience = 80
    best_val = float("inf"); best_state = None; bad = 0

    print("\n[Training LSTM]")
    for epoch in range(1, n_epochs + 1):
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
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:>3}  train={loss.item():.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}")
        if bad >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    @torch.no_grad()
    def predict_psi(t_dict, t_np):
        delta = model(t_dict["X_dyn"], t_dict["X_static"], t_dict["pvt"]).cpu().numpy()
        y_true, y_pred = [], []
        for i in range(len(t_np["sim_ids"])):
            T = int(t_np["mask"][i].sum())
            y_true.append(t_np["per_sim_meta"][i][TARGET].values[:T])
            y_pred.append(t_np["p_init"][i] + delta[i, :T] * PRESSURE_SCALE)
        return np.concatenate(y_true), np.concatenate(y_pred)

    print("\n[LSTM evaluation]")
    for tag, t_dict, t_np in [
        (f"Norne val sims {VAL_SIMS}", va, va_t),
        ("Volve (10 sims)", te, te_t),
    ]:
        y_true, y_pred = predict_psi(t_dict, t_np)
        m = evaluate_psi(y_true, y_pred)
        print(f"  {tag:<26} R²={m['R2']:+.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    # Save Volve predictions for plotting.
    y_true_volve, y_pred_volve = predict_psi(te, te_t)
    preds_dir = ROOT / "predictions"
    preds_dir.mkdir(exist_ok=True)
    rows = []
    cursor = 0
    for i in range(len(te_t["sim_ids"])):
        T = int(te_t["mask"][i].sum())
        meta = te_t["per_sim_meta"][i].iloc[:T].copy()
        meta["P_pred_lstm_psi"] = y_pred_volve[cursor:cursor + T]
        meta["P_pred_naive_psi"] = te_t["p_init"][i]
        rows.append(meta)
        cursor += T
    df_out = pd.concat(rows, ignore_index=True)
    df_out = df_out.rename(columns={TARGET: "P_true_psi"})
    df_out.to_csv(preds_dir / "preds_lstm_multipvt_volve.csv", index=False)
    print(f"\nSaved predictions → {preds_dir / 'preds_lstm_multipvt_volve.csv'}")


if __name__ == "__main__":
    sys.exit(main())
