"""PVT-table-as-input MLP using the multi-lever PVT pilot dataset.

This is the Phase 4/5 implementation of `05-30_01_tpp_multilever_pvt_perturbation.md`.

Training set = `dataset_norne.csv` (30 baseline sims, PVT = default) +
`dataset_norne_multipvt_n10.csv` (10 sims with LHS-sampled values for
pb_shift_bar / rs_pb_mult / bo_undersat_slope_mult / oil_density_mult).
Sim_ids of the second set are shifted by +30 to avoid collisions.

Each sim's PVT vector is computed by re-rendering the Norne PVT include
with its 4 lever values, parsing the result with `parse_pvt_include`, and
sampling Bo / Bg / Rs at a fixed pressure grid. This is the single source
of truth between the simulator and the ML model.

Eval = `dataset_volve.csv`. The Volve PVT vector comes from
`pvt_volve.json` (unchanged).
"""

from __future__ import annotations

import json
import sys
import tempfile
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

# PVT vector grid: 17 points from 1500 to 5500 psi (matches pvt_norne.json /
# pvt_volve.json). Output vector layout: [Pb_norm, 17×Bo, 17×Bg×1000, 17×Rs/1000].
P_GRID_PSI = np.linspace(1500.0, 5500.0, 17)

FEATURE_COLS = [
    "Porosidad", "log10_Permeabilidad_mD", "P_initial_norm",
    "Np_over_PV", "Wp_over_PV", "Winj_over_PV", "GOR_cum",
    "qo_over_PV", "qwinj_over_PV", "WOR_inst", "water_cut_cum", "VRR_simple",
]


# ---------------------------------------------------------------------------
# Feature engineering (same as previous PVT-as-input experiments)
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
# PVT vector — re-render the deck PVT include and sample at fixed grid.
# ---------------------------------------------------------------------------

def _load_base_norne_pvt() -> dict:
    """Load the operating-regime Norne PVT (computed by extract_pvt_tables.py
    from .UNRST per-cell aggregates). Pb here is the operating bubble point
    at initial reservoir conditions (~3625 psi), not the max of the PVTO
    table."""
    t = json.loads((DATASETS / "pvt_norne.json").read_text())
    return {
        "pb_psi": float(t["pb_psi"]),
        "p_grid": np.asarray(t["p_grid_psi"], dtype=np.float64),
        "bo": np.asarray(t["bo_rb_stb"], dtype=np.float64),
        "bg": np.asarray(t["bg_rb_scf"], dtype=np.float64),
        "rs": np.asarray(t["rs_scf_stb"], dtype=np.float64),
    }


_BASE_NORNE_PVT_CACHE: dict | None = None


def _pvt_vector_from_levers(pb_shift_bar: float, rs_mult: float,
                             bo_slope_mult: float, oil_density_mult: float,
                             bg_mult: float = 1.0
                             ) -> np.ndarray:
    """Apply the 5 PVT levers analytically to the base Norne PVT vector
    (from `pvt_norne.json`, which reflects the .UNRST-aggregated operating
    regime). Returns a length-52 vector matching the format expected by
    the model:

        [Pb_norm, 17×Bo, 17×Bg×1000, 17×Rs/1000]

    Transformations applied (in order):
      1. `pb_shift_bar`: shifts the P axis of the saturated curves so that
         new_Bo(P) = old_Bo(P − dp), new_Pb = base_Pb + dp.
      2. `rs_mult`: scales Rs values uniformly.
      3. `bo_slope_mult`: rescales Bo above the operating Pb:
            Bo_new = Bo_at_Pb + slope_mult · (Bo − Bo_at_Pb)
         The saturated branch is unchanged.
      4. `oil_density_mult`: scales Bo at every grid point by the multiplier.
         (Reflects Bo = ρ_surface_oil / ρ_reservoir_oil; scaling the surface
         density scales Bo proportionally to first order.)
      5. `bg_mult`: scales Bg at every grid point by the multiplier.

    This is an analytical approximation — the actual `.UNRST` per-cell
    aggregate may differ — but it is consistent with the base PVT and
    captures the first-order effect of each lever.
    """
    global _BASE_NORNE_PVT_CACHE
    if _BASE_NORNE_PVT_CACHE is None:
        _BASE_NORNE_PVT_CACHE = _load_base_norne_pvt()
    base = _BASE_NORNE_PVT_CACHE

    dp_psi = pb_shift_bar * BAR_TO_PSI
    pb_new = base["pb_psi"] + dp_psi
    p_query = P_GRID_PSI - dp_psi

    # Step 1 — shift P axis.
    bo = np.interp(p_query, base["p_grid"], base["bo"])
    bg = np.interp(p_query, base["p_grid"], base["bg"])
    rs = np.interp(p_query, base["p_grid"], base["rs"])

    # Step 2 — scale Rs.
    rs = rs * rs_mult

    # Step 3 — bo_slope above the new Pb.
    bo_at_pb = float(np.interp(base["pb_psi"], base["p_grid"], base["bo"]))
    undersat = P_GRID_PSI > pb_new
    bo[undersat] = bo_at_pb + bo_slope_mult * (bo[undersat] - bo_at_pb)

    # Step 4 — oil_density scaling on Bo.
    bo = bo * oil_density_mult

    # Step 5 — bg scaling.
    bg = bg * bg_mult

    return np.concatenate([
        [pb_new / PRESSURE_SCALE],
        bo, bg * 1000.0, rs / 1000.0,
    ]).astype(np.float32)


def volve_pvt_vector() -> np.ndarray:
    """Volve PVT from the canonical JSON (not perturbed)."""
    t = json.loads((DATASETS / "pvt_volve.json").read_text())
    pb = float(t["pb_psi"]) / PRESSURE_SCALE
    bo = np.asarray(t["bo_rb_stb"])
    bg = np.asarray(t["bg_rb_scf"]) * 1000.0
    rs = np.asarray(t["rs_scf_stb"]) / 1000.0
    return np.concatenate([[pb], bo, bg, rs]).astype(np.float32)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------

class PressurePredictor(nn.Module):
    def __init__(self, n_features: int, pvt_dim: int, context_dim: int = 8):
        super().__init__()
        self.pvt_encoder = nn.Sequential(
            nn.Linear(pvt_dim, 32), nn.GELU(), nn.Linear(32, context_dim))
        self.delta_head = nn.Sequential(
            nn.Linear(n_features + context_dim, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, pvt: torch.Tensor) -> torch.Tensor:
        c = self.pvt_encoder(pvt)
        return self.delta_head(torch.cat([x, c], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------------------

def make_arrays(df_feat: pd.DataFrame, pvt_lookup: dict[tuple, np.ndarray]):
    X = df_feat[FEATURE_COLS].values.astype(np.float32)
    p_target = df_feat[TARGET].values.astype(np.float32)
    p_init = (df_feat["P_initial_norm"].values * PRESSURE_SCALE).astype(np.float32)
    y_delta = ((p_target - p_init) / PRESSURE_SCALE).astype(np.float32)
    keys = list(zip(df_feat["reservoir_id"].values, df_feat["sim_id"].values))
    pvt = np.stack([pvt_lookup[k] for k in keys]).astype(np.float32)
    return X, y_delta, p_init, pvt


def evaluate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

    # Resolve which perturbation dataset to use. Falls back to the
    # multipvt-only set if the combined one isn't available yet.
    candidates = [
        ("dataset_norne_sched_multipvt_n10.csv",
         "runs_log_norne_sched_multipvt_n10.csv"),
        ("dataset_norne_multipvt_n10.csv",
         "runs_log_norne_multipvt_n10.csv"),
    ]
    ds_path = log_path = None
    for d, l in candidates:
        if (DATASETS / d).exists() and (DATASETS / l).exists():
            ds_path, log_path = DATASETS / d, DATASETS / l
            break
    if ds_path is None:
        raise FileNotFoundError(f"no perturbation dataset found ({candidates})")
    print(f"Loading datasets ({ds_path.name})…")
    norne_base  = pd.read_csv(DATASETS / "dataset_norne.csv")
    norne_multi = pd.read_csv(ds_path)
    runs_log    = pd.read_csv(log_path)
    volve_raw   = pd.read_csv(DATASETS / "dataset_volve.csv")

    # Shift multi-pvt sim_ids so they do not collide with baseline.
    shift = int(norne_base["sim_id"].max())
    norne_multi = norne_multi.copy()
    norne_multi["sim_id"] = norne_multi["sim_id"] + shift
    train_raw = pd.concat([norne_base, norne_multi], ignore_index=True)

    n_multi = len(set(norne_multi["sim_id"]))
    assert train_raw["sim_id"].nunique() == norne_base["sim_id"].nunique() + n_multi

    print(f"  norne baseline: {norne_base.shape}, sims 1-{shift}, Pr "
          f"{norne_base[TARGET].min():.0f}-{norne_base[TARGET].max():.0f}")
    print(f"  norne multipvt:  {norne_multi.shape}, sims {shift+1}-{shift+n_multi}, Pr "
          f"{norne_multi[TARGET].min():.0f}-{norne_multi[TARGET].max():.0f}")
    print(f"  train combined: {train_raw.shape}, sims={train_raw['sim_id'].nunique()}, "
          f"Pr {train_raw[TARGET].min():.0f}-{train_raw[TARGET].max():.0f}")
    print(f"  volve: {volve_raw.shape}, Pr {volve_raw[TARGET].min():.0f}-"
          f"{volve_raw[TARGET].max():.0f}")

    # ---- Build per-sim PVT vectors ----
    print("\nComputing per-sim PVT vectors…")
    pvt_lookup: dict[tuple, np.ndarray] = {}

    # Baseline Norne sims (1..30) — all levers at default → same vector for all.
    base_vec = _pvt_vector_from_levers(0.0, 1.0, 1.0, 1.0, 1.0)
    for sid in norne_base["sim_id"].unique():
        pvt_lookup[("norne", int(sid))] = base_vec

    # Multi-pvt sims (31..40) — read lever values from runs_log.
    ok_log = runs_log[runs_log["ok"]].copy()
    for _, row in ok_log.iterrows():
        original_sim_id = int(row["sim_id"])
        shifted_sim_id = original_sim_id + shift
        vec = _pvt_vector_from_levers(
            pb_shift_bar=float(row["pb_shift_bar"]),
            rs_mult=float(row["rs_pb_mult"]),
            bo_slope_mult=float(row["bo_undersat_slope_mult"]),
            oil_density_mult=float(row["oil_density_mult"]),
            bg_mult=float(row.get("bg_mult", 1.0)),
        )
        pvt_lookup[("norne", shifted_sim_id)] = vec

    # Volve — canonical PVT.
    volve_vec = volve_pvt_vector()
    for sid in volve_raw["sim_id"].unique():
        pvt_lookup[("volve", int(sid))] = volve_vec

    # Sanity: confirm PVT vectors actually differ across sims.
    print("\nPVT-vector spread across training sims (key dims):")
    print(f"  baseline    Pb_norm={base_vec[0]:.3f}  Bo[8]={base_vec[9]:.4f}  "
          f"Bg×1000[8]={base_vec[26]:.4f}  Rs/1000[8]={base_vec[43]:.4f}")
    for _, row in ok_log.head(5).iterrows():
        sid = int(row["sim_id"]) + shift
        v = pvt_lookup[("norne", sid)]
        print(f"  sim {sid:>2}      Pb_norm={v[0]:.3f}  Bo[8]={v[9]:.4f}  "
              f"Bg×1000[8]={v[26]:.4f}  Rs/1000[8]={v[43]:.4f}  "
              f"(pb={row['pb_shift_bar']:+.1f}, rs×{row['rs_pb_mult']:.2f}, "
              f"bo×{row['bo_undersat_slope_mult']:.2f}, od×{row['oil_density_mult']:.3f})")
    print(f"  volve       Pb_norm={volve_vec[0]:.3f}  Bo[8]={volve_vec[9]:.4f}  "
          f"Bg×1000[8]={volve_vec[26]:.4f}  Rs/1000[8]={volve_vec[43]:.4f}")

    # ---- Feature engineering ----
    print("\nBuilding features…")
    train_feat = build_features(train_raw)
    volve_feat = build_features(volve_raw)

    # ---- Split: hold out one baseline (30) + one multipvt (last) ----
    last_multi = shift + n_multi
    VAL_SIMS = [shift, last_multi]
    val_mask = train_feat["sim_id"].isin(VAL_SIMS).values
    tr_mask = ~val_mask
    print(f"  train rows: {tr_mask.sum()}, val rows: {val_mask.sum()} "
          f"(VAL_SIMS={VAL_SIMS}), eval Volve rows: {len(volve_feat)}")

    X_tr, yd_tr, pi_tr, pvt_tr = make_arrays(train_feat[tr_mask], pvt_lookup)
    X_va, yd_va, pi_va, pvt_va = make_arrays(train_feat[val_mask], pvt_lookup)
    X_te, yd_te, pi_te, pvt_te = make_arrays(volve_feat, pvt_lookup)
    p_te = pi_te + yd_te * PRESSURE_SCALE
    p_va_true = pi_va + yd_va * PRESSURE_SCALE

    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd[sd < 1e-8] = 1.0
    def norm(x): return ((x - mu) / sd).astype(np.float32)
    X_tr_n, X_va_n, X_te_n = norm(X_tr), norm(X_va), norm(X_te)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PressurePredictor(n_features=len(FEATURE_COLS), pvt_dim=pvt_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def to_t(*arrs):
        return [torch.from_numpy(a).to(device) for a in arrs]
    X_tr_t, yd_tr_t, pvt_tr_t = to_t(X_tr_n, yd_tr, pvt_tr)
    X_va_t, yd_va_t, pvt_va_t = to_t(X_va_n, yd_va, pvt_va)

    print("\n[Training PVT-encoded MLP with multi-lever PVT per sim]")
    n_epochs, batch_size, patience = 600, 256, 60
    best_val, best_state, bad = float("inf"), None, 0
    n_tr = X_tr_t.shape[0]

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        total = 0.0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            yhat = model(X_tr_t[idx], pvt_tr_t[idx])
            loss = loss_fn(yhat, yd_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * idx.shape[0]
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t, pvt_va_t), yd_va_t).item()
        if val_loss < best_val - 1e-7:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"  epoch {epoch:>3}  train_mse={total/n_tr:.5f}  "
                  f"val_mse={val_loss:.5f}  best={best_val:.5f}")
        if bad >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  trained — {n_params:,} parameters, best val MSE = {best_val:.6f}")

    @torch.no_grad()
    def predict_psi(X_n, pvt, p_init):
        xt, pt = to_t(X_n, pvt)
        delta = model(xt, pt).cpu().numpy()
        return p_init + delta * PRESSURE_SCALE

    print("\n[Evaluation]")
    m = evaluate_metric(p_va_true, predict_psi(X_va_n, pvt_va, pi_va))
    print(f"  Norne val sims {VAL_SIMS}  R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    pred_volve = predict_psi(X_te_n, pvt_te, pi_te)
    m = evaluate_metric(p_te, pred_volve)
    print(f"  Volve (10 sims)         R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    pred_naive = pi_te
    m_naive = evaluate_metric(p_te, pred_naive)
    print(f"  Naive on Volve          R²={m_naive['R2']:.3f}  RMSE={m_naive['RMSE']:.1f}  MAE={m_naive['MAE']:.1f}")

    preds_dir = ROOT / "predictions"
    preds_dir.mkdir(exist_ok=True)
    out = volve_raw[["sim_id", "reservoir_id", "tiempo_dias"]].copy()
    out["P_true_psi"] = p_te
    out["P_pred_psi"] = pred_volve
    out["P_pred_naive_psi"] = pred_naive
    out.to_csv(preds_dir / "preds_pvt_input_multipvt_volve.csv", index=False)
    print(f"\nSaved predictions → {preds_dir / 'preds_pvt_input_multipvt_volve.csv'}")


if __name__ == "__main__":
    sys.exit(main())
