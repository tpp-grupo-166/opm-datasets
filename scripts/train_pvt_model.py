"""Train an MLP that takes production features + a PVT table as input.

The PVT table is fed alongside the per-timestep production data:

  P_hat(t) = f( production_features(t), pvt_table[reservoir]; theta )

Same model weights work for any reservoir at inference time — only the
table swap. Training uses Volve only, split by sim_id; evaluation uses
one Norne simulation (held out entirely).

A Ridge baseline (without any PVT info) is trained on the same features
for a fair comparison.

Run:
  python3 scripts/train_pvt_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

SEED = 42
M3_TO_BBL = 6.28981
PRESSURE_SCALE = 5000.0  # divide target by this so the NN sees ~[0, 1]


# ---------------------------------------------------------------------------
# Feature engineering (same idea as the Ridge baseline notebook, minus the
# PVT columns — those are intentionally excluded to avoid target leakage).
# ---------------------------------------------------------------------------


FEATURE_COLS = [
    "Porosidad",
    "log10_Permeabilidad_mD",
    "P_initial_norm",         # initial pressure of the sim, scaled
    "Np_over_PV",
    "Wp_over_PV",
    "Winj_over_PV",
    "GOR_cum",
    "qo_over_PV",
    "qwinj_over_PV",
    "WOR_inst",
    "water_cut_cum",
    "VRR_simple",
]
TARGET_COL = "Presion_Reservorio_psi"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row features. Does NOT include Bo/Bg/Rs/Pb (those go via the PVT
    table). Mirrors the normalization used in the baseline."""
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    out["sim_id"] = df["sim_id"]
    out["reservoir_id"] = df["reservoir_id"]
    out["tiempo_dias"] = df["tiempo_dias"]

    pv_bbl = df["Area"] * df["Espesor_Neto_m"] * df["Porosidad"] * M3_TO_BBL

    out["Porosidad"] = df["Porosidad"]
    out["log10_Permeabilidad_mD"] = np.log10(df["Permeabilidad_mD"].clip(lower=1e-3))

    # Anchor: initial pressure of each sim (P at first timestep). Constant per
    # sim. Scaled so it sits in ~[0, 1.1]. Acts as a soft prior so the model
    # can learn to predict a *drop* from initial rather than absolute psi.
    p_first = df.groupby(["reservoir_id", "sim_id"])[TARGET_COL].transform("first")
    out["P_initial_norm"] = p_first / PRESSURE_SCALE

    out["Np_over_PV"] = df["Prod_Acumulada_Petroleo"] / pv_bbl
    out["Wp_over_PV"] = df["Prod_Acumulada_Agua"] / pv_bbl
    out["Winj_over_PV"] = df["Iny_Acumulada_Agua"] / pv_bbl

    with np.errstate(divide="ignore", invalid="ignore"):
        out["GOR_cum"] = np.where(
            df["Prod_Acumulada_Petroleo"] > 0,
            df["Prod_Acumulada_Gas"] / df["Prod_Acumulada_Petroleo"],
            0.0,
        )

    out["qo_over_PV"] = df["Caudal_Prod_Petroleo_bbl"] / pv_bbl
    out["qwinj_over_PV"] = df["Caudal_Iny_Agua_bbl"] / pv_bbl

    g = df.groupby(["reservoir_id", "sim_id"])
    dt = g["tiempo_dias"].diff().replace(0, np.nan)
    wp_rate = g["Prod_Acumulada_Agua"].diff() / dt
    with np.errstate(divide="ignore", invalid="ignore"):
        out["WOR_inst"] = np.where(
            df["Caudal_Prod_Petroleo_bbl"] > 0,
            wp_rate / df["Caudal_Prod_Petroleo_bbl"],
            0.0,
        )

    total_liq = df["Prod_Acumulada_Petroleo"] + df["Prod_Acumulada_Agua"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["water_cut_cum"] = np.where(
            total_liq > 0, df["Prod_Acumulada_Agua"] / total_liq, 0.0
        )
        out["VRR_simple"] = np.where(
            total_liq > 0, df["Iny_Acumulada_Agua"] / total_liq, 0.0
        )

    out[TARGET_COL] = df[TARGET_COL]
    out = out.fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# PVT table loading.
# ---------------------------------------------------------------------------


def load_pvt(reservoir_id: str) -> np.ndarray:
    """Returns a 1-D numpy array of length 52: Pb + 17*(Bo, Bg, Rs).
    Values rescaled so each property sits on a similar scale.
    """
    table = json.loads((DATASETS / f"pvt_{reservoir_id}.json").read_text())
    pb = table["pb_psi"] / PRESSURE_SCALE
    bo = np.array(table["bo_rb_stb"])           # ~1.0
    bg = np.array(table["bg_rb_scf"]) * 1000.0  # ~1.0 after rescale
    rs = np.array(table["rs_scf_stb"]) / 1000.0 # ~1.0 after rescale
    return np.concatenate([[pb], bo, bg, rs]).astype(np.float32)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class PvtEncoder(nn.Module):
    """Small MLP that compresses (Pb + bo/bg/rs grids) into a context vector.
    Kept narrow on purpose — only 10 Volve sims to train on, so capacity is
    the main lever against overfit."""

    def __init__(self, pvt_dim: int, context_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pvt_dim, 32),
            nn.GELU(),
            nn.Linear(32, context_dim),
        )

    def forward(self, pvt: torch.Tensor) -> torch.Tensor:
        return self.net(pvt)


class PressurePredictor(nn.Module):
    """Predicts pressure as an offset from the initial pressure of each
    simulation. The target during training is delta = (P - P_initial) /
    PRESSURE_SCALE, which has a similar distribution across reservoirs
    and is far more transferable than absolute psi."""

    def __init__(self, n_features: int, pvt_dim: int, context_dim: int = 8):
        super().__init__()
        self.pvt_encoder = PvtEncoder(pvt_dim, context_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(n_features + context_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_norm: torch.Tensor, pvt: torch.Tensor) -> torch.Tensor:
        c = self.pvt_encoder(pvt)
        z = torch.cat([x_norm, c], dim=-1)
        return self.delta_head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities.
# ---------------------------------------------------------------------------


def make_arrays(
    df_feat: pd.DataFrame, pvt_by_reservoir: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns X (features), y_delta (target as delta from P_initial,
    scaled), p_init_psi (per-row anchor in psi), pvt."""
    X = df_feat[FEATURE_COLS].values.astype(np.float32)
    p_target = df_feat[TARGET_COL].values.astype(np.float32)
    p_init = (df_feat["P_initial_norm"].values * PRESSURE_SCALE).astype(np.float32)
    y_delta = ((p_target - p_init) / PRESSURE_SCALE).astype(np.float32)
    pvt = np.stack(
        [pvt_by_reservoir[r] for r in df_feat["reservoir_id"].values]
    ).astype(np.float32)
    return X, y_delta, p_init, pvt


def fit_feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def evaluate(y_true_psi: np.ndarray, y_pred_psi: np.ndarray) -> dict:
    return {
        "R2": float(r2_score(y_true_psi, y_pred_psi)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_psi, y_pred_psi))),
        "MAE": float(mean_absolute_error(y_true_psi, y_pred_psi)),
    }


# ---------------------------------------------------------------------------
# Pipeline.
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading datasets…")
    volve = pd.read_csv(DATASETS / "dataset_volve.csv")
    norne = pd.read_csv(DATASETS / "dataset_norne.csv")

    print("Building features…")
    volve_feat = build_features(volve)
    norne_feat = build_features(norne)

    # ----- Splits -----
    train_sims = list(range(1, 9))   # Volve sims 1-8
    val_sims = [9]                   # Volve sim 9
    test_volve_sims = [10]           # Volve sim 10
    norne_eval_sim = 1               # Single Norne sim used for eval

    train_df = volve_feat[volve_feat["sim_id"].isin(train_sims)].reset_index(drop=True)
    val_df = volve_feat[volve_feat["sim_id"].isin(val_sims)].reset_index(drop=True)
    test_volve_df = volve_feat[volve_feat["sim_id"].isin(test_volve_sims)].reset_index(drop=True)
    test_norne_df = norne_feat[norne_feat["sim_id"] == norne_eval_sim].reset_index(drop=True)

    print(
        f"  Train Volve sims {train_sims}: {len(train_df)} rows\n"
        f"  Val   Volve sim  {val_sims}:    {len(val_df)} rows\n"
        f"  Test  Volve sim  {test_volve_sims}:    {len(test_volve_df)} rows\n"
        f"  Test  Norne sim  {norne_eval_sim}:     {len(test_norne_df)} rows"
    )

    # ----- PVT tables (the "input" beyond per-row features) -----
    pvt_by_reservoir = {"volve": load_pvt("volve"), "norne": load_pvt("norne")}
    pvt_dim = pvt_by_reservoir["volve"].shape[0]
    print(f"  PVT vector dim per reservoir: {pvt_dim}")

    X_tr, yd_tr, pi_tr, pvt_tr = make_arrays(train_df, pvt_by_reservoir)
    X_va, yd_va, pi_va, pvt_va = make_arrays(val_df, pvt_by_reservoir)
    X_te_v, yd_te_v, pi_te_v, pvt_te_v = make_arrays(test_volve_df, pvt_by_reservoir)
    X_te_n, yd_te_n, pi_te_n, pvt_te_n = make_arrays(test_norne_df, pvt_by_reservoir)

    # True pressure per split in psi (target for metrics).
    p_tr = pi_tr + yd_tr * PRESSURE_SCALE
    p_va = pi_va + yd_va * PRESSURE_SCALE
    p_te_v = pi_te_v + yd_te_v * PRESSURE_SCALE
    p_te_n = pi_te_n + yd_te_n * PRESSURE_SCALE

    mu, sd = fit_feature_stats(X_tr)
    def norm(x):
        return ((x - mu) / sd).astype(np.float32)
    X_tr_n = norm(X_tr); X_va_n = norm(X_va)
    X_te_v_n = norm(X_te_v); X_te_n_n = norm(X_te_n)

    # ----- Ridge baseline. Trained directly on absolute pressure with
    # RAW (non-standardized) features to match the baseline notebook. -----
    print("\n[Ridge baseline — same features, no PVT]")
    ridge = Ridge(alpha=0.0025)
    ridge.fit(X_tr, p_tr)
    for name, X, p in [
        ("Volve val", X_va, p_va),
        ("Volve test", X_te_v, p_te_v),
        ("Norne sim 1", X_te_n, p_te_n),
    ]:
        m = evaluate(p, ridge.predict(X))
        print(f"  {name:<12}  R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    # ----- MLP + PVT-encoder -----
    print("\n[MLP + PVT-encoder — PVT table is an input]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PressurePredictor(n_features=len(FEATURE_COLS), pvt_dim=pvt_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def to_torch(*arrs):
        return [torch.from_numpy(a).to(device) for a in arrs]

    X_tr_t, yd_tr_t, pvt_tr_t = to_torch(X_tr_n, yd_tr, pvt_tr)
    X_va_t, yd_va_t, pvt_va_t = to_torch(X_va_n, yd_va, pvt_va)

    batch_size = 256
    n_epochs = 400
    best_val = float("inf")
    best_state = None
    patience = 40
    bad_epochs = 0

    n = X_tr_t.shape[0]
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_tr_t[idx]; yb = yd_tr_t[idx]; pb = pvt_tr_t[idx]
            yhat = model(xb, pb)
            loss = loss_fn(yhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            yhat_va = model(X_va_t, pvt_va_t)
            val_loss = loss_fn(yhat_va, yd_va_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:>3}  train={train_loss:.5f}  val={val_loss:.5f}  best={best_val:.5f}"
            )

        if bad_epochs >= patience:
            print(f"  early stop at epoch {epoch} (no val improvement in {patience} epochs)")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    def predict_psi(X_n, pvt, p_init):
        """Returns absolute pressure in psi."""
        with torch.no_grad():
            xt, pt = to_torch(X_n, pvt)
            delta = model(xt, pt).cpu().numpy()
        return p_init + delta * PRESSURE_SCALE

    print("\n[MLP+PVT evaluation]")
    for name, X, p, pvt, p_init in [
        ("Volve val", X_va_n, p_va, pvt_va, pi_va),
        ("Volve test", X_te_v_n, p_te_v, pvt_te_v, pi_te_v),
        ("Norne sim 1", X_te_n_n, p_te_n, pvt_te_n, pi_te_n),
    ]:
        m = evaluate(p, predict_psi(X, pvt, p_init))
        print(f"  {name:<12}  R²={m['R2']:.3f}  RMSE={m['RMSE']:.1f}  MAE={m['MAE']:.1f}")

    # Save model + predictions
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "feature_mu": mu, "feature_sd": sd,
            "pressure_scale": PRESSURE_SCALE,
        },
        ckpt_dir / "mlp_pvt.pt",
    )
    print(f"\nSaved model to {ckpt_dir / 'mlp_pvt.pt'}")

    # Predictions per-sim CSV for plotting later.
    preds_dir = ROOT / "predictions"
    preds_dir.mkdir(exist_ok=True)
    for name, df_src, X, p, pvt, p_init, X_raw in [
        ("volve_val",  val_df,         X_va_n,   p_va,   pvt_va,   pi_va,   X_va),
        ("volve_test", test_volve_df,  X_te_v_n, p_te_v, pvt_te_v, pi_te_v, X_te_v),
        ("norne_sim1", test_norne_df,  X_te_n_n, p_te_n, pvt_te_n, pi_te_n, X_te_n),
    ]:
        out_df = df_src[["sim_id", "reservoir_id", "tiempo_dias"]].copy() \
            if "tiempo_dias" in df_src.columns else df_src[["sim_id", "reservoir_id"]].copy()
        out_df["P_true_psi"] = p
        out_df["P_pred_psi"] = predict_psi(X, pvt, p_init)
        out_df["P_pred_ridge_psi"] = ridge.predict(X_raw)
        out_df.to_csv(preds_dir / f"preds_{name}.csv", index=False)
    print(f"Saved per-row predictions to {preds_dir}/")


if __name__ == "__main__":
    sys.exit(main())
