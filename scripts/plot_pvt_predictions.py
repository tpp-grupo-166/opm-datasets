"""Plot predicted vs true reservoir pressure for the three eval splits.

Reads /predictions/preds_*.csv produced by train_pvt_model.py and writes
PNGs to /plots_pvt/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PREDS_DIR = ROOT / "predictions"
PLOTS_DIR = ROOT / "plots_pvt"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_split(name: str, title: str) -> None:
    df = pd.read_csv(PREDS_DIR / f"preds_{name}.csv")
    df = df.sort_values(["sim_id", "tiempo_dias"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series.
    ax = axes[0]
    for sim, g in df.groupby("sim_id"):
        ax.plot(g["tiempo_dias"], g["P_true_psi"], "k-", lw=2, label="True" if sim == df["sim_id"].iloc[0] else None)
        ax.plot(g["tiempo_dias"], g["P_pred_psi"], "C0-", lw=1.5, label="MLP+PVT" if sim == df["sim_id"].iloc[0] else None)
        ax.plot(g["tiempo_dias"], g["P_pred_ridge_psi"], "C3--", lw=1, label="Ridge (no PVT)" if sim == df["sim_id"].iloc[0] else None)
    ax.set_xlabel("tiempo [días]")
    ax.set_ylabel("Presión [psi]")
    ax.set_title(f"{title} — trayectoria")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Scatter.
    ax = axes[1]
    lo = min(df["P_true_psi"].min(), df["P_pred_psi"].min(), df["P_pred_ridge_psi"].min())
    hi = max(df["P_true_psi"].max(), df["P_pred_psi"].max(), df["P_pred_ridge_psi"].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
    ax.scatter(df["P_true_psi"], df["P_pred_psi"], s=8, alpha=0.6, label="MLP+PVT", color="C0")
    ax.scatter(df["P_true_psi"], df["P_pred_ridge_psi"], s=8, alpha=0.4, label="Ridge", color="C3")
    ax.set_xlabel("P real [psi]")
    ax.set_ylabel("P predicha [psi]")
    ax.set_title(f"{title} — parity")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_split("volve_val",  "Volve val (sim 9)")
    plot_split("volve_test", "Volve test (sim 10)")
    plot_split("norne_sim1", "Norne sim 1 (cross-reservoir)")


if __name__ == "__main__":
    main()
