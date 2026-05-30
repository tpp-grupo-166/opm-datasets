"""Plot Volve predictions from the combined-pilot MLP+PVT-input model.

Generates:
  plots_pvt/multipvt_volve_trajectories.png — 2x2 trajectory panels for
      4 Volve sims (true / MLP+PVT / Naive)
  plots_pvt/multipvt_volve_parity.png        — parity scatter for all
      10 Volve sims, plus per-sim MAE bar chart
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "predictions"
PLOTS = ROOT / "plots_pvt"
PLOTS.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(PREDS / "preds_pvt_input_multipvt_volve.csv")
    df = df.sort_values(["sim_id", "tiempo_dias"]).reset_index(drop=True)

    # --- Trajectories (4 sims) ---
    sims = [1, 4, 7, 10]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for ax, sim_id in zip(axes.flat, sims):
        g = df[df["sim_id"] == sim_id]
        ax.plot(g["tiempo_dias"], g["P_true_psi"], "k-", lw=2, label="True")
        ax.plot(g["tiempo_dias"], g["P_pred_psi"], "C0-", lw=1.4,
                label="MLP+PVT (combined)")
        ax.plot(g["tiempo_dias"], g["P_pred_naive_psi"], "C3--", lw=1.0,
                label="Naive (P_init broadcast)")
        mae = mean_absolute_error(g["P_true_psi"], g["P_pred_psi"])
        r2 = r2_score(g["P_true_psi"], g["P_pred_psi"])
        ax.set_xlabel("tiempo [días]")
        ax.set_ylabel("Pr [psi]")
        ax.set_title(f"Volve sim {sim_id}   MAE={mae:.0f} psi   R²={r2:+.2f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Volve cross-reservoir prediction — combined (schedule + 5-PVT) pilot, 10 train sims + 30 baseline",
                 fontsize=12)
    fig.tight_layout()
    out1 = PLOTS / "multipvt_volve_trajectories.png"
    fig.savefig(out1, dpi=110)
    plt.close(fig)
    print(f"wrote {out1}")

    # --- Parity + per-sim MAE ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    lo = min(df["P_true_psi"].min(), df["P_pred_psi"].min(),
             df["P_pred_naive_psi"].min())
    hi = max(df["P_true_psi"].max(), df["P_pred_psi"].max(),
             df["P_pred_naive_psi"].max())
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4,
            label="y = x")
    ax.scatter(df["P_true_psi"], df["P_pred_naive_psi"], s=5, alpha=0.30,
               color="C3", label="Naive")
    ax.scatter(df["P_true_psi"], df["P_pred_psi"], s=5, alpha=0.55,
               color="C0", label="MLP+PVT")
    ax.set_xlabel("Pr real [psi]")
    ax.set_ylabel("Pr predicha [psi]")
    overall_mae = mean_absolute_error(df["P_true_psi"], df["P_pred_psi"])
    overall_r2 = r2_score(df["P_true_psi"], df["P_pred_psi"])
    ax.set_title(f"Parity — all 10 Volve sims\n"
                 f"MLP+PVT: MAE={overall_mae:.1f}, R²={overall_r2:+.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    # Per-sim MAE bar chart.
    ax = axes[1]
    per_sim = (
        df.groupby("sim_id")
          .apply(lambda g: pd.Series({
              "mlp": mean_absolute_error(g["P_true_psi"], g["P_pred_psi"]),
              "naive": mean_absolute_error(g["P_true_psi"], g["P_pred_naive_psi"]),
          }))
          .reset_index()
    )
    width = 0.4
    x = np.arange(len(per_sim))
    ax.bar(x - width / 2, per_sim["mlp"], width, color="C0", label="MLP+PVT")
    ax.bar(x + width / 2, per_sim["naive"], width, color="C3", label="Naive")
    ax.set_xticks(x)
    ax.set_xticklabels(per_sim["sim_id"].astype(int))
    ax.set_xlabel("Volve sim_id")
    ax.set_ylabel("MAE [psi]")
    ax.set_title("Per-sim MAE: MLP+PVT vs Naive")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out2 = PLOTS / "multipvt_volve_parity.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
