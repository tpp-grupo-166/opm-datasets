"""Side-by-side plot of pressure trajectories: Norne baseline (30 sims)
vs Norne multi-PVT pilot (10 sims, 5-lever LHS).

Both panels share the y-axis so the spread of the PVT-perturbed sims is
directly comparable to the historical-schedule baseline. Writes
plots_pvt/norne_multipvt_training.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
PLOTS = ROOT / "plots_pvt"
PLOTS.mkdir(exist_ok=True)


def main() -> None:
    baseline = pd.read_csv(DATASETS / "dataset_norne.csv")
    multipvt = pd.read_csv(DATASETS / "dataset_norne_multipvt_n10.csv")

    p_min = min(baseline.Presion_Reservorio_psi.min(),
                multipvt.Presion_Reservorio_psi.min())
    p_max = max(baseline.Presion_Reservorio_psi.max(),
                multipvt.Presion_Reservorio_psi.max())
    pad = 0.03 * (p_max - p_min)
    ylim = (p_min - pad, p_max + pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for df, ax, title in [
        (baseline, axes[0], "dataset_norne.csv  (30 sims, baseline schedule, default PVT)"),
        (multipvt, axes[1], "dataset_norne_multipvt_n10.csv  (10 sims, 5-lever PVT LHS, schedule = 1.0)"),
    ]:
        for sim_id, g in df.groupby("sim_id"):
            g = g.sort_values("tiempo_dias")
            ax.plot(g["tiempo_dias"], g["Presion_Reservorio_psi"],
                    lw=1.0, alpha=0.7)
        pr_lo = df["Presion_Reservorio_psi"].min()
        pr_hi = df["Presion_Reservorio_psi"].max()
        ax.set_title(f"{title}\nFPR range: {pr_lo:.0f} – {pr_hi:.0f} psi  "
                     f"(Δ = {pr_hi - pr_lo:.0f} psi)")
        ax.set_xlabel("tiempo [días]")
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Presion_Reservorio_psi [psi]")

    fig.suptitle("Norne training set — baseline vs multi-lever PVT pilot",
                 fontsize=12)
    fig.tight_layout()
    out = PLOTS / "norne_multipvt_training.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
