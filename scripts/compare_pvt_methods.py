"""Compare old (FPR-only SPE9-table) vs new (per-cell, deck PVT) Bo/Bg/Rs
for one Norne simulation.

Reads:
  - datasets/dataset_norne.csv          (old method, sim_id == 1)
  - datasets/dataset_norne_one_sim_unrst.csv  (new method)

Writes:
  - plots_norne/pvt_method_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD = PROJECT_ROOT / "datasets" / "dataset_norne.csv"
NEW = PROJECT_ROOT / "datasets" / "dataset_norne_one_sim_unrst.csv"
OUT = PROJECT_ROOT / "plots_norne" / "pvt_method_comparison.png"


def main() -> None:
    old_all = pd.read_csv(OLD)
    new_all = pd.read_csv(NEW)

    old = old_all[old_all["sim_id"] == 1].copy() if "sim_id" in old_all.columns else old_all
    new = new_all.copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for ax, col, title, unit in [
        (axes[0], "Bo_rb_stb", "Bo", "rb/STB"),
        (axes[1], "Bg_rb_scf", "Bg", "rb/scf"),
        (axes[2], "Rs_scf_stb", "Rs", "scf/STB"),
    ]:
        ax.plot(old["tiempo_dias"], old[col], label="anterior (SPE9 PVT vs FPR)",
                linewidth=2, color="#d62728")
        ax.plot(new["tiempo_dias"], new[col], label="nuevo (PVT del deck, ponderado por celda)",
                linewidth=2, color="#1f77b4")
        ax.set_title(f"{title} vs tiempo — Norne, sim_id=1")
        ax.set_ylabel(f"{title} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Tiempo (días)")
    fig.suptitle(
        "Comparación de métodos: PVT viejo (interp. tabla SPE9 en FPR) vs "
        "nuevo (tablas del deck, promedio celda-por-celda)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
