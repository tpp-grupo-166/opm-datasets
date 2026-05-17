"""Extract a per-reservoir PVT table from the existing datasets.

For each reservoir, we use a single simulation (sim_id == 1) as the canonical
PVT source. The dataset already carries Bo, Bg, Rs evaluated at the field
pressure of each timestep; sorting those rows by pressure gives us a sampled
PVT curve. We then resample onto a fixed pressure grid so every reservoir
has the same input shape for the model.

The output is a JSON file with:

    {
      "reservoir_id": "...",
      "pb_psi": <single bubble-point used in that sim>,
      "p_grid_psi": [...],
      "bo_rb_stb":  [...],
      "bg_rb_scf":  [...],
      "rs_scf_stb": [...]
    }

"PVT as input" means: this table is fed to the model along with the
production features. Same model weights work for any reservoir by swapping
its table at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

# Common pressure grid used for ALL reservoirs (psia). Wide enough to cover
# both Volve (~2500-4800) and Norne (~3000-4200) without extrapolating much.
P_GRID = np.linspace(1500.0, 5500.0, 17)


def _build_table(df: pd.DataFrame, reservoir_id: str) -> dict:
    # Pool ALL sims of this reservoir so the curve covers the full observed
    # pressure range. pb_shift varies across sims, which introduces some
    # smearing near Pb, but the alternative (one sim) leaves the curve
    # defined only in a narrow band and forces the model to extrapolate.
    sub = df[df["reservoir_id"] == reservoir_id].copy()
    if sub.empty:
        raise RuntimeError(f"no rows for reservoir {reservoir_id}")

    # Drop rows where Bo/Bg/Rs/Pr are NaN.
    sub = sub.dropna(subset=["Presion_Reservorio_psi", "Bo_rb_stb",
                              "Bg_rb_scf", "Rs_scf_stb"])

    # Sort by pressure, average duplicates (sometimes the same P appears
    # across consecutive timesteps).
    g = sub.groupby(
        np.round(sub["Presion_Reservorio_psi"].values, 1)
    )[["Bo_rb_stb", "Bg_rb_scf", "Rs_scf_stb"]].mean()
    p_sorted = g.index.values
    bo_sorted = g["Bo_rb_stb"].values
    bg_sorted = g["Bg_rb_scf"].values
    rs_sorted = g["Rs_scf_stb"].values

    # Interpolate onto the common grid. np.interp clamps outside the table
    # (returns the boundary value), which is the right behavior for our PVT.
    bo = np.interp(P_GRID, p_sorted, bo_sorted)
    bg = np.interp(P_GRID, p_sorted, bg_sorted)
    rs = np.interp(P_GRID, p_sorted, rs_sorted)

    pb = float(sub["Presion_Burbuja_psi"].mean())

    return {
        "reservoir_id": reservoir_id,
        "pb_psi": pb,
        "p_grid_psi": P_GRID.tolist(),
        "bo_rb_stb": bo.tolist(),
        "bg_rb_scf": bg.tolist(),
        "rs_scf_stb": rs.tolist(),
        "p_range_observed": [float(p_sorted.min()), float(p_sorted.max())],
    }


def main() -> None:
    sources = {
        "volve": DATASETS / "dataset_volve.csv",
        "norne": DATASETS / "dataset_norne.csv",
    }

    for reservoir, path in sources.items():
        df = pd.read_csv(path)
        table = _build_table(df, reservoir)
        out = DATASETS / f"pvt_{reservoir}.json"
        out.write_text(json.dumps(table, indent=2))
        print(
            f"{reservoir}: Pb={table['pb_psi']:.1f} psi, "
            f"P observed {table['p_range_observed'][0]:.0f}-"
            f"{table['p_range_observed'][1]:.0f} psi, wrote {out}"
        )


if __name__ == "__main__":
    main()
