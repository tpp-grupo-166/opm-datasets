"""Per-step PVT aggregation from cell-level outputs.

Multi-region capable: each cell is evaluated with the PvtTable that
matches its PVTNUM. PVTNUM is read from the .INIT once. If a deck has
only one PVTNUM, this collapses to the single-table case used by Norne.

Field-level aggregates per report step:

    Bo_field = Σ(Bo_i * PORV_i * SOIL_i) / Σ(PORV_i * SOIL_i)
    Bg_field = Σ(Bg_i * PORV_i * SGAS_i) / Σ(PORV_i * SGAS_i)
    Rs_field = Σ(Rs_i * PORV_i * SOIL_i) / Σ(PORV_i * SOIL_i)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pvt import BAR_TO_PSI, PvtTable, SM3_GAS_TO_SCF, SM3_OIL_TO_STB
from unrst_reader import iter_report_steps, load_porv, load_pvtnum


def _eval_per_region(
    tables: list[PvtTable],
    pvtnum: np.ndarray | None,
    p_cells: np.ndarray,
    rs_cells: np.ndarray,
    rv_cells: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Bo and Bg per cell, dispatching by region."""
    n = p_cells.shape[0]
    bo = np.empty(n, dtype=float)
    bg = np.empty(n, dtype=float)
    if pvtnum is None or len(tables) == 1:
        t = tables[0]
        bo[:] = t.bo_cell(p_cells, rs_cells)
        bg[:] = t.bg_cell(p_cells, rv_cells)
        return bo, bg
    for region_idx in np.unique(pvtnum):
        mask = pvtnum == region_idx
        t = tables[int(region_idx) - 1]
        rv_sub = rv_cells[mask] if rv_cells is not None else None
        bo[mask] = t.bo_cell(p_cells[mask], rs_cells[mask])
        bg[mask] = t.bg_cell(p_cells[mask], rv_sub)
    return bo, bg


def aggregate_pvt(
    deck_basename: Path,
    pvt_tables: list[PvtTable],
    unit_system: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns sim_days, Bo_field, Bg_field, Rs_field per report step.

    All outputs in FIELD units (Bo: rb/STB, Bg: rb/scf, Rs: scf/STB).
    """
    init_path = deck_basename.with_suffix(".INIT")
    porv = load_porv(init_path)
    pvtnum = load_pvtnum(init_path, n_active=len(porv))

    if unit_system == "METRIC":
        p_to_psi = BAR_TO_PSI
        rs_to_field = SM3_GAS_TO_SCF / SM3_OIL_TO_STB
        rv_to_field = SM3_OIL_TO_STB / SM3_GAS_TO_SCF
    else:
        p_to_psi = rs_to_field = rv_to_field = 1.0

    days_list: list[float] = []
    bo_list: list[float] = []
    bg_list: list[float] = []
    rs_list: list[float] = []

    for rec in iter_report_steps(deck_basename.with_suffix(".UNRST")):
        p_cells = rec["pressure"] * p_to_psi
        rs_cells = (rec["rs"] if rec["rs"] is not None else np.zeros_like(p_cells)) * rs_to_field
        rv_cells = rec["rv"] * rv_to_field if rec["rv"] is not None else None
        sgas = rec["sgas"] if rec["sgas"] is not None else np.zeros_like(p_cells)
        if rec["soil"] is not None:
            soil = rec["soil"]
        elif rec["swat"] is not None:
            soil = 1.0 - rec["swat"] - sgas
        else:
            soil = np.zeros_like(p_cells)
        soil = np.clip(soil, 0.0, 1.0)

        bo_i, bg_i = _eval_per_region(pvt_tables, pvtnum, p_cells, rs_cells, rv_cells)

        oil_pv = porv * soil
        gas_pv = porv * sgas
        oil_total = oil_pv.sum()
        gas_total = gas_pv.sum()

        bo_field = float((bo_i * oil_pv).sum() / oil_total) if oil_total > 0 else float("nan")
        rs_field = float((rs_cells * oil_pv).sum() / oil_total) if oil_total > 0 else float("nan")
        # Bg fallback: when there is no free gas (field above bubble point),
        # PV-weighted Bg gives the in-situ gas FVF that *would* characterise
        # any gas evolved at the current cell pressures. Avoids NaN.
        if gas_total > 0:
            bg_field = float((bg_i * gas_pv).sum() / gas_total)
        else:
            pv_total = porv.sum()
            bg_field = float((bg_i * porv).sum() / pv_total) if pv_total > 0 else float("nan")

        days_list.append(rec["sim_days"])
        bo_list.append(bo_field)
        bg_list.append(bg_field)
        rs_list.append(rs_field)

    return (
        np.array(days_list, dtype=float),
        np.array(bo_list, dtype=float),
        np.array(bg_list, dtype=float),
        np.array(rs_list, dtype=float),
    )
