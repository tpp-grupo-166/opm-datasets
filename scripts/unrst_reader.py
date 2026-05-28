"""Read per-cell arrays from OPM .UNRST and .INIT outputs.

Returns one dict per report step containing PRESSURE, RS, RV (if present),
SOIL, SGAS, plus the simulation time in days. Static PORV is loaded once
from .INIT.

All pressures are returned in their native unit (BARSA for METRIC decks,
PSIA for FIELD) — conversion happens upstream in the aggregator/extractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from resdata.resfile import ResdataFile


def load_porv(init_path: Path) -> np.ndarray:
    """PORV restricted to active cells. INIT stores PORV on the full grid;
    UNRST keywords are on the active subgrid. We mask inactive cells via
    ACTNUM if available, else by PORV > 0."""
    init = ResdataFile(str(init_path))
    porv = np.array(init["PORV"][0].numpy_copy(), dtype=float)
    if init.num_named_kw("ACTNUM") > 0:
        actnum = np.array(init["ACTNUM"][0].numpy_copy(), dtype=int)
        return porv[actnum == 1]
    return porv[porv > 0]


def load_pvtnum(init_path: Path, n_active: int) -> np.ndarray | None:
    """PVTNUM per active cell (1-indexed as in Eclipse), or None if absent.

    INIT may store PVTNUM either on the full grid or the active subgrid;
    we accept either and convert to active-cell length."""
    init = ResdataFile(str(init_path))
    if init.num_named_kw("PVTNUM") == 0:
        return None
    raw = np.array(init["PVTNUM"][0].numpy_copy(), dtype=int)
    if raw.size == n_active:
        return raw
    # Full-grid case: mask via ACTNUM or by selecting non-zero PVTNUM
    if init.num_named_kw("ACTNUM") > 0:
        actnum = np.array(init["ACTNUM"][0].numpy_copy(), dtype=int)
        return raw[actnum == 1]
    return raw[raw > 0]


def _safe_named(rst: ResdataFile, name: str, step: int) -> np.ndarray | None:
    if rst.num_named_kw(name) == 0:
        return None
    kw = rst.iget_named_kw(name, step)
    return np.array(kw.numpy_copy(), dtype=float)


def iter_report_steps(unrst_path: Path) -> Iterator[dict]:
    rst = ResdataFile(str(unrst_path))
    n_steps = rst.num_named_kw("PRESSURE")
    for i in range(n_steps):
        yield {
            "step": i,
            "sim_days": rst.iget_restart_sim_days(i),
            "pressure": _safe_named(rst, "PRESSURE", i),
            "rs": _safe_named(rst, "RS", i),
            "rv": _safe_named(rst, "RV", i),
            "soil": _safe_named(rst, "SOIL", i),
            "sgas": _safe_named(rst, "SGAS", i),
            "swat": _safe_named(rst, "SWAT", i),
        }
