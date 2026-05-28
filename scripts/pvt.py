"""Parser for Eclipse PVTO/PVTG/PVDG include files.

Returns one `PvtTable` per PVTNUM region. Each table evaluates Bo, Bg, Rs
per cell given (P, Rs) and (P, Rv). Internal storage is always in FIELD
units (psia, rb/STB, rb/scf, scf/STB), independent of the source deck.

Region conventions:

- PVTO/PVTG (Norne): records terminated by `/`, regions separated by a
  bare `/`.
- PVDG (Volve): each region is a single record terminated by `/`; the
  number of regions equals the number of PVTO regions (Eclipse pairs
  them by PVTNUM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


BAR_TO_PSI = 14.5037738
SM3_OIL_TO_STB = 6.28981077
SM3_GAS_TO_SCF = 35.3146667


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.split("--", 1)[0].strip()
        if not line:
            continue
        for tok in line.split():
            out.append(tok)
    return out


def _find_keyword(tokens: list[str], name: str, start: int = 0) -> int:
    for i in range(start, len(tokens)):
        if tokens[i] == name:
            return i + 1
    return -1


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _parse_regions_with_separator(tokens: list[str], start: int) -> list[list[list[float]]]:
    """PVTO/PVTG style: records end with `/`, regions separated by bare `/`."""
    regions: list[list[list[float]]] = []
    current_region: list[list[float]] = []
    current_record: list[float] = []
    i = start
    while i < len(tokens):
        t = tokens[i]
        if t == "/":
            if current_record:
                current_region.append(current_record)
                current_record = []
            else:
                # Bare /: end of region
                if current_region:
                    regions.append(current_region)
                    current_region = []
        elif _is_number(t):
            current_record.append(float(t))
        else:
            # Next keyword
            break
        i += 1
    if current_record:
        current_region.append(current_record)
    if current_region:
        regions.append(current_region)
    return regions


def _parse_pvdg(tokens: list[str], start: int, n_regions: int) -> list[list[float]]:
    """PVDG style: each region is one record (flat list of P,Bg,mu triplets)."""
    records: list[list[float]] = []
    current: list[float] = []
    i = start
    while i < len(tokens) and len(records) < n_regions:
        t = tokens[i]
        if t == "/":
            if current:
                records.append(current)
                current = []
        elif _is_number(t):
            current.append(float(t))
        else:
            break
        i += 1
    if current and len(records) < n_regions:
        records.append(current)
    return records


@dataclass
class PvtTable:
    pvto_rs_scf_stb: np.ndarray
    pvto_pb_psi: np.ndarray
    pvto_bo_sat: np.ndarray
    pvto_undersat_slope: float
    pvtg_p_psi: np.ndarray
    pvtg_rv_grids: list[np.ndarray]
    pvtg_bg_grids: list[np.ndarray]

    def bo_cell(self, p_psi: np.ndarray, rs_scf_stb: np.ndarray) -> np.ndarray:
        p = np.asarray(p_psi, dtype=float)
        rs = np.asarray(rs_scf_stb, dtype=float)
        pb = np.interp(rs, self.pvto_rs_scf_stb, self.pvto_pb_psi)
        bo_sat = np.interp(rs, self.pvto_rs_scf_stb, self.pvto_bo_sat)
        dp = np.maximum(p - pb, 0.0)
        return bo_sat + dp * self.pvto_undersat_slope

    def bg_cell(self, p_psi: np.ndarray, rv_scf_stb: np.ndarray | None) -> np.ndarray:
        p = np.asarray(p_psi, dtype=float)
        bg_at_sat_p = np.array(
            [grid[-1] for grid in self.pvtg_bg_grids], dtype=float
        )
        if rv_scf_stb is None:
            return np.interp(p, self.pvtg_p_psi, bg_at_sat_p)

        rv = np.asarray(rv_scf_stb, dtype=float)
        idx_hi = np.searchsorted(self.pvtg_p_psi, p, side="right")
        idx_hi = np.clip(idx_hi, 1, len(self.pvtg_p_psi) - 1)
        idx_lo = idx_hi - 1

        out = np.empty_like(p, dtype=float)
        for lo in np.unique(idx_lo):
            mask = idx_lo == lo
            hi = lo + 1
            rv_lo = self.pvtg_rv_grids[lo]
            bg_lo = self.pvtg_bg_grids[lo]
            rv_hi = self.pvtg_rv_grids[hi]
            bg_hi = self.pvtg_bg_grids[hi]
            p_lo = self.pvtg_p_psi[lo]
            p_hi = self.pvtg_p_psi[hi]

            rv_m = rv[mask]
            p_m = p[mask]
            b_lo = np.interp(rv_m, rv_lo, bg_lo)
            b_hi = np.interp(rv_m, rv_hi, bg_hi)
            w = np.clip((p_m - p_lo) / (p_hi - p_lo), 0.0, 1.0)
            out[mask] = b_lo * (1.0 - w) + b_hi * w
        return out


def _build_pvto_arrays(region_records: list[list[float]],
                        p_conv: float, rs_conv: float, bo_conv: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rs_list, pb_list, bo_list, slopes = [], [], [], []
    for rec in region_records:
        rs = rec[0]
        rest = rec[1:]
        n = len(rest) // 3
        ps = np.array([rest[3 * k] for k in range(n)], dtype=float)
        bos = np.array([rest[3 * k + 1] for k in range(n)], dtype=float)
        rs_list.append(rs)
        pb_list.append(ps[0])
        bo_list.append(bos[0])
        if n > 1:
            slopes.append((bos[-1] - bos[0]) / (ps[-1] - ps[0]))
    rs_arr = np.array(rs_list) * rs_conv
    pb_arr = np.array(pb_list) * p_conv
    bo_arr = np.array(bo_list) * bo_conv
    order = np.argsort(rs_arr)
    slope = float(np.mean(slopes)) / p_conv if slopes else 0.0
    return rs_arr[order], pb_arr[order], bo_arr[order], slope


def _build_pvtg_arrays(region_records: list[list[float]],
                        p_conv: float, rs_conv: float, bg_conv: float
                        ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    ps, rv_grids, bg_grids = [], [], []
    for rec in region_records:
        p_val = rec[0]
        rest = rec[1:]
        n = len(rest) // 3
        rvs = np.array([rest[3 * k] for k in range(n)], dtype=float)
        bgs = np.array([rest[3 * k + 1] for k in range(n)], dtype=float)
        o = np.argsort(rvs)
        ps.append(p_val)
        rv_grids.append(rvs[o] * rs_conv)
        bg_grids.append(bgs[o] * bg_conv)
    p_arr = np.array(ps) * p_conv
    o = np.argsort(p_arr)
    return p_arr[o], [rv_grids[i] for i in o], [bg_grids[i] for i in o]


def _build_pvdg_arrays(record: list[float],
                        p_conv: float, bg_conv: float
                        ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    n = len(record) // 3
    ps = np.array([record[3 * k] for k in range(n)], dtype=float) * p_conv
    bgs = np.array([record[3 * k + 1] for k in range(n)], dtype=float) * bg_conv
    order = np.argsort(ps)
    ps = ps[order]
    bgs = bgs[order]
    rv_grids = [np.array([0.0]) for _ in ps]
    bg_grids = [np.array([bg]) for bg in bgs]
    return ps, rv_grids, bg_grids


def parse_pvt_include(path: Path, unit_system: Literal["METRIC", "FIELD"]) -> list[PvtTable]:
    """Parse PVTO + (PVTG or PVDG) into one PvtTable per PVTNUM region."""
    text = Path(path).read_text(encoding="latin-1")
    tokens = _tokenize(text)

    if unit_system == "METRIC":
        p_conv = BAR_TO_PSI
        rs_conv = SM3_GAS_TO_SCF / SM3_OIL_TO_STB
        bo_conv = 1.0
        bg_conv = SM3_OIL_TO_STB / SM3_GAS_TO_SCF
    else:
        p_conv = rs_conv = bo_conv = bg_conv = 1.0

    pvto_idx = _find_keyword(tokens, "PVTO")
    if pvto_idx < 0:
        raise RuntimeError(f"PVTO not found in {path}")
    pvto_regions = _parse_regions_with_separator(tokens, pvto_idx)
    n_regions = len(pvto_regions)

    pvtg_idx = _find_keyword(tokens, "PVTG")
    is_pvtg = pvtg_idx >= 0
    if is_pvtg:
        pvtg_regions = _parse_regions_with_separator(tokens, pvtg_idx)
        if len(pvtg_regions) != n_regions:
            raise RuntimeError(
                f"PVTO has {n_regions} regions but PVTG has {len(pvtg_regions)}"
            )
    else:
        pvdg_idx = _find_keyword(tokens, "PVDG")
        if pvdg_idx < 0:
            raise RuntimeError(f"Neither PVTG nor PVDG in {path}")
        pvdg_records = _parse_pvdg(tokens, pvdg_idx, n_regions)
        if len(pvdg_records) != n_regions:
            raise RuntimeError(
                f"PVTO has {n_regions} regions but PVDG parsed {len(pvdg_records)}"
            )

    tables: list[PvtTable] = []
    for ri in range(n_regions):
        rs, pb, bo, slope = _build_pvto_arrays(
            pvto_regions[ri], p_conv, rs_conv, bo_conv
        )
        if is_pvtg:
            pp, rv_grids, bg_grids = _build_pvtg_arrays(
                pvtg_regions[ri], p_conv, rs_conv, bg_conv
            )
        else:
            pp, rv_grids, bg_grids = _build_pvdg_arrays(
                pvdg_records[ri], p_conv, bg_conv
            )
        tables.append(
            PvtTable(
                pvto_rs_scf_stb=rs,
                pvto_pb_psi=pb,
                pvto_bo_sat=bo,
                pvto_undersat_slope=slope,
                pvtg_p_psi=pp,
                pvtg_rv_grids=rv_grids,
                pvtg_bg_grids=bg_grids,
            )
        )
    return tables
