"""Norne model configuration: 7 levers, METRIC units, 3-file render.

The four PVT levers below produce a *different* PVT per sim — useful for
training a PVT-as-input model that needs to see multiple PVTs to learn how
the PVT modulates the response.

  - pb_shift_bar          shifts every pressure in PVTO (saturated +
                          undersaturated continuation rows)
  - rs_pb_mult            scales the Rs of every PVTO record (the gas-in-
                          solution at the saturated bubble point)
  - bo_undersat_slope_mult rescales the slope of Bo above Pb inside each
                          PVTO record:
                              Bo_new[i] = Bo_sat + slope_mult · (Bo_old[i] − Bo_sat)
                          The saturated point itself is preserved.
  - oil_density_mult       multiplies the oil surface density in each row
                          of the DENSITY block.

The 3 schedule/geometry levers (k_mult, phi_mult, p_init_shift_bar) remain
unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import DeckConfig
from pvt import parse_pvt_include

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECK_DIR = PROJECT_ROOT / "models" / "norne"
MAIN_DECK = "NORNE_ATW2013.DATA"
EQUIL_INCLUDE_REL = "INCLUDE/PETRO/E3.prop"
PVT_INCLUDE_REL = "INCLUDE/PVT/PVT-WET-GAS.INC"
PVT_INCLUDE = DECK_DIR / PVT_INCLUDE_REL


LEVER_RANGES = {
    "k_mult": (0.7, 1.5),
    "phi_mult": (0.85, 1.15),
    "p_init_shift_bar": (-13.79, 13.79),
    "pb_shift_bar":           (-35.0, 35.0),   # ±~500 psi shift of saturated curve
    "rs_pb_mult":             (0.7, 1.4),      # gas solubility at Pb scaled
    "bo_undersat_slope_mult": (0.5, 2.0),      # oil compressibility above Pb
    "oil_density_mult":       (0.92, 1.08),    # ±8% oil density → Bo level shift
    "bg_mult":                (0.75, 1.30),    # scales Bg in PVTG (gas formation factor)
}


# Calibrated once from runs/norne_baseline INIT (volume-weighted, NTG-aware)
STATIC_FEATURES = {
    "baseline_porosity": 0.2470,
    "baseline_perm_md": 487.49,
    "espesor_neto_m": 172.41,
    "area_m2": 1.6962e7,
}

BASELINE_PB_BAR = 250.0


def render_deck(params: dict) -> dict[str, str]:
    deck = (DECK_DIR / MAIN_DECK).read_text()
    equil = (DECK_DIR / EQUIL_INCLUDE_REL).read_text()
    pvt = PVT_INCLUDE.read_text()

    deck = _insert_multiply(deck, params["k_mult"], params["phi_mult"])
    equil = _shift_equil(equil, params["p_init_shift_bar"])
    pvt = _apply_pvto_levers(
        pvt,
        pb_shift_bar=params.get("pb_shift_bar", 0.0),
        rs_mult=params.get("rs_pb_mult", 1.0),
        bo_slope_mult=params.get("bo_undersat_slope_mult", 1.0),
    )
    pvt = _scale_pvtg_bg(pvt, bg_mult=params.get("bg_mult", 1.0))
    pvt = _scale_density(pvt, oil_mult=params.get("oil_density_mult", 1.0))
    return {MAIN_DECK: deck, EQUIL_INCLUDE_REL: equil, PVT_INCLUDE_REL: pvt}


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
    block = (
        "\n"
        "MULTIPLY\n"
        f"   'PORO'  {phi_mult:.5f} /\n"
        f"   'PERMX' {k_mult:.5f} /\n"
        f"   'PERMY' {k_mult:.5f} /\n"
        f"   'PERMZ' {k_mult:.5f} /\n"
        "/\n"
    )
    marker = "\nEDIT\n"
    if marker not in text:
        raise RuntimeError("EDIT marker not found in deck")
    return text.replace(marker, block + marker, 1)


_EQUIL_ROW = re.compile(
    r"^(\s*)([-\d.]+)(\s+)([-\d.]+)(\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s*/.*)$"
)


def _shift_equil(text: str, p_shift_bar: float) -> str:
    if p_shift_bar == 0:
        return text
    out: list[str] = []
    in_equil = False
    rows_shifted = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_equil:
            if stripped == "EQUIL":
                in_equil = True
            out.append(line)
            continue
        if not stripped or stripped.startswith("--"):
            out.append(line)
            continue
        match = _EQUIL_ROW.match(line)
        if match is None:
            in_equil = False
            out.append(line)
            continue
        leading_ws, datum, sep, pressure, tail = match.groups()
        new_pressure = float(pressure) + p_shift_bar
        out.append(f"{leading_ws}{datum}{sep}{new_pressure:.4f}{tail}")
        rows_shifted += 1
    if rows_shifted == 0:
        raise RuntimeError("No EQUIL data rows matched")
    return "\n".join(out)


# PVTO line patterns.
# Record-start line: "RS  P  Bo  mu" — 4 numeric tokens.
# Continuation:      "    P  Bo  mu" — 3 numeric tokens.
_PVTO_RECORD_START = re.compile(
    r"^(\s*)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s*/?\s*)$"
)
_PVTO_CONT = re.compile(
    r"^(\s+)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s*/?\s*)$"
)


def _apply_pvto_levers(text: str, pb_shift_bar: float = 0.0,
                       rs_mult: float = 1.0,
                       bo_slope_mult: float = 1.0) -> str:
    """Apply (pb_shift, rs_mult, bo_slope_mult) to the PVTO block in a
    single pass. Tracks the current record's saturated Bo so the slope
    of the undersaturated extension can be rescaled correctly.

    `pb_shift_bar`     shifts every PVTO pressure by this delta (bar).
    `rs_mult`           multiplies Rs in every record-start line.
    `bo_slope_mult`     rescales Bo on continuation rows:
                            Bo_new = Bo_sat + slope_mult · (Bo_old − Bo_sat)
                       (preserves the saturated point.)
    """
    no_op = (pb_shift_bar == 0.0 and rs_mult == 1.0 and bo_slope_mult == 1.0)
    if no_op:
        return text

    out: list[str] = []
    in_pvto = False
    rows_changed = 0
    current_bo_sat: float | None = None

    for line in text.split("\n"):
        stripped = line.strip()

        if not in_pvto:
            if stripped.startswith("PVTO"):
                in_pvto = True
            out.append(line)
            continue

        # Inside PVTO. Pass through blanks and comments.
        if not stripped or stripped.startswith("--"):
            out.append(line); continue
        # Bare "/" or non-numeric line means block end / next keyword.
        if stripped == "/":
            in_pvto = False
            current_bo_sat = None
            out.append(line); continue
        if not stripped[0].isdigit():
            in_pvto = False
            current_bo_sat = None
            out.append(line); continue

        num_tokens = sum(
            1 for t in stripped.split() if re.fullmatch(r"[\d.]+", t)
        )

        if num_tokens == 4:
            m = _PVTO_RECORD_START.match(line)
            if m:
                ws1, rs_str, ws2, p_str, ws3, bo_str, ws4, mu_str, tail = m.groups()
                rs_new = max(0.0, float(rs_str) * rs_mult)
                p_new = max(1.0, float(p_str) + pb_shift_bar)
                bo_sat = float(bo_str)   # saturated Bo — preserved
                current_bo_sat = bo_sat
                out.append(
                    f"{ws1}{rs_new:.4f}{ws2}{p_new:.4f}{ws3}"
                    f"{bo_sat:.5f}{ws4}{mu_str}{tail}"
                )
                rows_changed += 1
                continue
        elif num_tokens == 3:
            m = _PVTO_CONT.match(line)
            if m:
                ws1, p_str, ws2, bo_str, ws3, mu_str, tail = m.groups()
                p_new = max(1.0, float(p_str) + pb_shift_bar)
                bo_old = float(bo_str)
                if current_bo_sat is not None and bo_slope_mult != 1.0:
                    bo_new = current_bo_sat + bo_slope_mult * (bo_old - current_bo_sat)
                else:
                    bo_new = bo_old
                out.append(
                    f"{ws1}{p_new:.4f}{ws2}{bo_new:.5f}{ws3}{mu_str}{tail}"
                )
                rows_changed += 1
                continue

        # Anything unusual passes through unchanged.
        out.append(line)

    if rows_changed == 0:
        raise RuntimeError("No PVTO rows matched — check the deck format")
    return "\n".join(out)


# PVTG row patterns.
# Saturated row (record start): "P  Rv  Bg  mu" — 4 numeric tokens.
# Continuation:                 "    Rv  Bg  mu" — 3 numeric tokens.
_PVTG_RECORD_START = re.compile(
    r"^(\s*)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s*/?\s*)$"
)
_PVTG_CONT = re.compile(
    r"^(\s+)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s*/?\s*)$"
)


def _scale_pvtg_bg(text: str, bg_mult: float = 1.0) -> str:
    """Multiply the Bg column inside PVTG by `bg_mult`. Bg sits in column 3
    on record-start lines (after P and Rv) and column 2 on continuation
    lines. Other columns (P, Rv, mu) untouched."""
    if bg_mult == 1.0:
        return text
    out: list[str] = []
    in_pvtg = False
    rows_scaled = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_pvtg:
            if stripped.startswith("PVTG"):
                in_pvtg = True
            out.append(line); continue
        if not stripped or stripped.startswith("--"):
            out.append(line); continue
        if stripped == "/":
            in_pvtg = False
            out.append(line); continue
        if not stripped[0].isdigit():
            in_pvtg = False
            out.append(line); continue

        num_tokens = sum(
            1 for t in stripped.split() if re.fullmatch(r"[\d.]+", t)
        )
        if num_tokens == 4:
            m = _PVTG_RECORD_START.match(line)
            if m:
                ws1, p, ws2, rv, ws3, bg, ws4, mu, tail = m.groups()
                new_bg = max(1e-9, float(bg) * bg_mult)
                out.append(
                    f"{ws1}{p}{ws2}{rv}{ws3}{new_bg:.6f}{ws4}{mu}{tail}"
                )
                rows_scaled += 1
                continue
        elif num_tokens == 3:
            m = _PVTG_CONT.match(line)
            if m:
                ws1, rv, ws2, bg, ws3, mu, tail = m.groups()
                new_bg = max(1e-9, float(bg) * bg_mult)
                out.append(
                    f"{ws1}{rv}{ws2}{new_bg:.6f}{ws3}{mu}{tail}"
                )
                rows_scaled += 1
                continue
        out.append(line)
    if rows_scaled == 0:
        raise RuntimeError("No PVTG rows matched — check the deck format")
    return "\n".join(out)


# DENSITY block. Each data line: oil_density  water_density  gas_density  /
_DENSITY_ROW = re.compile(
    r"^(\s*)([\d.]+)(\s+)([\d.]+)(\s+)([\d.]+)(\s*/.*)$"
)


def _scale_density(text: str, oil_mult: float = 1.0) -> str:
    """Multiply the oil surface density (column 1) in every DENSITY data
    row by `oil_mult`. Water and gas densities are left untouched."""
    if oil_mult == 1.0:
        return text
    out: list[str] = []
    in_density = False
    rows_scaled = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_density:
            if stripped == "DENSITY":
                in_density = True
            out.append(line)
            continue
        if not stripped or stripped.startswith("--"):
            out.append(line); continue
        if not stripped[0].isdigit():
            in_density = False
            out.append(line); continue
        m = _DENSITY_ROW.match(line)
        if m:
            ws1, od_str, ws2, wd_str, ws3, gd_str, tail = m.groups()
            new_od = float(od_str) * oil_mult
            out.append(
                f"{ws1}{new_od:.4f}{ws2}{wd_str}{ws3}{gd_str}{tail}"
            )
            rows_scaled += 1
        else:
            out.append(line)
    if rows_scaled == 0:
        raise RuntimeError("No DENSITY rows matched — check the deck format")
    return "\n".join(out)


CONFIG = DeckConfig(
    name="norne",
    deck_dir=DECK_DIR,
    main_deck_filename=MAIN_DECK,
    lever_ranges=LEVER_RANGES,
    render_deck=render_deck,
    static_features=STATIC_FEATURES,
    unit_system="METRIC",
    baseline_pb=BASELINE_PB_BAR,
    flow_timeout_s=1800,
    pvt_tables=parse_pvt_include(PVT_INCLUDE, unit_system="METRIC"),
)
