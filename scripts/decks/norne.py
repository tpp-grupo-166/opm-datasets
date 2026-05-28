"""Norne model configuration: 3 levers, METRIC units, 2-file render."""

from __future__ import annotations

import re
from pathlib import Path

from .base import DeckConfig
from pvt import parse_pvt_include

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECK_DIR = PROJECT_ROOT / "models" / "norne"
MAIN_DECK = "NORNE_ATW2013.DATA"
EQUIL_INCLUDE_REL = "INCLUDE/PETRO/E3.prop"
PVT_INCLUDE = DECK_DIR / "INCLUDE" / "PVT" / "PVT-WET-GAS.INC"


LEVER_RANGES = {
    "k_mult": (0.7, 1.5),
    "phi_mult": (0.85, 1.15),
    "p_init_shift_bar": (-13.79, 13.79),
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
    deck = _insert_multiply(deck, params["k_mult"], params["phi_mult"])
    equil = _shift_equil(equil, params["p_init_shift_bar"])
    return {MAIN_DECK: deck, EQUIL_INCLUDE_REL: equil}


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
