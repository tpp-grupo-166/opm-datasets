"""Volve model configuration: 4 levers, METRIC units, single-deck render."""

from __future__ import annotations

import re
from pathlib import Path

from .base import DeckConfig
from pvt import parse_pvt_include

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECK_DIR = PROJECT_ROOT / "models" / "volve"
MAIN_DECK = "VOLVE_2016.DATA"
PVT_INCLUDE = DECK_DIR / "pvt_input_new_combined_PVDG_020610_perch_water_2914m.E100"


LEVER_RANGES = {
    "k_mult": (0.7, 1.5),
    "phi_mult": (0.85, 1.15),
    "p_init_shift_bar": (-13.79, 13.79),
    "qwinj_group_mult": (0.6, 1.4),
}


# Calibrated once from VOLVE_2016.INIT (volume-weighted, NTG-aware)
STATIC_FEATURES = {
    "baseline_porosity": 0.2212,
    "baseline_perm_md": 1645.55,
    "espesor_neto_m": 77.75,
    "area_m2": 6.7962e6,
}

BASELINE_PB_BAR = 280.0


def render_deck(params: dict) -> dict[str, str]:
    text = (DECK_DIR / MAIN_DECK).read_bytes().decode("latin-1")
    text = _insert_multiply(text, params["k_mult"], params["phi_mult"])
    text = _shift_equil(text, params["p_init_shift_bar"])
    text = _scale_gconinje(text, params["qwinj_group_mult"])
    return {MAIN_DECK: text}


def _insert_multiply(text: str, k_mult: float, phi_mult: float) -> str:
    block = (
        "\n"
        "-- Per-simulation MULTIPLY block injected by deck templater\n"
        "MULTIPLY\n"
        f"   'PORO'  {phi_mult:.5f} /\n"
        f"   'PERMX' {k_mult:.5f} /\n"
        f"   'PERMY' {k_mult:.5f} /\n"
        f"   'PERMZ' {k_mult:.5f} /\n"
        "/\n"
    )
    marker = "\nEDIT\n"
    if marker not in text:
        raise RuntimeError("EDIT marker not found in Volve deck")
    return text.replace(marker, block + marker, 1)


_EQUIL_ROW = re.compile(
    r"^(\s*[-\d.]+\s+)([-\d.]+)(\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\d+\s+\d+\s+\d+\s*/.*)$"
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
        prefix, pressure, tail = match.groups()
        new_pressure = float(pressure) + p_shift_bar
        out.append(f"{prefix}{new_pressure:.4f}{tail}")
        rows_shifted += 1
    if rows_shifted != 12:
        raise RuntimeError(f"Expected 12 EQUIL rows in Volve, shifted {rows_shifted}")
    return "\n".join(out)


_GCONINJE_LINE = re.compile(
    r"('FIELD'\s+'WAT'\s+'RATE'\s+)([\d.]+)(\s+\S+\s+\S+\s+[\d.]+\s+/)"
)


def _scale_gconinje(text: str, mult: float) -> str:
    if mult == 1.0:
        return text

    def _sub(match: re.Match) -> str:
        prefix, rate, tail = match.group(1), float(match.group(2)), match.group(3)
        return f"{prefix}{rate * mult:.2f}{tail}"

    new, n = _GCONINJE_LINE.subn(_sub, text, count=1)
    if n != 1:
        raise RuntimeError("GCONINJE FIELD WAT RATE record not found")
    return new


CONFIG = DeckConfig(
    name="volve",
    deck_dir=DECK_DIR,
    main_deck_filename=MAIN_DECK,
    lever_ranges=LEVER_RANGES,
    render_deck=render_deck,
    static_features=STATIC_FEATURES,
    unit_system="METRIC",
    baseline_pb=BASELINE_PB_BAR,
    flow_timeout_s=5400,
    pvt_tables=parse_pvt_include(PVT_INCLUDE, unit_system="METRIC"),
)
