"""Run 5 hardcoded Norne variants with drastic schedule changes.

Variants modify only WCONINJE rates inside the historical schedule
(`INCLUDE/BC0407_HIST01122006.SCH`). Producer controls (WCONHIST) are
left alone — they represent observed history and changing them would
require also changing the well-management heuristics.

Each WCONINJE block gets multiplied by a per-variant factor that may
depend on its position in the schedule (a proxy for time). The blocks
are roughly time-ordered, so block_idx / total_blocks ≈ fraction of
simulated time.

Output:
  - runs/norne_variant_N/      per-sim work dirs (auto-cleaned after run)
  - norne_variants_output/fpr_variant_N.csv
  - plots_pvt/norne_variants_fpr.png
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resdata.summary import Summary

# Make scripts/decks and scripts/extractor importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decks.norne import CONFIG as NORNE_CONFIG  # noqa: E402
from extractor import extract_features  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DECK_DIR = ROOT / "models" / "norne"
SCH_REL = "INCLUDE/BC0407_HIST01122006.SCH"
MAIN_DECK = "NORNE_ATW2013.DATA"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "norne_variants_output"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = ROOT / "plots_pvt"
PLOTS_DIR.mkdir(exist_ok=True)
DOCKER_IMAGE = "openporousmedia/opmreleases:latest"
FLOW_TIMEOUT_S = 1800


# Match a rate line inside a WCONINJE block.
#   'WELL'    'GAS|WAT'  1*    'RATE'   <number>   ...
INJ_RATE_RE = re.compile(
    r"^(\s*'[^']+'\s+'(?:GAS|WAT)'\s+\S+\s+'RATE'\s+)([\d.]+)(.*)$"
)


@dataclass
class Variant:
    name: str
    sim_id: int
    description: str
    factor: Callable[[int, int], float]   # (block_idx, total_blocks) -> mult


def _ramp(idx: int, total: int, off_frac: float, ramp_frac: float, hi: float) -> float:
    """0 for the first off_frac, ramp from 0 to hi over the next ramp_frac,
    then hold at hi until the end."""
    frac = idx / max(1, total - 1)
    if frac < off_frac:
        return 0.01
    if frac < off_frac + ramp_frac:
        return 0.01 + (hi - 0.01) * (frac - off_frac) / ramp_frac
    return hi


def _shut_in(idx: int, total: int) -> float:
    """Normal injection, then a shut-in window, then enhanced recovery."""
    frac = idx / max(1, total - 1)
    if frac < 0.35:
        return 1.0
    if frac < 0.55:        # 20% of schedule with effectively zero injection
        return 0.01
    return 1.6


VARIANTS = [
    Variant("v1_baseline", 1,
            "Original historical schedule, no modification.",
            lambda i, n: 1.0),
    Variant("v2_no_injection", 2,
            "All WCONINJE rates set ~0. Pure depletion.",
            lambda i, n: 0.01),
    Variant("v3_aggressive_inj", 3,
            "All WCONINJE rates × 3. Forced over-pressurization.",
            lambda i, n: 3.0),
    Variant("v4_ramp_up", 4,
            "No injection for first 25% of schedule, then ramp to 2× over next 25%, then hold.",
            lambda i, n: _ramp(i, n, off_frac=0.25, ramp_frac=0.25, hi=2.0)),
    Variant("v5_mid_shut_in", 5,
            "Normal injection, mid-life shut-in (35-55%), then enhanced (1.6×) recovery.",
            _shut_in),
]


# ---------------------------------------------------------------------------


def scale_wconinje(text: str, factor_fn: Callable[[int, int], float]) -> tuple[str, int]:
    """Walk the schedule line by line. Inside each WCONINJE block, multiply
    rate values by factor_fn(block_idx, total). Returns (new_text, n_records_changed).
    """
    total = sum(1 for L in text.splitlines() if L.lstrip().startswith("WCONINJE"))
    out: list[str] = []
    in_block = False
    block_idx = -1
    current_factor = 1.0
    n_records = 0

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("WCONINJE"):
            in_block = True
            block_idx += 1
            current_factor = factor_fn(block_idx, total)
            out.append(line)
            continue
        if in_block:
            if stripped == "/":
                in_block = False
                out.append(line)
                continue
            m = INJ_RATE_RE.match(line)
            if m:
                prefix, rate_str, tail = m.groups()
                new_rate = max(0.001, float(rate_str) * current_factor)
                out.append(f"{prefix}{new_rate:.4f}{tail}")
                n_records += 1
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out), n_records


def run_variant(variant: Variant) -> dict:
    sim_dir = RUNS_DIR / f"norne_{variant.name}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    started = time.perf_counter()
    print(f"\n[{variant.name}] {variant.description}")
    print(f"  copying deck → {sim_dir}")
    shutil.copytree(DECK_DIR, sim_dir)

    sch_path = sim_dir / SCH_REL
    original = sch_path.read_text(encoding="latin-1")
    patched, n_changed = scale_wconinje(original, variant.factor)
    sch_path.write_bytes(patched.encode("latin-1"))
    print(f"  patched {n_changed} WCONINJE rate records")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{sim_dir.resolve()}:/shared_host",
        DOCKER_IMAGE,
        "flow",
        "--output-dir=/shared_host",
        f"/shared_host/{MAIN_DECK}",
    ]
    print(f"  running OPM Flow…")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FLOW_TIMEOUT_S)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        # Persist log tail for debugging.
        log = sim_dir.parent / f"norne_{variant.name}.failed.log"
        log.write_text(
            "=== STDOUT tail ===\n" + "\n".join(proc.stdout.splitlines()[-200:])
            + "\n\n=== STDERR tail ===\n" + "\n".join(proc.stderr.splitlines()[-200:])
        )
        print(f"  FAILED (exit {proc.returncode}) — log → {log}")
        return {"variant": variant.name, "ok": False, "elapsed_s": elapsed, "df": None}

    summary_base = sim_dir / MAIN_DECK.removesuffix(".DATA")
    print(f"  extracting full schema with extract_features…")
    # Variants only modify the schedule, not the geology/PVT levers, so we
    # pass the LHS defaults. The extractor consumes UNRST/INIT for per-cell
    # PVT aggregation, so the sim dir must still be alive at this point.
    params = {"phi_mult": 1.0, "k_mult": 1.0, "p_init_shift_bar": 0.0}
    df = extract_features(NORNE_CONFIG, summary_base, variant.sim_id, params)

    csv_path = OUT_DIR / f"fpr_{variant.name}.csv"
    df_quick = pd.DataFrame({
        "tiempo_dias": df["tiempo_dias"].values,
        "FPR_psi": df["Presion_Reservorio_psi"].values,
    })
    df_quick.to_csv(csv_path, index=False)
    fpr_psi = df_quick["FPR_psi"].values
    print(f"  wrote {csv_path}  ({len(df)} timesteps, "
          f"FPR {fpr_psi.min():.0f}–{fpr_psi.max():.0f} psi, "
          f"elapsed {elapsed:.0f}s)")

    # Clean up the per-sim dir (40+ MB of OPM outputs).
    shutil.rmtree(sim_dir, ignore_errors=True)
    return {"variant": variant.name, "sim_id": variant.sim_id, "ok": True,
            "elapsed_s": elapsed, "df": df, "df_quick": df_quick,
            "description": variant.description}


def plot(results: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["k", "C3", "C0", "C2", "C1"]
    for r, c in zip(results, colors):
        if not r["ok"]:
            continue
        df = r["df_quick"]
        label = f"{r['variant']} — FPR Δ {df['FPR_psi'].max() - df['FPR_psi'].min():.0f} psi"
        ax.plot(df["tiempo_dias"], df["FPR_psi"], lw=1.6, color=c, label=label)
    ax.set_xlabel("tiempo [días]")
    ax.set_ylabel("Field Pressure (FPR) [psi]")
    ax.set_title("Norne — 5 schedule variants (WCONINJE multiplier varied)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "norne_variants_fpr.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel OPM Flow processes. Each docker container "
                             "uses ~3-4 GB. Default Docker Desktop = 8 GB → "
                             "≥3 workers may OOM-kill containers.")
    args = parser.parse_args()
    n_workers = max(1, args.workers)

    print(f"Running {len(VARIANTS)} Norne variants with {n_workers} worker(s).\n")
    results: list[dict] = []
    if n_workers == 1:
        for v in VARIANTS:
            results.append(run_variant(v))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(run_variant, v): v for v in VARIANTS}
            for fut in as_completed(futures):
                results.append(fut.result())
        # Sort to keep deterministic order for plotting/printing.
        order = {v.name: i for i, v in enumerate(VARIANTS)}
        results.sort(key=lambda r: order[r["variant"]])

    # Summary table.
    print("\n=== Summary ===")
    for r in results:
        if r["ok"]:
            df = r["df_quick"]
            print(f"  {r['variant']:<20}  FPR {df.FPR_psi.min():.0f}–{df.FPR_psi.max():.0f} psi  "
                  f"({df.FPR_psi.max() - df.FPR_psi.min():.0f} psi range, {r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['variant']:<20}  FAILED")

    # Concat all variant DataFrames into the unified dataset CSV.
    dataset_dir = ROOT / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    dfs = [r["df"] for r in results if r["ok"]]
    if dfs:
        full = pd.concat(dfs, ignore_index=True)
        out_csv = dataset_dir / "dataset_norne_schedule.csv"
        full.to_csv(out_csv, index=False)
        print(f"\nWrote dataset → {out_csv} "
              f"({len(full)} rows, {full['sim_id'].nunique()} sims, "
              f"FPR {full['Presion_Reservorio_psi'].min():.0f}–"
              f"{full['Presion_Reservorio_psi'].max():.0f} psi)")

    out = plot(results)
    print(f"Wrote plot → {out}")


if __name__ == "__main__":
    main()
