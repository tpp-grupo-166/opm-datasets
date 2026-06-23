"""Run N Norne variants sampled with Latin Hypercube over 4 levers.

  - inj_mult_early / mid / late  — three time-chunked multipliers for
    every WCONINJE record in the historical schedule
    (`INCLUDE/BC0407_HIST01122006.SCH`).
  - pb_shift_bar  — shifts every pressure in the PVTO saturated table
    by ±21 bar (~±300 psi), giving each sim a *different* PVT curve.
    Critical for training a PVT-as-input model that needs to see
    multiple PVTs to learn how the PVT modulates the response.

The 3 injection multipliers are drawn from one of three regimes:

  - realistic   (60% of sims): each multiplier in [0.7, 1.3]
  - low-inj     (20% of sims): each multiplier in [0.01, 0.4]
  - high-inj    (20% of sims): each multiplier in [1.7, 3.0]

`pb_shift_bar` is uniform across [-21, +21] bar for every regime, so
schedule and PVT are decorrelated. LHS picks the 4-D tuple per sim.
Producer controls (WCONHIST) are left alone — they represent observed
history.

Output (filenames are parameterized by N to avoid clobbering prior runs):
  - runs/norne_simNN_<category>/  per-sim work dirs (auto-cleaned after run)
  - norne_variants_output/fpr_simNN_<category>.csv
  - datasets/dataset_norne_schedule_pvt_n{N}.csv      full schema, all sims
  - datasets/runs_log_norne_schedule_pvt_n{N}.csv     per-sim params + outcome
  - plots_pvt/norne_variants_n{N}_fpr.png         overlay of all curves
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
    category: str = ""
    params: dict | None = None             # per-third multipliers, for runs_log


PB_SHIFT_RANGE = (-21.0, 21.0)   # bar; ~±300 psi shift of PVTO saturated curve

# Multi-lever PVT space. Used by the `--pvt-only` pilot mode. Schedule
# multipliers stay at 1.0 so the PVT effect is isolated.
# pb_shift_bar widened to ±35 bar (~±500 psi) so Volve's Pb (~3625 + 430 psi)
# falls inside the training envelope. bg_mult added so Volve's lower Bg
# is also reachable.
PVT_LEVER_RANGES: dict[str, tuple[float, float]] = {
    "pb_shift_bar":           (-35.0, 35.0),
    "rs_pb_mult":             (0.7, 1.4),
    "bo_undersat_slope_mult": (0.5, 2.0),
    "oil_density_mult":       (0.92, 1.08),
    "bg_mult":                (0.75, 1.30),
}

# Combined 8-D LHS space (3 schedule chunks + 5 PVT levers). Used by the
# `--combined` pilot. Schedule range kept moderate so 10 sims cover a wider
# swath of (schedule × PVT) without devolving into purely extreme combos.
COMBINED_LEVER_RANGES: dict[str, tuple[float, float]] = {
    "inj_mult_early":         (0.3, 1.7),
    "inj_mult_mid":           (0.3, 1.7),
    "inj_mult_late":          (0.3, 1.7),
    "pb_shift_bar":           (-35.0, 35.0),
    "rs_pb_mult":             (0.7, 1.4),
    "bo_undersat_slope_mult": (0.5, 2.0),
    "oil_density_mult":       (0.92, 1.08),
    "bg_mult":                (0.75, 1.30),
}

# Regimes for the three-chunk WCONINJE multiplier. pb_shift is sampled
# uniformly across the SAME range in every regime, so schedule and PVT are
# decorrelated dimensions of the LHS.
REGIME_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "realistic": {
        "inj_mult_early": (0.7, 1.3),
        "inj_mult_mid":   (0.7, 1.3),
        "inj_mult_late":  (0.7, 1.3),
        "pb_shift_bar":   PB_SHIFT_RANGE,
    },
    "low_inj": {
        "inj_mult_early": (0.01, 0.4),
        "inj_mult_mid":   (0.01, 0.4),
        "inj_mult_late":  (0.01, 0.4),
        "pb_shift_bar":   PB_SHIFT_RANGE,
    },
    "high_inj": {
        "inj_mult_early": (1.7, 3.0),
        "inj_mult_mid":   (1.7, 3.0),
        "inj_mult_late":  (1.7, 3.0),
        "pb_shift_bar":   PB_SHIFT_RANGE,
    },
}
REGIME_SHARES: dict[str, float] = {"realistic": 0.60, "low_inj": 0.20, "high_inj": 0.20}


def _piecewise_factor(e: float, m: float, l: float) -> Callable[[int, int], float]:
    """Return factor_fn(block_idx, total) that maps each block to e / m / l
    according to which third of the schedule it falls in."""
    def factor(idx: int, total: int) -> float:
        frac = idx / max(1, total - 1)
        if frac < 1.0 / 3.0:
            return e
        if frac < 2.0 / 3.0:
            return m
        return l
    return factor


def build_variants(n: int, seed: int = 42) -> list[Variant]:
    """Stratified LHS over (inj_mult_early, inj_mult_mid, inj_mult_late).
    Allocates `n` sims across the three regimes per REGIME_SHARES."""
    from sampling import sample_lhs  # local import keeps top of file lean

    # Allocate per-regime counts. Use rounding then top up the realistic
    # bucket with any leftover so the totals match `n`.
    counts = {k: max(1, round(n * REGIME_SHARES[k])) for k in REGIME_RANGES}
    delta = n - sum(counts.values())
    counts["realistic"] += delta

    variants: list[Variant] = []
    sim_id = 0
    for ri, (regime, ranges) in enumerate(REGIME_RANGES.items()):
        samples = sample_lhs(counts[regime], ranges, seed=seed + ri)
        for p in samples:
            sim_id += 1
            e  = float(p["inj_mult_early"])
            m  = float(p["inj_mult_mid"])
            l  = float(p["inj_mult_late"])
            pb = float(p["pb_shift_bar"])
            variants.append(Variant(
                name=f"sim{sim_id:02d}_{regime}",
                sim_id=sim_id,
                description=f"{regime}: inj ×{e:.2f}/{m:.2f}/{l:.2f}, pb_shift={pb:+.1f} bar",
                factor=_piecewise_factor(e, m, l),
                category=regime,
                params={"inj_mult_early": e, "inj_mult_mid": m,
                        "inj_mult_late":  l, "pb_shift_bar":  pb},
            ))
    return variants


def build_combined_variants(n: int, seed: int = 42) -> list[Variant]:
    """LHS over 8 dimensions: 3 schedule chunks + 5 PVT levers. Each sim has
    a distinct schedule perturbation AND a distinct PVT — both the
    operational and PVT feature spaces are exercised in training.

    Category is always "combined"."""
    from sampling import sample_lhs

    samples = sample_lhs(n, COMBINED_LEVER_RANGES, seed=seed)
    variants: list[Variant] = []
    for sim_id, p in enumerate(samples, start=1):
        e  = float(p["inj_mult_early"])
        m  = float(p["inj_mult_mid"])
        l  = float(p["inj_mult_late"])
        pb = float(p["pb_shift_bar"])
        rs = float(p["rs_pb_mult"])
        bo = float(p["bo_undersat_slope_mult"])
        od = float(p["oil_density_mult"])
        bg = float(p["bg_mult"])
        variants.append(Variant(
            name=f"sim{sim_id:02d}_combined",
            sim_id=sim_id,
            description=(
                f"combined: inj×{e:.2f}/{m:.2f}/{l:.2f}, "
                f"pb={pb:+.1f}, rs×{rs:.2f}, bo_sl×{bo:.2f}, "
                f"od×{od:.3f}, bg×{bg:.2f}"
            ),
            factor=_piecewise_factor(e, m, l),
            category="combined",
            params={
                "inj_mult_early": e, "inj_mult_mid": m, "inj_mult_late": l,
                "pb_shift_bar":    pb,
                "rs_pb_mult":      rs,
                "bo_undersat_slope_mult": bo,
                "oil_density_mult": od,
                "bg_mult":         bg,
            },
        ))
    return variants


def build_pvt_only_variants(n: int, seed: int = 42) -> list[Variant]:
    """LHS over the 5 PVT levers only (`pb_shift_bar`, `rs_pb_mult`,
    `bo_undersat_slope_mult`, `oil_density_mult`, `bg_mult`). Schedule
    multipliers fixed at 1.0 so the PVT effect is isolated.

    Variant category is always "pvt_pilot"; factor returns 1.0 for every
    schedule block."""
    from sampling import sample_lhs

    samples = sample_lhs(n, PVT_LEVER_RANGES, seed=seed)
    variants: list[Variant] = []
    for sim_id, p in enumerate(samples, start=1):
        pb = float(p["pb_shift_bar"])
        rs = float(p["rs_pb_mult"])
        bo = float(p["bo_undersat_slope_mult"])
        od = float(p["oil_density_mult"])
        bg = float(p["bg_mult"])
        variants.append(Variant(
            name=f"sim{sim_id:02d}_pvt",
            sim_id=sim_id,
            description=(
                f"pvt_pilot: pb_shift={pb:+.1f} bar, rs×{rs:.2f}, "
                f"bo_slope×{bo:.2f}, oil_dens×{od:.3f}, bg×{bg:.2f}"
            ),
            factor=lambda i, n: 1.0,    # no schedule perturbation
            category="pvt_pilot",
            params={
                "inj_mult_early": 1.0, "inj_mult_mid": 1.0, "inj_mult_late": 1.0,
                "pb_shift_bar":    pb,
                "rs_pb_mult":      rs,
                "bo_undersat_slope_mult": bo,
                "oil_density_mult": od,
                "bg_mult":         bg,
            },
        ))
    return variants


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

    # 1. Render the deck (applies geometry/PVT levers via NORNE_CONFIG.render_deck).
    pb_shift = variant.params.get("pb_shift_bar", 0.0)
    rs_mult  = variant.params.get("rs_pb_mult", 1.0)
    bo_slope = variant.params.get("bo_undersat_slope_mult", 1.0)
    oil_dens = variant.params.get("oil_density_mult", 1.0)
    bg_mult  = variant.params.get("bg_mult", 1.0)
    render_params = {
        "k_mult": 1.0, "phi_mult": 1.0, "p_init_shift_bar": 0.0,
        "pb_shift_bar":           pb_shift,
        "rs_pb_mult":             rs_mult,
        "bo_undersat_slope_mult": bo_slope,
        "oil_density_mult":       oil_dens,
        "bg_mult":                bg_mult,
    }
    for rel_path, text in NORNE_CONFIG.render_deck(render_params).items():
        (sim_dir / rel_path).write_bytes(text.encode("latin-1"))
    pvt_changes = (pb_shift != 0.0 or rs_mult != 1.0 or
                   bo_slope != 1.0 or oil_dens != 1.0 or bg_mult != 1.0)
    if pvt_changes:
        print(f"  applied PVT: pb_shift={pb_shift:+.2f} bar, rs×{rs_mult:.2f}, "
              f"bo_slope×{bo_slope:.2f}, oil_dens×{oil_dens:.3f}, bg×{bg_mult:.2f}")

    # 2. Patch WCONINJE rate records in the schedule include.
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
    # Note: extract_features uses params["p_init_shift_bar"] for the
    # METRIC-branch Pb_psi computation (legacy naming — semantically it is
    # the bubble-point shift). We pass pb_shift_bar through this slot so
    # the dataset's Presion_Burbuja_psi column matches the actual shifted
    # bubble point of this sim.
    params = {
        "phi_mult": 1.0, "k_mult": 1.0,
        "p_init_shift_bar": pb_shift,
    }
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


CATEGORY_COLORS = {"realistic": "0.45", "low_inj": "C3", "high_inj": "C0"}


def plot(results: list[dict], variants: list[Variant], n: int) -> Path:
    var_by_name = {v.name: v for v in variants}
    fig, ax = plt.subplots(figsize=(11.5, 6))
    seen_legend: set[str] = set()
    for r in results:
        if not r["ok"]:
            continue
        v = var_by_name[r["variant"]]
        df = r["df_quick"]
        label = v.category if v.category not in seen_legend else None
        seen_legend.add(v.category)
        ax.plot(df["tiempo_dias"], df["FPR_psi"], lw=0.9,
                color=CATEGORY_COLORS.get(v.category, "k"),
                alpha=0.75, label=label)
    ax.set_xlabel("tiempo [días]")
    ax.set_ylabel("Field Pressure (FPR) [psi]")
    ax.set_title(f"Norne — {n} schedule variants (WCONINJE multiplier sampled by LHS)")
    ax.legend(loc="best", fontsize=9, title="regime")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / f"norne_variants_pvt_n{n}_fpr.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25,
                        help="Total number of sims to generate (default 25). "
                             "Allocated across regimes per REGIME_SHARES.")
    parser.add_argument("--seed", type=int, default=42,
                        help="LHS seed for reproducibility.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel OPM Flow processes. Each docker "
                             "container uses ~3-4 GB. Default Docker Desktop "
                             "= 8 GB → ≥3 workers may OOM-kill containers.")
    parser.add_argument("--pvt-only", action="store_true",
                        help="Pilot mode: LHS over the 5 PVT levers only, "
                             "schedule fixed at 1.0. Output filenames use "
                             "'multipvt'.")
    parser.add_argument("--combined", action="store_true",
                        help="Combined pilot mode: 8-D LHS over 3 schedule "
                             "chunks + 5 PVT levers. Output filenames use "
                             "'sched_multipvt'.")
    args = parser.parse_args()
    n_workers = max(1, args.workers)

    if args.combined and args.pvt_only:
        raise SystemExit("--combined and --pvt-only are mutually exclusive")
    if args.combined:
        variants = build_combined_variants(args.n, seed=args.seed)
    elif args.pvt_only:
        variants = build_pvt_only_variants(args.n, seed=args.seed)
    else:
        variants = build_variants(args.n, seed=args.seed)
    by_cat: dict[str, int] = {}
    for v in variants:
        by_cat[v.category] = by_cat.get(v.category, 0) + 1
    print(f"Running {len(variants)} Norne variants with {n_workers} worker(s).")
    print("Regime breakdown: " + ", ".join(f"{k}={v}" for k, v in by_cat.items()))

    results: list[dict] = []
    if n_workers == 1:
        for v in variants:
            results.append(run_variant(v))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(run_variant, v): v for v in variants}
            for fut in as_completed(futures):
                results.append(fut.result())
        order = {v.name: i for i, v in enumerate(variants)}
        results.sort(key=lambda r: order[r["variant"]])

    # Summary table.
    print("\n=== Summary ===")
    for r in results:
        if r["ok"]:
            df = r["df_quick"]
            print(f"  {r['variant']:<28}  FPR {df.FPR_psi.min():.0f}–{df.FPR_psi.max():.0f} psi  "
                  f"({df.FPR_psi.max() - df.FPR_psi.min():.0f} psi range, "
                  f"{r['elapsed_s']:.0f}s)")
        else:
            print(f"  {r['variant']:<28}  FAILED")

    # Concat sims that converged into the unified dataset CSV.
    dataset_dir = ROOT / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    dfs = [r["df"] for r in results if r["ok"]]
    if args.combined:
        tag = "sched_multipvt"
    elif args.pvt_only:
        tag = "multipvt"
    else:
        tag = "schedule_pvt"
    out_csv = dataset_dir / f"dataset_norne_{tag}_n{args.n}.csv"
    if dfs:
        full = pd.concat(dfs, ignore_index=True)
        full.to_csv(out_csv, index=False)
        print(f"\nWrote dataset → {out_csv} "
              f"({len(full)} rows, {full['sim_id'].nunique()} sims, "
              f"FPR {full['Presion_Reservorio_psi'].min():.0f}–"
              f"{full['Presion_Reservorio_psi'].max():.0f} psi)")

    # Per-sim params + outcome log.
    var_by_name = {v.name: v for v in variants}
    log_rows = []
    for r in results:
        v = var_by_name[r["variant"]]
        row = {
            "sim_id": v.sim_id, "name": v.name, "category": v.category,
            "inj_mult_early": v.params["inj_mult_early"],
            "inj_mult_mid":   v.params["inj_mult_mid"],
            "inj_mult_late":  v.params["inj_mult_late"],
            "pb_shift_bar":   v.params.get("pb_shift_bar", 0.0),
            "rs_pb_mult":     v.params.get("rs_pb_mult", 1.0),
            "bo_undersat_slope_mult": v.params.get("bo_undersat_slope_mult", 1.0),
            "oil_density_mult": v.params.get("oil_density_mult", 1.0),
            "bg_mult":        v.params.get("bg_mult", 1.0),
            "ok": r["ok"], "elapsed_s": round(r["elapsed_s"], 1),
        }
        if r["ok"]:
            fpr = r["df_quick"]["FPR_psi"].values
            row.update({"fpr_min": float(fpr.min()), "fpr_max": float(fpr.max()),
                        "fpr_range": float(fpr.max() - fpr.min())})
        else:
            row.update({"fpr_min": None, "fpr_max": None, "fpr_range": None})
        log_rows.append(row)
    log_path = dataset_dir / f"runs_log_norne_{tag}_n{args.n}.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Wrote runs_log → {log_path}")

    out = plot(results, variants, args.n)
    print(f"Wrote plot → {out}")


if __name__ == "__main__":
    main()
