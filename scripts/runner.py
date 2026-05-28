"""Generic per-simulation worker.

Operates on a `DeckConfig`: copies the deck dir to a per-sim folder,
overwrites the files returned by `config.render_deck(params)`, runs OPM
Flow via Docker, parses the SUMMARY into a DataFrame, and cleans up.

Replaces the model-specific runner_*.py files. Backwards-compatible
naming convention: per-sim dirs are `runs/{config.name}_sim_{NNNN}`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from decks.base import DeckConfig
from extractor import extract_features

DOCKER_IMAGE = "openporousmedia/opmreleases:latest"


def run_simulation(
    config: DeckConfig,
    sim_id: int,
    params: dict,
    runs_dir: Path,
    keep_outputs: bool = False,
) -> dict:
    # Per-cell PVT extraction needs the .UNRST and .INIT alive during
    # extract_features; the finally block still cleans up afterwards.
    sim_dir = runs_dir / f"{config.name}_sim_{sim_id:04d}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    started = time.perf_counter()
    try:
        shutil.copytree(config.deck_dir, sim_dir)

        for rel_path, text in config.render_deck(params).items():
            (sim_dir / rel_path).write_bytes(text.encode("latin-1"))

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{sim_dir.resolve()}:/shared_host",
            DOCKER_IMAGE,
            "flow",
            "--output-dir=/shared_host",
            f"/shared_host/{config.main_deck_filename}",
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.flow_timeout_s,
        )
        if proc.returncode != 0:
            _persist_failure_log(sim_dir, proc.stdout, proc.stderr)
            return _result(sim_id, False, f"docker exit {proc.returncode}",
                           started, None, params)

        summary_base = sim_dir / config.summary_basename
        df = extract_features(config, summary_base, sim_id, params)
        return _result(sim_id, True, None, started, df, params)

    except subprocess.TimeoutExpired as exc:
        return _result(sim_id, False, f"timeout after {exc.timeout:.0f}s",
                       started, None, params)
    except Exception as exc:
        return _result(sim_id, False, f"{type(exc).__name__}: {exc}",
                       started, None, params)
    finally:
        if not keep_outputs and sim_dir.exists():
            shutil.rmtree(sim_dir, ignore_errors=True)


def _result(sim_id, ok, error, started, df, params) -> dict:
    return {
        "sim_id": sim_id,
        "ok": ok,
        "error": error,
        "runtime_s": time.perf_counter() - started,
        "df": df,
        "params": params,
    }


def _persist_failure_log(sim_dir: Path, stdout: str, stderr: str) -> None:
    log_path = sim_dir.parent / f"{sim_dir.name}.failed.log"
    tail_lines = 300
    body = (
        "=== STDOUT (tail) ===\n"
        + "\n".join(stdout.splitlines()[-tail_lines:])
        + "\n\n=== STDERR (tail) ===\n"
        + "\n".join(stderr.splitlines()[-tail_lines:])
        + "\n"
    )
    log_path.write_text(body)
