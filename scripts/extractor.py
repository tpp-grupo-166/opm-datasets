"""Generic SUMMARY-to-DataFrame extractor.

Takes a `DeckConfig` and produces an 18-column row per timestep matching
the project schema. METRIC simulations get unit-converted to FIELD on
the fly so all reservoirs land in the same units.

PVT-derived columns (Bo, Bg, Rs) are interpolated from the SPE9 PVT
tables in `pvt_tables.py`. They are flagged as leakage in the model and
are not used for prediction; the cross-deck approximation is acceptable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from resdata.summary import Summary

from decks.base import DeckConfig
from pvt_aggregate import aggregate_pvt
from pvt_tables import bg_from_pressure, bo_from_pressure, rs_from_pressure


SCHEMA_COLUMNS = [
    "reservoir_id",
    "tiempo_dias",
    "Porosidad",
    "Permeabilidad_mD",
    "Espesor_Neto_m",
    "Area",
    "Presion_Burbuja_psi",
    "Bo_rb_stb",
    "Bg_rb_scf",
    "Rs_scf_stb",
    "Caudal_Prod_Petroleo_bbl",
    "Caudal_Prod_Gas_Mpc",
    "Caudal_Iny_Agua_bbl",
    "Prod_Acumulada_Petroleo",
    "Prod_Acumulada_Gas",
    "Prod_Acumulada_Agua",
    "Iny_Acumulada_Agua",
    "Presion_Reservorio_psi",
]

# METRIC -> FIELD conversions
BAR_TO_PSI = 14.5037738
SM3_TO_STB = 6.28981077
SM3_TO_SCF = 35.3146667
SM3_TO_MSCF = SM3_TO_SCF / 1000.0


def extract_features(
    config: DeckConfig,
    summary_basename: Path | str,
    sim_id: int,
    params: dict,
) -> pd.DataFrame:
    summary_basename = Path(summary_basename)
    sm = Summary(str(summary_basename))

    tiempo_dias = sm.numpy_vector("TIME")
    fpr = sm.numpy_vector("FPR")
    fopr = sm.numpy_vector("FOPR")
    fgpr = sm.numpy_vector("FGPR")
    fwir = sm.numpy_vector("FWIR")
    fopt = sm.numpy_vector("FOPT")
    fgpt = sm.numpy_vector("FGPT")
    fwpt = sm.numpy_vector("FWPT")
    fwit = sm.numpy_vector("FWIT")

    if config.unit_system == "METRIC":
        fpr_psi = fpr * BAR_TO_PSI
        fopr_field = fopr * SM3_TO_STB
        fgpr_field = fgpr * SM3_TO_MSCF
        fwir_field = fwir * SM3_TO_STB
        fopt_field = fopt * SM3_TO_STB
        fgpt_field = fgpt * SM3_TO_SCF
        fwpt_field = fwpt * SM3_TO_STB
        fwit_field = fwit * SM3_TO_STB
        pb_psi = (config.baseline_pb + params.get("p_init_shift_bar", 0.0)) * BAR_TO_PSI
        pb_shift_for_pvt = 0.0  # METRIC decks: PVT is approximate cross-deck
    else:
        fpr_psi = fpr
        fopr_field = fopr
        fgpr_field = fgpr  # FGPR already in MSCF/DAY (FIELD convention)
        fwir_field = fwir
        fopt_field = fopt
        # SPE9 stores FGPT in MSCF; the schema column is in SCF.
        fgpt_field = fgpt * 1000.0
        fwpt_field = fwpt
        fwit_field = fwit
        pb_psi = config.baseline_pb + params.get("pb_shift", 0.0)
        pb_shift_for_pvt = params.get("pb_shift", 0.0)

    if config.pvt_tables is not None:
        sim_days, bo_cell_avg, bg_cell_avg, rs_cell_avg = aggregate_pvt(
            summary_basename, config.pvt_tables, config.unit_system
        )
        # Align UNRST report-step series to SUMMARY tiempo_dias via linear interp
        bo = np.interp(tiempo_dias, sim_days, bo_cell_avg)
        bg = np.interp(tiempo_dias, sim_days, bg_cell_avg)
        rs = np.interp(tiempo_dias, sim_days, rs_cell_avg)
    else:
        rs = rs_from_pressure(fpr_psi, pb_shift_for_pvt)
        bo = bo_from_pressure(fpr_psi, pb_shift_for_pvt)
        bg = bg_from_pressure(fpr_psi)

    porosidad = config.static_features["baseline_porosity"] * params["phi_mult"]
    permeabilidad = config.static_features["baseline_perm_md"] * params["k_mult"]
    espesor = config.static_features["espesor_neto_m"]
    area = config.static_features["area_m2"]

    n = len(fpr_psi)
    return pd.DataFrame(
        {
            "sim_id": np.full(n, sim_id, dtype=int),
            "reservoir_id": config.name,
            "tiempo_dias": tiempo_dias,
            "Porosidad": porosidad,
            "Permeabilidad_mD": permeabilidad,
            "Espesor_Neto_m": espesor,
            "Area": area,
            "Presion_Burbuja_psi": pb_psi,
            "Bo_rb_stb": bo,
            "Bg_rb_scf": bg,
            "Rs_scf_stb": rs,
            "Caudal_Prod_Petroleo_bbl": fopr_field,
            "Caudal_Prod_Gas_Mpc": fgpr_field,
            "Caudal_Iny_Agua_bbl": fwir_field,
            "Prod_Acumulada_Petroleo": fopt_field,
            "Prod_Acumulada_Gas": fgpt_field,
            "Prod_Acumulada_Agua": fwpt_field,
            "Iny_Acumulada_Agua": fwit_field,
            "Presion_Reservorio_psi": fpr_psi,
        }
    )
