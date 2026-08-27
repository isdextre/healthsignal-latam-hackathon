"""Carga el contrato provisional (submission_kit/signals.csv + evidence.csv).

Este loader es la ÚNICA pieza que debe cambiar cuando el pipeline de
detección/priorización de tu compañero entregue `candidate_signals_df`
real: el resto del módulo de explicación (templates, Gemini, guardrails,
API) consume siempre el mismo shape intermedio devuelto aquí.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNALS_CSV = REPO_ROOT / "submission_kit" / "signals.csv"
EVIDENCE_CSV = REPO_ROOT / "submission_kit" / "evidence.csv"


def load_raw_signals(signals_csv: Path = SIGNALS_CSV, evidence_csv: Path = EVIDENCE_CSV) -> list[dict]:
    """Devuelve una lista de dicts: una señal + su evidencia asociada.

    Shape de cada elemento:
        {
          "signal_id": str, "patient_id": str, "decision_datetime": str,
          "risk_score": float, "priority_level": str, "confidence_score": float,
          "evidence_start": str, "evidence_end": str,
          "explanation_raw": str, "model_version": str,
          "evidence": [ {source_file, record_id, variable_code,
                          event_datetime, available_datetime,
                          evidence_role, contribution}, ... ]
        }
    """
    signals_df = pd.read_csv(signals_csv)
    evidence_df = pd.read_csv(evidence_csv)

    out = []
    for _, row in signals_df.iterrows():
        sig_evidence = evidence_df[evidence_df["signal_id"] == row["signal_id"]]
        evidence_records = [
            {
                "source_file": e["source_file"],
                "record_id": e["record_id"],
                "variable_code": None if pd.isna(e["variable_code"]) else e["variable_code"],
                "event_datetime": e["event_datetime"],
                "available_datetime": e["available_datetime"],
                "evidence_role": e["evidence_role"],
                "contribution": float(e["contribution"]),
            }
            for _, e in sig_evidence.iterrows()
        ]
        out.append(
            {
                "signal_id": row["signal_id"],
                "patient_id": row["patient_id"],
                "decision_datetime": row["decision_datetime"],
                "risk_score": float(row["risk_score"]),
                "priority_level": row["priority_level"],
                "confidence_score": None if pd.isna(row["confidence_score"]) else float(row["confidence_score"]),
                "evidence_start": row["evidence_start"],
                "evidence_end": row["evidence_end"],
                "explanation_raw": row["explanation"],
                "model_version": row["model_version"],
                "evidence": evidence_records,
            }
        )
    return out
