"""Orquesta: datos crudos -> plantilla determinista -> Gemini -> guardrail
-> caché -> `SignalDetail` (el shape que consume el front).

Este es el único punto de entrada que debe usar `src/api/main.py`.
"""

from __future__ import annotations

import logging

from . import cache, guardrails
from .data_loader import load_raw_signals
from .gemini_client import generate_narrative
from .schemas import EvidenceItem, SignalDetail, VariableFactor
from .templates import build_deterministic_narrative, build_gemini_context
from .variable_catalog import ROLE_LABELS

logger = logging.getLogger("healthsignal.explain")


def _assemble_variables(factors: list[dict], factor_explanations_by_code: dict[str, str]) -> list[VariableFactor]:
    variables = []
    for f in factors:
        explain = factor_explanations_by_code.get(
            f["code"],
            f"{f['roleLabel']} para {f['name']}, con una contribución registrada de {f['contribution']:.2f}.",
        )
        variables.append(
            VariableFactor(
                code=f["code"],
                name=f["name"],
                unit=f["unit"],
                ref=f["ref"],
                contribution=f["contribution"],
                contributionLabel=f["contributionLabel"],
                role=f["role"],
                roleLabel=f["roleLabel"],
                explain=explain,
                direction=None,
                lastValue=None,
            )
        )
    return variables


def _assemble_evidence(evidence: list[dict]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            sourceFile=e["source_file"],
            recordId=e["record_id"],
            variableCode=e["variable_code"],
            eventDatetime=e["event_datetime"],
            availableDatetime=e["available_datetime"],
            role=e["evidence_role"],
            roleLabel=ROLE_LABELS.get(e["evidence_role"], e["evidence_role"]),
            contribution=e["contribution"],
        )
        for e in evidence
    ]


def build_signal_detail(raw_signal: dict, *, force_regenerate: bool = False) -> SignalDetail:
    det = build_deterministic_narrative(raw_signal)
    context = build_gemini_context(raw_signal, det)

    llm_output = None if force_regenerate else cache.get(raw_signal["signal_id"], context)

    if llm_output is None:
        try:
            parsed = generate_narrative(context)
            ok, reasons = guardrails.validate(parsed, context)
            if not ok:
                logger.warning("Guardrail rechazó respuesta de Gemini para %s: %s", raw_signal["signal_id"], reasons)
                raise ValueError(f"guardrail rejected: {reasons}")
            llm_output = parsed.model_dump()
            cache.set(raw_signal["signal_id"], context, llm_output)
        except Exception:
            logger.exception("Fallo generando explicación con Gemini para %s, usando fallback determinista", raw_signal["signal_id"])
            llm_output = None

    explanation_source = "llm" if llm_output else "template_fallback"

    if llm_output:
        headline = llm_output["headline"]
        what_happened = llm_output["what_happened"]
        pattern = llm_output["pattern_label"]
        pattern_explain = llm_output["pattern_explain"]
        priority_reason = llm_output["priority_reason"]
        factor_explanations_by_code = {f["variable_code"]: f["explain"] for f in llm_output["factor_explanations"]}
    else:
        headline = det["headline_raw"]
        what_happened = det["what_happened_raw"]
        pattern = det["pattern_label"]
        pattern_explain = det["pattern_explain"]
        priority_reason = det["priority_reason_raw"]
        factor_explanations_by_code = {}

    return SignalDetail(
        id=raw_signal["signal_id"],
        patientId=raw_signal["patient_id"],
        priority=raw_signal["priority_level"],
        riskScore=raw_signal["risk_score"],
        confidence=raw_signal["confidence_score"],
        decisionDatetime=raw_signal["decision_datetime"],
        evidenceStart=raw_signal["evidence_start"],
        evidenceEnd=raw_signal["evidence_end"],
        isDemo=raw_signal["signal_id"].startswith("SIG-TEST"),
        headline=headline,
        whatHappened=what_happened,
        periodRecords=det["period_records"],
        changeStart=det["change_start"],
        persistence=det["persistence"],
        pattern=pattern,
        patternExplain=pattern_explain,
        increasedFactors=det["increased_factors"],
        decreasedFactors=det["decreased_factors"],
        dataQualityNote=det["data_quality_note"],
        generalRule=det["general_rule"],
        priorityReason=priority_reason,
        variables=_assemble_variables(det["factors"], factor_explanations_by_code),
        evidence=_assemble_evidence(det["evidence"]),
        patientAge=None,
        patientProgram=None,
        modelVersion=raw_signal["model_version"],
        explanationRaw=raw_signal["explanation_raw"],
        explanationSource=explanation_source,
    )


def build_all_signal_details() -> list[SignalDetail]:
    return [build_signal_detail(raw) for raw in load_raw_signals()]
