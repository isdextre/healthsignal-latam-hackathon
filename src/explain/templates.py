"""Plantillas deterministas — NINGUNA función de este archivo llama a un LLM.

Calcula todo lo que se puede derivar directamente de signals.csv +
evidence.csv: factores por variable, evidencia trazable, nota de calidad,
regla general y una primera versión (`explanation_raw`) de cada campo
narrativo. El LLM (Gemini) solo reformula estos textos — nunca los genera
desde cero — y el guardrail verifica que no haya agregado cifras nuevas.

Nota de alcance: el contrato provisional (submission_kit/signals.csv +
evidence.csv) no incluye datos maestros del paciente (edad, programa) ni
el valor/dirección numérica de cada variable — solo su `variable_code` y
`contribution`. Esos campos se dejan en `None`/neutros aquí, listos para
poblarse en cuanto el pipeline de detección del equipo entregue
`candidate_signals_df` con esa información.
"""

from __future__ import annotations

from .variable_catalog import ROLE_LABELS, ROLE_ORDER, describe_variable


def _contribution_label(contribution: float) -> str:
    if contribution >= 0.75:
        return "Influencia muy alta"
    if contribution >= 0.5:
        return "Influencia alta"
    if contribution >= 0.25:
        return "Influencia moderada"
    return "Influencia baja"


def _build_factors(evidence: list[dict]) -> list[dict]:
    """Agrupa la evidencia por variable_code -> un factor por variable."""
    by_var: dict[str, list[dict]] = {}
    for e in evidence:
        code = e["variable_code"]
        if not code:
            continue
        by_var.setdefault(code, []).append(e)

    factors = []
    for code, rows in by_var.items():
        best_role = min(rows, key=lambda r: ROLE_ORDER.get(r["evidence_role"], 99))["evidence_role"]
        max_contribution = max(r["contribution"] for r in rows)
        meta = describe_variable(code)
        factors.append(
            {
                "code": code,
                "name": meta["name"],
                "unit": meta["unit"],
                "ref": meta["ref"],
                "contribution": round(max_contribution, 3),
                "contributionLabel": _contribution_label(max_contribution),
                "role": best_role,
                "roleLabel": ROLE_LABELS.get(best_role, best_role),
                "direction": None,
                "lastValue": None,
            }
        )
    factors.sort(key=lambda f: f["contribution"], reverse=True)
    return factors


def _classify_pattern(factors: list[dict], evidence: list[dict]) -> tuple[str, str]:
    primary_vars = {f["code"] for f in factors if f["role"] == "PRIMARY"}
    has_quality = any(e["evidence_role"] == "QUALITY" for e in evidence)

    if has_quality and len(primary_vars) <= 1:
        label = "Anomalía relacionada con calidad del dato"
        explain = (
            "La evidencia incluye una señal de calidad de dispositivo, por lo que la alerta "
            "se prioriza como posible problema de calidad antes que como riesgo clínico directo."
        )
    elif len(primary_vars) >= 2:
        label = "Cambios simultáneos en varias variables"
        explain = (
            f"Se identificaron {len(primary_vars)} variables con evidencia principal en la misma "
            "ventana temporal, lo que sugiere un patrón conjunto más que un valor aislado."
        )
    elif len(primary_vars) == 1:
        label = "Variable aislada fuera de rango"
        explain = "Una única variable concentra la evidencia principal de esta señal en la ventana analizada."
    else:
        label = "Patrón no clasificado"
        explain = "La evidencia disponible no permitió clasificar un patrón específico con el contrato de datos actual."
    return label, explain


def _describe_model_version(model_version: str) -> str:
    parts = []
    if "rules" in model_version:
        parts.append("un motor de reglas explícitas (umbrales y persistencia)")
    if "iforest" in model_version:
        parts.append("un modelo de detección de anomalías (Isolation Forest)")
    engines = ", ".join(parts) if parts else "el motor de detección de HealthSignal LATAM"
    return f"Esta señal fue generada por {engines} (versión: {model_version})."


def build_deterministic_narrative(raw_signal: dict) -> dict:
    """Construye todos los campos deterministas + el paquete de contexto para Gemini."""
    evidence = raw_signal["evidence"]
    factors = _build_factors(evidence)
    pattern_label, pattern_explain = _classify_pattern(factors, evidence)

    event_times = [e["event_datetime"] for e in evidence if e.get("event_datetime")]
    change_start = min(event_times) if event_times else None

    quality_flag = any(e["evidence_role"] == "QUALITY" for e in evidence)
    confidence_score = raw_signal["confidence_score"]
    confidence_clause = (
        f"Nivel de confianza asignado a la señal: {round(confidence_score * 100)}%. "
        if confidence_score is not None
        else "Nivel de confianza no disponible para esta señal. "
    )
    data_quality_note = confidence_clause + (
        "Se registró evidencia de calidad de señal que fue considerada en el análisis."
        if quality_flag
        else "No se registraron problemas de calidad de datos en la evidencia disponible."
    )

    increased_factors = [
        f"La variable {f['name']} intervino como {f['roleLabel'].lower()} con una contribución de {f['contribution']:.2f}."
        for f in factors
        if f["role"] in ("PRIMARY", "SUPPORTING")
    ]
    decreased_factors = [
        f"La variable {f['name']} aportó como {f['roleLabel'].lower()}, lo cual da contexto sin elevar la prioridad por sí sola."
        for f in factors
        if f["role"] in ("CONTEXT", "QUALITY")
    ]

    persistence = (
        f"La ventana de evidencia abarca de {raw_signal['evidence_start']} a {raw_signal['evidence_end']}, "
        f"con {len(evidence)} registro(s) de evidencia considerados."
    )

    confidence_fragment = (
        f"y una confianza de {round(confidence_score * 100)}%, "
        if confidence_score is not None
        else "sin confianza reportada para esta señal, "
    )
    priority_reason_raw = (
        f"Se asignó prioridad {raw_signal['priority_level']} con un puntaje de riesgo de "
        f"{raw_signal['risk_score']:.2f} {confidence_fragment}"
        f"con base en {len(factors)} variable(s) de evidencia."
    )

    return {
        "factors": factors,
        "evidence": evidence,
        "period_records": len(evidence),
        "change_start": change_start,
        "persistence": persistence,
        "pattern_label": pattern_label,
        "pattern_explain": pattern_explain,
        "data_quality_note": data_quality_note,
        "general_rule": _describe_model_version(raw_signal["model_version"]),
        "increased_factors": increased_factors,
        "decreased_factors": decreased_factors,
        "priority_reason_raw": priority_reason_raw,
        "headline_raw": raw_signal["explanation_raw"],
        "what_happened_raw": raw_signal["explanation_raw"],
    }


def build_gemini_context(raw_signal: dict, deterministic: dict) -> dict:
    """Paquete mínimo y auto-contenido que se envía a Gemini para reformular.

    Solo contiene texto y cifras que YA existen en el pipeline determinista;
    el guardrail usa este mismo paquete como fuente de verdad para validar
    que la respuesta no introduzca números nuevos.
    """
    return {
        "explanation_raw": raw_signal["explanation_raw"],
        "priority_level": raw_signal["priority_level"],
        "risk_score": raw_signal["risk_score"],
        "confidence_score": raw_signal["confidence_score"],
        "pattern_label_hint": deterministic["pattern_label"],
        "pattern_explain_hint": deterministic["pattern_explain"],
        "priority_reason_hint": deterministic["priority_reason_raw"],
        "factors": [
            {
                "variable_code": f["code"],
                "variable_name": f["name"],
                "role": f["roleLabel"],
                "contribution": f["contribution"],
            }
            for f in deterministic["factors"]
        ],
    }
