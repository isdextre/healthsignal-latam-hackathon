"""Guardrails: verifica que la reformulación de Gemini no invente cifras
ni haga afirmaciones fuera del alcance de RISA (diagnóstico, prescripción).

Esta es la implementación concreta del requisito de la §11 del contexto
oficial de RISA: "la explicación generada debe mantenerse claramente
separada de la evidencia extraída de los datos". Si el output no pasa
estas verificaciones, el pipeline nunca lo sirve — cae automáticamente a
`explanation_raw` (la versión determinista).
"""

from __future__ import annotations

import re

from .schemas import NarrativeLLMOutput

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Vocabulario que excede el alcance de RISA (§10: "apoyo a la decisión, no
# diagnóstico ni prescripción autónoma"). Lista corta e intencionalmente
# conservadora — se amplía si el jurado/pruebas detectan fugas.
_FORBIDDEN_PATTERNS = [
    r"\bdiagnostic\w*\b",
    r"\brecet\w*\b",
    r"\bprescrib\w*\b",
    r"\badminist(re|rar)\s+(medicaci[oó]n|f[aá]rmaco|dosis)",
    r"\busted (tiene|padece|sufre)\b",
    r"\bel paciente (tiene|padece|sufre) de\b",
    r"\bconfirmo que\b",
]
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)


def _extract_numbers(text: str) -> set[str]:
    return {m.replace(",", ".").rstrip("0").rstrip(".") or "0" for m in _NUMBER_RE.findall(text)}


def _all_roundings(value: float) -> set[str]:
    """Genera las representaciones plausibles de un número (Gemini puede
    citarlo con distinta cantidad de decimales, ej. 0.853 vs 0.85 vs 85%)."""
    variants = {str(value)}
    for decimals in (0, 1, 2, 3):
        variants.add(f"{value:.{decimals}f}".rstrip("0").rstrip("."))
    return {v if v else "0" for v in variants}


def _allowed_numbers(context: dict) -> set[str]:
    allowed: set[str] = set()
    allowed |= _extract_numbers(context["explanation_raw"])
    allowed |= _extract_numbers(context["pattern_explain_hint"])
    allowed |= _extract_numbers(context["priority_reason_hint"])
    allowed |= _all_roundings(context["risk_score"])
    if context["confidence_score"] is not None:
        allowed |= _all_roundings(context["confidence_score"])
        allowed |= _all_roundings(round(context["confidence_score"] * 100))
    for f in context["factors"]:
        allowed |= _all_roundings(f["contribution"])
        allowed |= _extract_numbers(f["variable_name"])
    return allowed


def _all_text(output: NarrativeLLMOutput) -> str:
    parts = [output.headline, output.what_happened, output.pattern_label, output.pattern_explain, output.priority_reason]
    parts += [f.explain for f in output.factor_explanations]
    return "\n".join(parts)


def validate(output: NarrativeLLMOutput, context: dict) -> tuple[bool, list[str]]:
    """Devuelve (ok, motivos_de_rechazo)."""
    reasons: list[str] = []
    text = _all_text(output)

    if _FORBIDDEN_RE.search(text):
        reasons.append("lenguaje de diagnóstico/prescripción fuera del alcance de RISA")

    allowed_numbers = _allowed_numbers(context)
    found_numbers = _extract_numbers(text)
    unexpected = found_numbers - allowed_numbers
    if unexpected:
        reasons.append(f"números no presentes en la evidencia de entrada: {sorted(unexpected)}")

    expected_codes = {f["variable_code"] for f in context["factors"]}
    got_codes = {f.variable_code for f in output.factor_explanations}
    if got_codes != expected_codes:
        reasons.append(f"factor_explanations no coincide con las variables de entrada (esperado={expected_codes}, recibido={got_codes})")

    return (len(reasons) == 0, reasons)
