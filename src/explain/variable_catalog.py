"""Catálogo estático de las 14 variables monitoreadas en RISA.

Fuente: data/processed/diccionario_datos_features_df.md (sección 2).
Se usa para traducir un `variable_code` (ej. "HR") en nombre legible,
unidad y rango de referencia sin depender de un LLM ni de datos externos.
"""

VARIABLE_CATALOG: dict[str, dict[str, str]] = {
    "HR": {"name": "Frecuencia cardiaca", "unit": "lpm", "ref": "60–100 lpm"},
    "RR": {"name": "Frecuencia respiratoria", "unit": "rpm", "ref": "12–20 rpm"},
    "SBP": {"name": "Presión arterial sistólica", "unit": "mmHg", "ref": "90–120 mmHg"},
    "DBP": {"name": "Presión arterial diastólica", "unit": "mmHg", "ref": "60–80 mmHg"},
    "SpO2": {"name": "Saturación de oxígeno", "unit": "%", "ref": "95–100 %"},
    "TEMP": {"name": "Temperatura corporal", "unit": "°C", "ref": "36.1–37.2 °C"},
    "WEARABLE_HR": {"name": "Frecuencia cardiaca (wearable)", "unit": "lpm", "ref": "60–100 lpm"},
    "STEPS": {"name": "Pasos registrados", "unit": "pasos", "ref": "según rutina habitual"},
    "ACTIVITY_LEVEL": {"name": "Nivel de actividad física", "unit": "categoría", "ref": "según rutina habitual"},
    "SIGNAL_QUALITY_INDEX": {"name": "Índice de calidad de señal", "unit": "proporción", "ref": "0.4–1.0"},
    "LAB_A": {"name": "Marcador de laboratorio A", "unit": "uA", "ref": "0–50 uA"},
    "LAB_B": {"name": "Marcador de laboratorio B", "unit": "uB", "ref": "0–300 uB"},
    "LAB_C": {"name": "Marcador de laboratorio C", "unit": "uC", "ref": "0–10 uC"},
    "LAB_D": {"name": "Marcador de laboratorio D", "unit": "uD", "ref": "0–10 uD"},
}

ROLE_LABELS: dict[str, str] = {
    "PRIMARY": "Evidencia principal",
    "SUPPORTING": "Evidencia de apoyo",
    "CONTEXT": "Información de contexto",
    "QUALITY": "Señal de calidad de datos",
}

ROLE_ORDER = {"PRIMARY": 0, "SUPPORTING": 1, "CONTEXT": 2, "QUALITY": 3}


def describe_variable(code: str) -> dict[str, str]:
    return VARIABLE_CATALOG.get(code, {"name": code, "unit": "", "ref": "no documentado"})
