"""Cliente Gemini para la Etapa 6+ ("Redacción con LLM trazable").

Reglas de este módulo, no negociables:
- Nunca inventa cifras: recibe un `context` ya calculado por
  `templates.py` y solo puede reformular texto.
- Nunca ve `evidence.csv` en crudo (record_id, source_file) — solo los
  campos agregados (variable_code, rol, contribución) necesarios para
  redactar. Así la evidencia y la explicación generada quedan separadas
  por diseño, no solo por convención (§11 del contexto oficial RISA).
- Si Gemini falla, tarda demasiado o el guardrail rechaza su respuesta,
  se propaga la excepción para que el llamador (pipeline.py) haga el
  fallback a `explanation_raw` — este módulo no decide el fallback.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from .schemas import NarrativeLLMOutput

_SYSTEM_INSTRUCTION = """\
Eres un asistente de redacción clínica para HealthSignal LATAM, una herramienta de APOYO A LA DECISIÓN \
(no de diagnóstico ni prescripción autónoma) sobre la red ficticia RISA.

Tu única tarea es REFORMULAR en español latinoamericano, con lenguaje natural y profesional, una explicación \
técnica que un motor de reglas ya generó. Reglas estrictas:
1. No inventes, cambies ni redondees de forma distinta ninguna cifra, porcentaje, unidad o nombre de variable \
que no esté explícitamente presente en el texto o los datos que se te entregan.
2. No emitas diagnósticos, no sugieras tratamientos ni prescribas medicación. Tu texto es apoyo informativo, \
nunca una conclusión clínica autónoma.
3. Devuelve una explicación por cada variable en "factors", identificada por su variable_code exacto, ni más \
ni menos de las que se te dan.
4. Si el texto de entrada no da un dato, no lo completes ni lo asumas: describe solo lo que sí está dado.
5. Tono: profesional, claro, empático, apto para un profesional de salud ocupado. Evita jerga innecesaria.
"""


def _build_prompt(context: dict) -> str:
    factors_lines = "\n".join(
        f"- {f['variable_code']} ({f['variable_name']}): rol={f['role']}, contribución={f['contribution']}"
        for f in context["factors"]
    )
    confidence_text = context["confidence_score"] if context["confidence_score"] is not None else "no disponible"
    return f"""\
Explicación técnica generada por el motor de reglas (fuente de verdad, no la copies literal, reformúlala):
\"\"\"{context['explanation_raw']}\"\"\"

Prioridad asignada: {context['priority_level']}
Puntaje de riesgo: {context['risk_score']}
Confianza: {confidence_text}

Patrón detectado (nombre sugerido, puedes pulir la redacción sin cambiar el significado): {context['pattern_label_hint']}
Explicación del patrón (base determinista): {context['pattern_explain_hint']}
Razón de prioridad (base determinista): {context['priority_reason_hint']}

Variables de evidencia (variable_code, nombre, rol, contribución):
{factors_lines}

Genera la reformulación en el formato estructurado solicitado.
"""


def generate_narrative(context: dict, model: str | None = None, timeout_s: float = 12.0) -> NarrativeLLMOutput:
    api_key = os.environ["GEMINI_API_KEY"]
    model_name = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=_build_prompt(context),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=NarrativeLLMOutput,
            temperature=0.3,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        ),
    )

    if response.parsed is None:
        raise ValueError(f"Gemini no devolvió una respuesta parseable: {response.text!r}")
    return response.parsed
