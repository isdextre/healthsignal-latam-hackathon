"""Modelos de datos del módulo de explicación.

Dos familias de modelos:
- `NarrativeLLMOutput`: el ÚNICO shape que Gemini puede producir (salida
  estructurada). No incluye cifras, IDs ni evidencia — solo texto natural
  que reformula lo que el pipeline determinista ya calculó.
- `SignalDetail` / `SignalSummary`: el shape que consume el front
  (mismo contrato que `SIGNALS_DATA` en docs/bosquejo_front.html), con
  campos adicionales (`explanationRaw`, `explanationSource`) para cumplir
  la trazabilidad exigida por RISA (evidencia y explicación generada
  deben poder distinguirse).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Salida estructurada exigida a Gemini ----------

class FactorExplanation(BaseModel):
    variable_code: str = Field(description="Código de variable, debe existir en la lista de factores dada como entrada")
    explain: str = Field(description="Una oración en español que reformula el rol de este factor, sin inventar cifras nuevas")


class NarrativeLLMOutput(BaseModel):
    headline: str = Field(description="2-3 oraciones: resumen ejecutivo de la señal, en lenguaje natural")
    what_happened: str = Field(description="1 oración en lenguaje cotidiano sobre qué se observó")
    pattern_label: str = Field(description="Nombre corto (3-6 palabras) del patrón observado")
    pattern_explain: str = Field(description="1-2 oraciones explicando por qué ese patrón aplica, basado solo en los factores dados")
    priority_reason: str = Field(description="1-2 oraciones justificando el nivel de prioridad asignado, citando los factores dados")
    factor_explanations: list[FactorExplanation] = Field(description="Una entrada por cada variable_code recibido en la entrada, ni más ni menos")


# ---------- Contrato de salida hacia el front ----------

class VariableFactor(BaseModel):
    code: str
    name: str
    unit: str
    ref: str
    contribution: float
    contributionLabel: str
    role: str
    roleLabel: str
    explain: str
    direction: Optional[str] = None
    lastValue: Optional[str] = None


class EvidenceItem(BaseModel):
    sourceFile: str
    recordId: str
    variableCode: Optional[str]
    eventDatetime: str
    availableDatetime: str
    role: str
    roleLabel: str
    contribution: float


class SignalSummary(BaseModel):
    id: str
    patientId: str
    priority: str
    riskScore: float
    confidence: Optional[float]
    decisionDatetime: str
    headline: str
    isDemo: bool
    patientAge: Optional[int] = None
    patientProgram: Optional[str] = None
    explanationSource: Literal["llm", "template_fallback"]


class SignalDetail(BaseModel):
    id: str
    patientId: str
    priority: str
    riskScore: float
    confidence: Optional[float]
    decisionDatetime: str
    evidenceStart: str
    evidenceEnd: str
    isDemo: bool = False

    headline: str
    whatHappened: str
    periodRecords: int
    changeStart: Optional[str]
    persistence: str
    pattern: str
    patternExplain: str
    increasedFactors: list[str]
    decreasedFactors: list[str]
    dataQualityNote: str
    generalRule: str
    priorityReason: str

    variables: list[VariableFactor]
    evidence: list[EvidenceItem]

    patientAge: Optional[int] = None
    patientProgram: Optional[str] = None

    modelVersion: str

    # Trazabilidad exigida por RISA: la explicación generada por LLM se
    # mantiene siempre junto a la versión determinista de la que partió.
    explanationRaw: str
    explanationSource: Literal["llm", "template_fallback"]
