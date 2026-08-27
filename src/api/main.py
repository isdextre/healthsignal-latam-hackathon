"""API HealthSignal LATAM — sirve señales con explicación redactada por Gemini.

Uso:
    uvicorn src.api.main:app --reload --port 8000

Todas las llamadas a Gemini ocurren en el backend (nunca en el navegador);
el front solo consume estos endpoints vía fetch. Ver docs/arquitectura_completa_azul.html
(Etapa 6+) y RISA_CONTEXTO_OFICIAL_HEALTHSIGNAL_LATAM.pdf (§11) para el
porqué de la separación evidencia/explicación generada.
"""

from __future__ import annotations

import logging
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from explain.data_loader import load_raw_signals  # noqa: E402
from explain.pipeline import build_signal_detail  # noqa: E402
from explain.schemas import SignalDetail, SignalSummary  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthsignal.api")

app = FastAPI(title="HealthSignal LATAM API", version="0.1.0")

# Desarrollo local: el front puede servirse desde file:// o un servidor
# estático distinto. Restringir esto antes de cualquier despliegue real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_raw_by_id: dict[str, dict] = {}
_warmup_lock = threading.Lock()
_warmup_done = False


def _index_raw_signals() -> None:
    global _raw_by_id
    _raw_by_id = {s["signal_id"]: s for s in load_raw_signals()}


def _warmup_cache() -> None:
    """Genera (o reutiliza la caché de) explicaciones LLM para todas las
    señales conocidas, para que /api/signals responda rápido. Corre en un
    hilo aparte para no bloquear el arranque del servidor."""
    global _warmup_done
    with _warmup_lock:
        if _warmup_done:
            return
        logger.info("Precalentando explicaciones para %d señales...", len(_raw_by_id))
        for raw in _raw_by_id.values():
            try:
                build_signal_detail(raw)
            except Exception:
                logger.exception("Fallo precalentando %s", raw["signal_id"])
        _warmup_done = True
        logger.info("Precalentamiento completo.")


@app.on_event("startup")
def on_startup() -> None:
    _index_raw_signals()
    threading.Thread(target=_warmup_cache, daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "signals_loaded": len(_raw_by_id), "cache_warm": _warmup_done}


@app.get("/api/signals", response_model=list[SignalSummary])
def list_signals() -> list[SignalSummary]:
    summaries = []
    for raw in _raw_by_id.values():
        detail = build_signal_detail(raw)
        summaries.append(
            SignalSummary(
                id=detail.id,
                patientId=detail.patientId,
                priority=detail.priority,
                riskScore=detail.riskScore,
                confidence=detail.confidence,
                decisionDatetime=detail.decisionDatetime,
                headline=detail.headline,
                isDemo=detail.isDemo,
                patientAge=detail.patientAge,
                patientProgram=detail.patientProgram,
                explanationSource=detail.explanationSource,
            )
        )
    return summaries


@app.get("/api/signals/{signal_id}", response_model=SignalDetail)
def get_signal(signal_id: str) -> SignalDetail:
    raw = _raw_by_id.get(signal_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Señal '{signal_id}' no encontrada")
    return build_signal_detail(raw)


@app.post("/api/signals/{signal_id}/regenerate", response_model=SignalDetail)
def regenerate_signal(signal_id: str) -> SignalDetail:
    """Fuerza una nueva llamada a Gemini para esta señal (ignora la caché).

    Pensado para la demo: muestra al jurado que la redacción con IA ocurre
    en vivo, no es texto precocinado.
    """
    raw = _raw_by_id.get(signal_id)
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Señal '{signal_id}' no encontrada")
    return build_signal_detail(raw, force_regenerate=True)
