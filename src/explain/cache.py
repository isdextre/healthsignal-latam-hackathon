"""Caché local en disco para respuestas de Gemini ya validadas.

Evita volver a pagar/latencia de una llamada a Gemini en cada request del
backend. La clave incluye un hash del contexto enviado, así que si cambian
los datos de entrada (nuevo pipeline de detección real) la caché se
invalida sola en vez de servir una explicación desactualizada.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_FILE = CACHE_DIR / "explanations_cache.json"


def _context_hash(signal_id: str, context: dict) -> str:
    payload = json.dumps({"signal_id": signal_id, "context": context}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_all() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_all(data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get(signal_id: str, context: dict) -> dict | None:
    key = _context_hash(signal_id, context)
    return _load_all().get(key)


def set(signal_id: str, context: dict, narrative_output: dict) -> None:
    key = _context_hash(signal_id, context)
    data = _load_all()
    data[key] = narrative_output
    _save_all(data)
