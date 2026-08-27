# healthsignal-latam-hackathon

## Módulo de explicación con IA (Gemini)

Backend FastAPI que toma `submission_kit/signals.csv` + `evidence.csv` (o, más adelante, la salida real del pipeline de detección), genera una explicación determinista por plantillas, la reformula con Gemini (sin permitirle inventar cifras) y la sirve al front `docs/bosquejo_front.html`.

### 1. Configurar credenciales

```bash
cp .env.example .env
# editar .env y poner tu GEMINI_API_KEY real
```

### 2. Crear el entorno e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate   # cada vez que abras una terminal nueva
pip install -r requirements.txt
```

### 3. Levantar el backend

```bash
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

Verificar: `curl http://127.0.0.1:8000/api/health`

### 4. Abrir el front

Servir `docs/` con cualquier servidor estático (o abrir el HTML directamente) y entrar con cualquier email/contraseña (el login es simulado). Si el backend no está disponible, el front cae automáticamente a datos de demostración locales (`dataSource = "offline_fallback"`), mostrando un aviso — así el prototipo sigue siendo demostrable sin conexión.

```bash
cd docs && python3 -m http.server 8080
# abrir http://127.0.0.1:8080/bosquejo_front.html
```

### Arquitectura del módulo (`src/explain/`)

| Archivo | Responsabilidad |
|---|---|
| `data_loader.py` | Único punto que cambiará cuando el pipeline de detección del equipo entregue `candidate_signals_df` real — hoy lee `submission_kit/*.csv` |
| `templates.py` | Explicación determinista (sin LLM): factores por variable, patrón, regla general, evidencia |
| `gemini_client.py` | Llama a Gemini con salida estructurada para reformular el texto determinista, sin poder inventar cifras |
| `guardrails.py` | Rechaza la respuesta de Gemini si introduce números no presentes en la entrada o lenguaje de diagnóstico/prescripción — si falla, se usa la plantilla determinista como respaldo |
| `cache.py` | Evita repetir llamadas a Gemini para la misma señal/contexto |
| `pipeline.py` | Orquesta todo lo anterior y arma el objeto que consume el front |

`explanationRaw` (plantilla) y `explanationSource` (`llm` o `template_fallback`) siempre viajan junto a la explicación reformulada, para mantener evidencia y texto generado por IA claramente separados (requisito §11 del contexto oficial de RISA).