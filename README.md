# healthsignal-latam-hackathon

## Módulo de explicación con IA (Gemini)

Backend FastAPI que toma `submission_kit/signals.csv` + `evidence.csv` (o, más adelante, la salida real del pipeline de detección), genera una explicación determinista por plantillas, la reformula con Gemini (sin permitirle inventar cifras) y la sirve al front `docs/bosquejo_front.html`.

Este repositorio cubre la detección de la señal, su explicación y el front de revisión. La **notificación al personal de salud** (envío de la alerta ya generada a Telegram o WhatsApp) se resuelve fuera de este repo, con un flujo de **n8n** que consume la señal y dispara el mensaje — no está incluido aquí.

### 1. Configurar credenciales

```bash
cp .env.example .env
# editar .env y poner tu GEMINI_API_KEY real
```

En Windows (PowerShell), si `cp` no está disponible:

```powershell
copy .env.example .env
# editar .env y poner tu GEMINI_API_KEY real
```

### 2. Crear el entorno e instalar dependencias

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # cada vez que abras una terminal nueva
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # cada vez que abras una terminal nueva
pip install -r requirements.txt
```

Si PowerShell bloquea el script de activación (`no se puede cargar porque la ejecución de scripts está deshabilitada`), habilítalo una sola vez con:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 3. Levantar el backend

**macOS / Linux:**

```bash
PYTHONPATH=src uvicorn api.main:app --reload --port 8000
```

**Windows (PowerShell):**

```powershell
$env:PYTHONPATH = "src"
uvicorn api.main:app --reload --port 8000
```

**Windows (CMD):**

```cmd
set PYTHONPATH=src
uvicorn api.main:app --reload --port 8000
```

Verificar: `curl http://127.0.0.1:8000/api/health` (en Windows, `curl` funciona igual desde PowerShell; si no está disponible, abre esa URL directamente en el navegador).

### 4. Abrir el front

Servir `docs/` con cualquier servidor estático (o abrir el HTML directamente) y entrar con cualquier email/contraseña (el login es simulado). Si el backend no está disponible, el front cae automáticamente a datos de demostración locales (`dataSource = "offline_fallback"`), mostrando un aviso — así el prototipo sigue siendo demostrable sin conexión.

**macOS / Linux:**

```bash
cd docs && python3 -m http.server 8080
# abrir http://127.0.0.1:8080/bosquejo_front.html
```

**Windows (PowerShell / CMD):**

```powershell
cd docs
python -m http.server 8080
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

### Notificación (fuera de este repo)

La señal con su explicación, tal como la sirve `/api/signals`, es el input de un flujo de **n8n** que arma el mensaje de alerta y lo entrega al personal de salud por **Telegram o WhatsApp**. Ese flujo vive fuera de este repositorio; aquí solo se documenta como parte del recorrido completo de la alerta, desde la detección hasta la notificación.
