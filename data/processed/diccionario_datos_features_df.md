# Diccionario de datos — `features_df.csv`

**Qué es este archivo:** es la tabla final (Base 2) que alimenta el modelo de detección de riesgo (Isolation Forest + reglas). Cada fila **no es un paciente, es una "ventana de tiempo" de un paciente**: un bloque de hasta 6 horas dentro de un episodio de atención (hospitalización, consulta ambulatoria o monitoreo domiciliario). Un mismo paciente aparece en varias filas, una por cada ventana de 6 horas que duró su episodio.

- **Filas:** 24,859 (ventanas de 6 horas)
- **Columnas:** 95
- **Pacientes distintos:** 1,000

Este documento explica qué significa cada columna, qué tipo de valor tiene y qué valores son posibles/esperados, en lenguaje simple para cualquiera que reciba este diccionario.

---

## 1. Identificación y ventana de tiempo (5 columnas)

| Columna | Qué significa | Tipo de valor | Valores posibles |
|---|---|---|---|
| `patient_id` | Identificador único del paciente | Texto | Ej: `PAT-0001` … `PAT-1000` (1,000 valores distintos) |
| `window_id` | Identificador único de la ventana de 6 horas (combina paciente + número de ventana) | Texto | Ej: `PAT-0001_w3`. Cada fila tiene un `window_id` distinto (24,859 valores únicos) |
| `window_start` | Fecha y hora en que empieza la ventana | Fecha/hora | Cualquier fecha/hora dentro del periodo de datos de RISA |
| `window_end` | Fecha y hora en que termina la ventana | Fecha/hora | Igual que arriba. Siempre posterior a `window_start` |
| `window_duration_h` | Duración real de la ventana, en horas | Número decimal | Normalmente **6.0**. Puede ser menor (mínimo observado: 0.25 h) cuando el episodio del paciente terminó antes de completar 6 horas — es la última ventana "recortada" de ese episodio |

---

## 2. Variables clínicas y de dispositivos monitoreadas

RISA registra 14 tipos de mediciones distintas. Para **cada una** de estas 14 variables, `features_df` calcula el mismo conjunto de 5 estadísticos (explicados en la sección 3). Por eso el nombre de casi todas las columnas sigue el patrón `VARIABLE_estadistico`.

| Código de variable | Qué mide | Unidad | Rango fisiológico plausible | Fuente típica |
|---|---|---|---|---|
| `HR` | Frecuencia cardiaca | latidos/min (bpm) | 20 – 220 | Signos vitales (monitor clínico) |
| `RR` | Frecuencia respiratoria | respiraciones/min (rpm) | 5 – 60 | Signos vitales |
| `SBP` | Presión arterial sistólica ("la alta") | mmHg | 60 – 240 | Signos vitales |
| `DBP` | Presión arterial diastólica ("la baja") | mmHg | 30 – 150 | Signos vitales |
| `SpO2` | Saturación de oxígeno en sangre | % | 50 – 100 | Signos vitales |
| `TEMP` | Temperatura corporal | °C | 30 – 45 | Signos vitales |
| `WEARABLE_HR` | Frecuencia cardiaca medida por el wearable (reloj/pulsera) | latidos/min (bpm) | 20 – 220 | Wearable |
| `STEPS` | Cantidad de pasos caminados | número de pasos | 0 – 1000 (por lectura) | Wearable |
| `ACTIVITY_LEVEL` | Nivel de actividad física reportado por el wearable | categoría (texto) | — | Wearable *(ver nota: en este archivo está siempre vacía, ver sección 4)* |
| `SIGNAL_QUALITY_INDEX` | Qué tan confiable es la señal del dispositivo médico/IoT en ese momento | proporción (0 a 1) | 0 – 1 | Dispositivo médico/IoT |
| `LAB_A` | Marcador de laboratorio sintético A | uA (unidad ficticia) | 0 – 50 | Laboratorio |
| `LAB_B` | Marcador de laboratorio sintético B | uB (unidad ficticia) | 0 – 300 | Laboratorio |
| `LAB_C` | Marcador de laboratorio sintético C | uC (unidad ficticia) | 0 – 10 | Laboratorio |
| `LAB_D` | Marcador de laboratorio sintético D | uD (unidad ficticia) | 0 – 10 | Laboratorio |

> Los rangos "plausibles" son los definidos en `05_metadata/variable_catalog.csv` del dataset RISA. Un valor fuera de ese rango se considera un dato implausible (posible error de sensor o de captura), no necesariamente una emergencia médica.

---

## 3. Los 5 estadísticos calculados por variable (70 columnas: 14 variables × 5)

Para cada una de las 14 variables de la sección 2, existen estas 5 columnas. Ejemplo con `HR`: `HR_coverage`, `HR_n_lecturas`, `HR_valor_medio`, `HR_zscore`, `HR_tendencia`.

| Sufijo | Qué significa | Tipo de valor | Cómo leerlo |
|---|---|---|---|
| `_n_lecturas` | Cuántas mediciones de esa variable se registraron dentro de esta ventana de 6 horas | Número entero (0, 1, 2, 3…) | 0 o vacío = no hubo ninguna lectura de esa variable en esta ventana (dato faltante, no un valor real de "cero") |
| `_coverage` | Qué tan completa fue la medición respecto a lo esperado según la frecuencia de muestreo típica de esa variable | Número decimal entre 0 y 1 | 1.0 = se registraron todas las lecturas esperadas; 0.5 = solo la mitad de las lecturas esperadas llegaron; valores bajos indican huecos de datos (por ejemplo, un sensor desconectado) |
| `_valor_medio` | Promedio de los valores registrados de esa variable dentro de la ventana | Número decimal, en la unidad de la variable (ver tabla de la sección 2) | Es el "valor típico" del paciente en esa ventana. Ejemplo: `HR_valor_medio = 82.3` significa que el pulso promedio en esas 6 horas fue 82.3 bpm |
| `_zscore` | Qué tan alejado está el `_valor_medio` de esta ventana respecto al comportamiento *habitual de ese mismo paciente* (no de la población general) | Número decimal, puede ser negativo o positivo | 0 = igual al promedio histórico del paciente; +2 o más = notablemente más alto de lo usual para ese paciente; -2 o menos = notablemente más bajo. **Advertencia:** cuando el paciente aún tiene muy pocas ventanas previas o su historial es casi constante, el z-score puede salir como un número extremadamente grande o incluso infinito (visto en `STEPS_zscore` y `SIGNAL_QUALITY_INDEX_zscore`) — esto debe filtrarse/acotarse antes de usarlo como entrada del modelo |
| `_tendencia` | Diferencia entre el valor de esta ventana y el de la ventana inmediatamente anterior del mismo paciente (pendiente simple, no promedio) | Número decimal, mismas unidades que la variable, puede ser negativo o positivo | Positivo = está subiendo respecto a la ventana anterior; negativo = está bajando; vacío = no hay ventana anterior con la que comparar (por ejemplo, es la primera ventana del episodio) |

**Cómo interpretar los "missing" (vacíos):** en todas estas columnas, un valor vacío significa "esa variable no se midió en esa ventana" — es información real (falta de dato), no un cero. Por ejemplo, los 4 marcadores de laboratorio (`LAB_A`…`LAB_D`) están vacíos en ~95-96% de las ventanas porque los exámenes de laboratorio no se toman cada 6 horas como los signos vitales, sino ocasionalmente.

---

## 4. Columnas que están siempre vacías en esta versión de los datos

Al revisar el archivo real generado, estas 9 columnas no tienen **ningún** valor (100% vacías) en las 24,859 filas:

- `ACTIVITY_LEVEL_valor_medio`, `ACTIVITY_LEVEL_coverage`, `ACTIVITY_LEVEL_zscore`, `ACTIVITY_LEVEL_tendencia`
- `LAB_A_coverage`, `LAB_B_coverage`, `LAB_C_coverage`, `LAB_D_coverage`
- `SIGNAL_QUALITY_INDEX_coverage`

**Por qué:** `ACTIVITY_LEVEL` es una variable categórica (texto: "LIGHT", "MODERATE", "HIGH"), no numérica, así que no se le puede calcular un promedio, z-score o tendencia numérica de la forma en que se calcula para HR o TEMP — su información equivalente ya está capturada de otra forma en la columna `context_physical_activity` (ver sección 6). El `_coverage` de los laboratorios y de `SIGNAL_QUALITY_INDEX` quedó vacío porque esa métrica se calculó solo para variables con una frecuencia de muestreo regular esperada (signos vitales y wearable); los laboratorios y la calidad de señal se registran de forma puntual/event-based, no periódica, así que no aplica directamente el mismo cálculo de cobertura.

Esto es importante dejarlo explícito en el README/evidencia técnica del hackathon: no es un bug oculto, es una limitación conocida y explicable de esta versión del pipeline.

---

## 5. Datos del paciente y del episodio de atención (contexto estático — 10 columnas)

Estas columnas no cambian entre ventanas de un mismo episodio: describen quién es el paciente y en qué tipo de atención está.

| Columna | Qué significa | Valores posibles observados |
|---|---|---|
| `sex_at_birth` | Sexo del paciente | `F` (12,939 ventanas), `M` (11,920 ventanas) |
| `age_years` | Edad del paciente en años | Números enteros, de 18 a 90 años |
| `age_group` | Grupo etario del paciente | `18-39`, `40-59`, `60-74`, `75+` |
| `care_program` | Programa de atención en el que está el paciente | `HOME_MONITORING`, `AMBULATORY`, `POST_DISCHARGE`, `GENERAL_FOLLOWUP` |
| `baseline_risk_profile` | Perfil de riesgo de base asignado al paciente (según sus condiciones previas) | `GENERAL`, `CARDIORESPIRATORY_CONTEXT`, `METABOLIC_CONTEXT`, `OLDER_ADULT_CONTEXT` |
| `facility_id` | Centro de salud donde ocurre el episodio | `FAC-01`, `FAC-02`, `FAC-05` |
| `care_setting` | Lugar físico de la atención | `HOME` (monitoreo domiciliario), `FACILITY` (en un centro de salud) |
| `encounter_type` | Tipo de episodio clínico | `HOME_MONITORING_EPISODE`, `HOSPITAL_OBSERVATION`, `AMBULATORY_FOLLOWUP` |
| `digital_maturity` | Nivel de madurez tecnológica del centro/canal que generó los datos de esta ventana | `HIGH`, `MEDIUM_HIGH`, `VARIABLE` |
| `connectivity_profile` | Qué tan estable es la conectividad esperada en ese contexto de atención | `STABLE`, `VARIABLE` |

---

## 6. Historial clínico y medicación (6 columnas — todas son "banderas" de 0/1)

| Columna | Qué significa | Valores posibles |
|---|---|---|
| `tiene_cardiovascular_history` | El paciente tiene alguna condición cardiovascular registrada en su historia clínica | `1` = sí (30% de las ventanas), `0` = no |
| `tiene_metabolic_history` | El paciente tiene alguna condición metabólica registrada (ej. diabetes) | `1` = sí (32%), `0` = no |
| `tiene_renal_history` | El paciente tiene alguna condición renal registrada | `1` = sí (27%), `0` = no |
| `tiene_respiratory_history` | El paciente tiene alguna condición respiratoria registrada | `1` = sí (31%), `0` = no |
| `tiene_no_major_recorded_history` | El paciente no tiene ninguna condición mayor registrada en su historia clínica | `1` = sí (29%), `0` = no |
| `med_active` | El paciente tenía al menos un medicamento activo (en administración) durante esta ventana de 6 horas | `1` = sí (6% de las ventanas), `0` = no |

> Un mismo paciente puede tener varias banderas en `1` a la vez (por ejemplo, historia cardiovascular Y metabólica simultáneamente) — no son mutuamente excluyentes.

---

## 7. Contexto dinámico de la ventana (4 columnas)

A diferencia de la sección 5 (que no cambia en el episodio), estas columnas sí pueden variar ventana a ventana, porque describen eventos o condiciones puntuales que ocurrieron *durante* esa ventana específica.

| Columna | Qué significa | Valores posibles | Vacíos |
|---|---|---|---|
| `connectivity_flag` | Si hubo algún problema de conectividad del dispositivo/canal durante esta ventana | `DISCONNECTED` (dispositivo desconectado), `INTERMITTENT` (conexión intermitente), `DELAYED_SYNC` (los datos llegaron con retraso) | Vacío en 97.5% de las ventanas = no se registró ningún problema de conectividad en ese periodo |
| `context_physical_activity` | Nivel de actividad física del paciente reportado como contexto durante la ventana | `LIGHT`, `MODERATE`, `HIGH` | Vacío en 90.1% = no hay un registro de contexto de actividad para esa ventana |
| `context_recovery_phase` | Si el paciente estaba en una fase de recuperación post-actividad | `POST_ACTIVITY_RECOVERY` | Vacío en 99.7% = casi nunca se registra este contexto (solo 72 ventanas lo tienen) |
| `context_sleep_state` | Si el paciente estaba dormido durante (parte de) la ventana | `SLEEP` | Vacío en 49.8% = la mitad de las ventanas caen totalmente fuera de un periodo de sueño registrado |

**Por qué hay tantos vacíos aquí:** estas columnas vienen de eventos de contexto (`patient_context`) que no cubren las 24 horas del día de forma continua — solo se registran cuando efectivamente ocurre algo relevante (el paciente hace ejercicio, se está recuperando, o está durmiendo). Un vacío significa "no aplica / sin información de contexto para ese momento", no "el paciente no hizo esa actividad".

---

## Resumen para quien vaya a usar esta tabla en el modelo

- Cada fila = una ventana de 6 horas de un paciente dentro de un episodio de atención.
- Las columnas con sufijo `_valor_medio`, `_zscore`, `_tendencia`, `_coverage`, `_n_lecturas` describen **cómo se comportó cada variable clínica/dispositivo en esa ventana**, tanto en términos absolutos (`_valor_medio`) como relativos al propio historial del paciente (`_zscore`, `_tendencia`) y a la calidad/cantidad del dato (`_coverage`, `_n_lecturas`).
- Las columnas de la sección 5 y 6 dan el **contexto fijo** del paciente (quién es, qué antecedentes tiene).
- Las columnas de la sección 7 dan **contexto puntual** de esa ventana específica (conectividad, actividad, sueño).
- Antes de usar `_zscore` como entrada directa del Isolation Forest, conviene acotar (winsorizar/clip) los valores extremos o infinitos descritos en la sección 3, para que no dominen artificialmente el resultado del modelo.
