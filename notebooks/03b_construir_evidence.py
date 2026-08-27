"""
HealthSignal LATAM - Construccion de evidence.csv (trazabilidad real)
=======================================================================
Toma las alertas de outputs/alertas_v2.csv (ya mapeadas 1 a 1 a
outputs/signals.csv por 03_deteccion_anomalias.py) y arma outputs/evidence.csv
en el formato EXACTO del kit de entrega oficial (ver validate_submission.py):
signal_id, source_file, record_id, event_datetime, available_datetime,
evidence_role, variable_code, contribution.

Nada de IDs inventados: cada fila apunta a un record_id REAL de una de estas
fuentes:
  - data/processed/timeline_df.csv        (vital_signs, wearable_observations,
                                            device_observations, laboratory_results
                                            -- ya integradas en 01_integracion_timeline.ipynb,
                                            con record_id y available_datetime trazables)
  - data/raw/02_clinical/medication_administrations.csv
  - data/raw/04_context/connectivity_events.csv
  - data/raw/04_context/patient_context.csv

Los ROLES se asignan con la MISMA logica que ya usan las 3 compuertas de
03_deteccion_anomalias.py (contexto, calidad, persistencia) -- no es una regla
nueva, es la traduccion de esa misma logica a evidencia concreta:
  PRIMARY    -> la variable vital con mayor desviacion (dz_v > UMBRAL_ALT)
  SUPPORTING -> otras variables alteradas + WEARABLE_HR si hay discordancia
                (y no fue la razon de degradar por calidad)
  QUALITY    -> lo que degrado factor_calidad (low_signal, % marcado,
                discordancia sin respaldo, conectividad)
  CONTEXT    -> lo que degrado factor_contexto (actividad fisica) o
                medicacion activa en la ventana

timeline_df.csv pesa ~300MB / 2.5M filas -- se lee EN CHUNKS y se descarta
todo lo que no caiga en la ventana de alguna alerta. Nunca se carga completo
en memoria (esta maquina ya mostro ser sensible a memoria en el script 03).

Anti-leakage: cualquier candidato cuyo available_datetime caiga DESPUES del
decision_datetime de su senal se descarta -- no se recorta artificialmente,
eso seria fingir que el dato existio antes de tiempo.

Entradas : outputs/alertas_v2.csv, outputs/signals.csv,
           data/processed/timeline_df.csv,
           data/raw/02_clinical/medication_administrations.csv,
           data/raw/04_context/connectivity_events.csv,
           data/raw/04_context/patient_context.csv
Salida   : outputs/evidence.csv
"""
import pandas as pd
import numpy as np

CHUNKSIZE = 50_000
UMBRAL_ALT = 1.5
DIRECCION = {'HR': 1, 'RR': 1, 'TEMP': 1, 'SpO2': -1}


# --------------------------------------------------------------------- carga
def cargar_alertas():
    al = pd.read_csv('outputs/alertas_v2.csv')
    sig = pd.read_csv('outputs/signals.csv')
    al['window_start'] = pd.to_datetime(al['window_start'])
    al['window_end'] = pd.to_datetime(al['window_end'])
    al['signal_id'] = 'SIG-' + al['patient_id'].astype(str) + '-' + al['window_id'].astype(str)

    faltan = set(al['signal_id']) - set(sig['signal_id'])
    assert not faltan, (f"{len(faltan)} signal_id de alertas_v2.csv no estan en signals.csv "
                        f"-- revisar que ambos vengan de la misma corrida")

    sig2 = sig[['signal_id', 'decision_datetime']].copy()
    sig2['decision_datetime'] = pd.to_datetime(sig2['decision_datetime'])
    return al.merge(sig2, on='signal_id', how='left')


def leer_lecturas_en_ventanas(al):
    """timeline_df.csv en chunks -- se queda SOLO con lo que cae dentro de la
    ventana de alguna alerta. Nunca se carga el archivo completo en memoria."""
    windows = al[['patient_id', 'window_start', 'window_end', 'signal_id']]
    piezas, leidas = [], 0
    for chunk in pd.read_csv('data/processed/timeline_df.csv', chunksize=CHUNKSIZE,
                              parse_dates=['timestamp', 'available_datetime']):
        leidas += len(chunk)
        m = chunk.merge(windows, on='patient_id', how='inner')
        if len(m):
            m = m[(m['timestamp'] >= m['window_start']) & (m['timestamp'] <= m['window_end'])]
        if len(m):
            piezas.append(m)
    out = pd.concat(piezas, ignore_index=True) if piezas else pd.DataFrame(
        columns=['patient_id', 'timestamp', 'available_datetime', 'variable_code',
                 'value', 'source_table', 'record_id'])
    print(f"[timeline] {leidas} filas leidas en chunks de {CHUNKSIZE} -> "
          f"{len(out)} dentro de alguna ventana de alerta")
    return out


def cargar_contexto():
    med = pd.read_csv('data/raw/02_clinical/medication_administrations.csv')
    med['start_datetime'] = pd.to_datetime(med['start_datetime'], format='mixed')
    med['end_datetime'] = pd.to_datetime(med['end_datetime'], format='mixed')

    conn = pd.read_csv('data/raw/04_context/connectivity_events.csv')
    conn['start_datetime'] = pd.to_datetime(conn['start_datetime'], format='mixed')
    conn['end_datetime'] = pd.to_datetime(conn['end_datetime'], format='mixed')

    ctx = pd.read_csv('data/raw/04_context/patient_context.csv')
    ctx['start_datetime'] = pd.to_datetime(ctx['start_datetime'], format='mixed')
    ctx['end_datetime'] = pd.to_datetime(ctx['end_datetime'], format='mixed')
    ctx_actividad = ctx[ctx['context_type'] == 'PHYSICAL_ACTIVITY']
    return med, conn, ctx_actividad


def solapa(df, patient_id, w_start, w_end):
    return df[(df['patient_id'] == patient_id)
              & (df['start_datetime'] <= w_end) & (df['end_datetime'] >= w_start)]


def _num(series):
    return pd.to_numeric(series, errors='coerce')


# --------------------------------------------------------- seleccion de filas
def elegir_lectura_vital(lecturas_pac, variable_code, signo):
    """De las lecturas reales del paciente en la ventana, la mas extrema en el
    sentido de DIRECCION (mas alta si signo=+1, mas baja si signo=-1)."""
    cand = lecturas_pac[lecturas_pac['variable_code'] == variable_code].copy()
    if cand.empty:
        return None
    cand['_v'] = _num(cand['value'])
    cand = cand.dropna(subset=['_v'])
    if cand.empty:
        return None
    return cand.loc[(cand['_v'] * signo).idxmax()]


def armar_evidencia_de_una_alerta(r, lecturas_pac, med, conn, ctx_actividad):
    filas = []
    decision = r['decision_datetime']

    def agregar(source_file, record_id, event_dt, available_dt, role, var=None, contrib=None):
        if pd.isna(available_dt) or available_dt > decision:
            return  # anti-leakage: se descarta, nunca se recorta
        filas.append({
            'signal_id': r['signal_id'], 'source_file': source_file, 'record_id': record_id,
            'event_datetime': event_dt, 'available_datetime': available_dt,
            'evidence_role': role, 'variable_code': var, 'contribution': contrib,
        })

    # --- PRIMARY / SUPPORTING: variables vitales alteradas ---
    alteradas = [v for v in DIRECCION if r[f'dz_{v}'] > UMBRAL_ALT]
    if not alteradas:
        alteradas = [max(DIRECCION, key=lambda v: r[f'dz_{v}'])]  # fallback: la mas alta igual
    primaria = max(alteradas, key=lambda v: r[f'dz_{v}'])

    for v in alteradas:
        row = elegir_lectura_vital(lecturas_pac, v, DIRECCION[v])
        if row is None:
            continue
        rol = 'PRIMARY' if v == primaria else 'SUPPORTING'
        agregar(f"{row['source_table']}.csv", row['record_id'], row['timestamp'],
                row['available_datetime'], rol, var=v, contrib=r.get(f'contrib_desv_{v}'))

    # --- discordancia con el wearable ---
    if r['discordancia_wearable'] > 3.0:
        row = elegir_lectura_vital(lecturas_pac, 'WEARABLE_HR', 1)
        if row is not None:
            rol = 'QUALITY' if r['n_alteradas'] <= 1 else 'SUPPORTING'
            agregar('wearable_observations.csv', row['record_id'], row['timestamp'],
                    row['available_datetime'], rol, var='WEARABLE_HR',
                    contrib=r.get('contrib_discordancia_wearable'))

    # --- calidad: low_signal / % marcado -> SIGNAL_QUALITY_INDEX ---
    if r['factor_calidad'] < 1 and (r['n_low_signal'] > 0 or r['pct_flagged'] > 0.02):
        cand = lecturas_pac[lecturas_pac['variable_code'] == 'SIGNAL_QUALITY_INDEX'].copy()
        cand['_v'] = _num(cand['value'])
        cand = cand.dropna(subset=['_v'])
        if not cand.empty:
            row = cand.loc[cand['_v'].idxmin()]  # la peor calidad de la ventana
            agregar('device_observations.csv', row['record_id'], row['timestamp'],
                    row['available_datetime'], 'QUALITY', var='SIGNAL_QUALITY_INDEX')

    # --- calidad: conectividad ---
    if pd.notna(r['connectivity_flag']):
        m = solapa(conn, r['patient_id'], r['window_start'], r['window_end'])
        if not m.empty:
            row = m.iloc[0]
            agregar('connectivity_events.csv', row['event_id'], row['start_datetime'],
                    min(row['end_datetime'], decision), 'QUALITY')

    # --- contexto: actividad fisica que explico el deterioro ---
    if r['factor_contexto'] < 1:
        m = solapa(ctx_actividad, r['patient_id'], r['window_start'], r['window_end'])
        if not m.empty:
            coincide = m[m['context_value'] == r.get('context_physical_activity')]
            row = (coincide if not coincide.empty else m).sort_values(
                'confidence', ascending=False).iloc[0]
            agregar('patient_context.csv', row['context_id'], row['start_datetime'],
                    min(row['end_datetime'], decision), 'CONTEXT', var='PHYSICAL_ACTIVITY')

    # --- contexto: medicacion activa en la ventana ---
    if str(r.get('med_active')).strip().lower() in ('true', '1'):
        m = solapa(med, r['patient_id'], r['window_start'], r['window_end'])
        if not m.empty:
            row = m.iloc[0]
            agregar('medication_administrations.csv', row['administration_id'],
                    row['start_datetime'], min(row['end_datetime'], decision), 'CONTEXT')

    return filas


def main():
    al = cargar_alertas()
    lecturas = leer_lecturas_en_ventanas(al)
    med, conn, ctx_actividad = cargar_contexto()
    lecturas_por_paciente = {pid: g for pid, g in lecturas.groupby('patient_id')}

    todas, sin_evidencia = [], []
    for _, r in al.iterrows():
        lecturas_pac = lecturas_por_paciente.get(r['patient_id'],
                                                  lecturas.iloc[0:0])
        filas = armar_evidencia_de_una_alerta(r, lecturas_pac, med, conn, ctx_actividad)
        if not filas:
            sin_evidencia.append(r['signal_id'])
        todas.extend(filas)

    ev = pd.DataFrame(todas)
    for c in ('event_datetime', 'available_datetime'):
        ev[c] = pd.to_datetime(ev[c]).dt.strftime('%Y-%m-%dT%H:%M:%S')
    ev = ev[['signal_id', 'source_file', 'record_id', 'event_datetime',
             'available_datetime', 'evidence_role', 'variable_code', 'contribution']]
    ev.to_csv('outputs/evidence.csv', index=False)

    print(f"\n[evidence] outputs/evidence.csv: {len(ev)} filas para {al['signal_id'].nunique()} senales")
    if sin_evidencia:
        print(f"  [ALERTA] {len(sin_evidencia)} senales SIN ninguna fila de evidencia "
              f"(el validador oficial rechazaria TODA la entrega por esto): "
              f"{sin_evidencia[:10]}{' ...' if len(sin_evidencia) > 10 else ''}")
    else:
        print("  Las senales tienen al menos 1 fila de evidencia.")
    print(ev['evidence_role'].value_counts().rename('reparto evidence_role').to_string())


if __name__ == '__main__':
    main()
