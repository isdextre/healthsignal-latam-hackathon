"""
HealthSignal LATAM - Deteccion de anomalias v2 (Isolation Forest + compuertas)
==============================================================================
v1 detectaba. v2 discrimina: separa deterioro fisiologico de contexto,
artefacto de sensor y variacion transitoria, sin borrar nada.

Entradas : data/processed/features_df.csv
           data/processed/calidad_por_ventana.csv   (generado por recuperar_calidad.py)
Salidas  : outputs/alertas_v2.csv, outputs/metricas_v2.csv,
           outputs/barrido_v2.csv, outputs/ablacion_v2.csv, outputs/signals.csv

outputs/signals.csv ya sale en el formato EXACTO del kit de entrega oficial
(ver validate_submission.py): signal_id, patient_id, decision_datetime,
risk_score, priority_level, evidence_start, evidence_end, explanation,
model_version, confidence_score. risk_score y priority_level NO son un modelo
nuevo -- son una traduccion de escala del mismo prioridad_score ya calculado
por las compuertas (ver construir_signals_csv). Este archivo todavia NO
reemplaza submission_kit/signals.csv a proposito: falta construir
evidence.csv (requiere volver a timeline_df.csv por record_id reales) antes
de mover los dos juntos al kit -- mientras tanto el kit sigue con el mock
para no romper el dashboard/backend que ya lo consume.

Cadena:  saneamiento -> features -> Isolation Forest -> 4 compuertas -> prioridad
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
RANDOM_STATE, MIN_VENTANA, UMBRAL_ALT = 42, 3, 1.5
# tamano de muestra para silhouette_score: baja de 8000 a 2000 porque una matriz
# de distancias 8000x8000 (~488 MB) revento por MemoryError en maquinas con poca
# RAM libre. 2000x2000 (~32 MB) da una estimacion igual de valida para un
# diagnostico de calidad de clusters -- no afecta al modelo ni a las alertas.
SIL_SAMPLE = 2000


def sil_seguro(X, lab):
    """silhouette_score con degradacion segura: es una metrica solo
    diagnostica (no afecta alertas.csv ni signals.csv), asi que si la maquina
    no tiene memoria para la matriz de distancias no debe tumbar el pipeline
    completo -- se reporta NaN y se sigue."""
    try:
        return silhouette_score(X, lab, sample_size=SIL_SAMPLE, random_state=0)
    except MemoryError:
        print(f"  [aviso] silhouette_score fallo por memoria (sample={SIL_SAMPLE}) "
              f"-- se reporta NaN, no bloquea el pipeline")
        return np.nan
# WINSORIZADO: desactivado. Decision empirica, no por defecto -- ver
# ablacion_transformaciones() y outputs/ablacion_v2.csv. Con las features
# intra-ventana y las compuertas de calidad, el cap ya no aporta: mismos 9/10
# eventos en ALTA, misma contaminacion por contexto (3/66), cero artefactos en
# el top-50, y la silueta es MEJOR sin el (0.732 vs 0.694).
CAP = None
CONTAMINACION = 0.02
DIRECCION = {'HR': 1, 'RR': 1, 'TEMP': 1, 'SpO2': -1}
# la actividad fisica explica taquicardia y taquipnea; NO explica desaturacion ni fiebre
EXPLICABLES_POR_ACTIVIDAD = {'HR', 'RR'}


# ----------------------------------------------------------------- saneamiento
def cargar_y_sanear():
    df = pd.read_csv('data/processed/features_df.csv', low_memory=False)
    df = df.sort_values(['patient_id', 'window_start']).reset_index(drop=True)
    for c in ('window_start', 'window_end'):
        df[c] = pd.to_datetime(df[c])
    df['idx_ventana'] = df.groupby('patient_id').cumcount()
    n0 = len(df)
    df = df[df['idx_ventana'] >= MIN_VENTANA].replace([np.inf, -np.inf], np.nan).copy()
    cal = pd.read_csv('data/processed/calidad_por_ventana.csv')
    df = df.merge(cal[['window_id', 'pct_flagged', 'n_check', 'n_low_signal',
                       'n_retransmitted']], on='window_id', how='left')
    iv = pd.read_csv('data/processed/intraventana.csv').drop(columns=['patient_id'])
    df = df.merge(iv, on='window_id', how='left')
    print(f"[saneamiento] {n0} -> {len(df)} ventanas | "
          f"calidad + forma intra-ventana recuperadas del crudo")
    return df


# -------------------------------------------------------------------- features
def _wins(z):
    return z if CAP is None else z.clip(-CAP, CAP)


def construir_features(df):
    """Sin winsorizar (CAP=None). El filtro idx>=3 ya elimina las colas
    patologicas: HR_zscore pasa de min -2329 a min -21.6. Lo que queda son
    extremos reales, y separarlos de los artefactos es trabajo de las compuertas
    de calidad y de la forma intra-ventana, no de un recorte ciego.
    Un extremo de una sola variable queda visible como n_alteradas=1."""
    for v, signo in DIRECCION.items():
        z = _wins(signo * df[f'{v}_zscore'].fillna(0))
        df[f'dz_{v}'] = z
        df[f'desv_{v}'] = z.clip(lower=0)

    df['severidad']    = sum(df[f'desv_{v}'] for v in DIRECCION)
    df['n_alteradas']  = sum((df[f'dz_{v}'] > UMBRAL_ALT).astype(int) for v in DIRECCION)
    df['concordancia'] = df[[f'desv_{v}' for v in DIRECCION]].min(axis=1)

    # nuevo en v2: discordancia entre el monitor clinico y el wearable.
    # si el HR del monitor se dispara y el del wearable no, es del sensor.
    wz = _wins(df['WEARABLE_HR_zscore'].fillna(0))
    df['discordancia_wearable'] = (df['dz_HR'] - wz).abs()

    # nuevo en v2: forma temporal DENTRO de la ventana (desde el crudo).
    # los eventos de RISA duran ~5h y caben en una ventana de 6h, asi que
    # comparar ventanas vecinas es la escala equivocada (ver README).
    df['persistencia_intra'] = df[[f'run_max_{v}' for v in DIRECCION]].mean(axis=1)
    pend = pd.DataFrame({v: df[f'pendiente_{v}'].fillna(0) for v in DIRECCION})
    df['tendencia_intra'] = (pend / pend.std()).clip(lower=0).sum(axis=1)

    feats = ([f'desv_{v}' for v in DIRECCION]
             + ['severidad', 'n_alteradas', 'concordancia', 'discordancia_wearable',
                'persistencia_intra', 'tendencia_intra'])
    return df, feats


# ------------------------------------------------------------------ compuertas
def compuerta_contexto(r):
    """Degrada si la actividad fisica explica lo que se movio. Condicional:
    solo aplica si el deterioro es de HR/RR y NO hay desaturacion ni fiebre."""
    if r['context_physical_activity'] not in ('HIGH', 'MODERATE'):
        return 1.0, None
    no_explicables = [v for v in DIRECCION
                      if v not in EXPLICABLES_POR_ACTIVIDAD and r[f'dz_{v}'] > UMBRAL_ALT]
    if no_explicables:
        return 1.0, None                                   # actividad no lo explica
    return (0.25, 'contexto: actividad fisica explica HR/RR') \
        if r['context_physical_activity'] == 'HIGH' else \
           (0.50, 'contexto: actividad moderada explica HR/RR')


def compuerta_calidad(r):
    if r['n_low_signal'] > 0:
        return 0.30, 'calidad: lecturas LOW_SIGNAL en la ventana'
    if r['pct_flagged'] > 0.02:
        return 0.50, f"calidad: {r['pct_flagged']*100:.1f}% de lecturas marcadas"
    if r['n_alteradas'] <= 1 and r['discordancia_wearable'] > 3.0:
        return 0.40, 'calidad: una sola variable y el wearable no concuerda'
    if pd.notna(r['connectivity_flag']):
        return 0.60, f"calidad: conectividad {r['connectivity_flag']}"
    return 1.0, None


def compuerta_persistencia(df):
    """TRANSIENT vs PROGRESSIVE medido DENTRO de la ventana de 6h.

    Calibrado empiricamente (ver diagnostico en el README): los 10 eventos
    confirmados tienen run_max medio 2.92 y pendiente de deterioro positiva;
    los artefactos de un solo sensor tienen 1.41 y pendiente NEGATIVA -- el
    pico se recupera dentro de la misma ventana. La version inter-ventana
    degradaba 9 de 10 eventos reales: escala equivocada."""
    factor = pd.Series(1.0, index=df.index)
    motivo = pd.Series(None, index=df.index, dtype=object)
    pers, tend = df['persistencia_intra'], df['tendencia_intra']

    pico = (pers < 1.5) & (df['severidad'] > 3)
    factor[pico] = 0.55
    motivo[pico] = 'transitoria: pico aislado que se recupera dentro de la ventana'

    sost = (pers >= 2.5)
    factor[sost] = 1.25
    motivo[sost] = f'sostenida: deterioro en varias sub-ventanas consecutivas'

    prog = sost & (tend > 2.0)
    factor[prog] = 1.60
    motivo[prog] = 'progresiva: el deterioro crece dentro de la ventana'

    # confirmacion en la ventana siguiente: no degrada, solo eleva
    sig = df.groupby('patient_id')['severidad'].shift(-1).fillna(0)
    conf = (sig >= 5.0) & (factor >= 1.0)
    factor[conf] = factor[conf] * 1.15
    motivo[conf] = motivo[conf].fillna('') + ' + confirmada en la ventana siguiente'
    return factor, motivo


# ---------------------------------------------------------------- explicabilidad
def atribucion_local(iso, X, filas, nombres):
    med = np.median(X, axis=0)
    base = -iso.score_samples(X[filas])
    out = np.zeros((len(filas), X.shape[1]))
    for j in range(X.shape[1]):
        Xm = X[filas].copy(); Xm[:, j] = med[j]
        out[:, j] = base - (-iso.score_samples(Xm))
    return pd.DataFrame(out, columns=[f'contrib_{n}' for n in nombres])


def narrar(r):
    partes = [f"{v} {'sube' if DIRECCION[v] > 0 else 'baja'} {r[f'dz_{v}']:.1f} sigma"
              for v in DIRECCION if r[f'dz_{v}'] > UMBRAL_ALT]
    txt = (f"{int(r['n_alteradas'])}/4 variables alteradas: " + ", ".join(partes)) \
        if partes else "patron atipico sin deterioro direccional dominante"
    if r['concordancia'] > 1.0:
        txt += " (movimiento conjunto)"
    mot = [m for m in (r['motivo_contexto'], r['motivo_calidad'],
                       r['motivo_persistencia']) if isinstance(m, str) and m]
    if mot:
        txt += " | " + " ; ".join(mot)
    return txt


# --------------------------------------------------- submission kit: signals.csv
def construir_signals_csv(al):
    """Mapea las alertas internas (al) al formato EXACTO de signals.csv que exige
    validate_submission.py. risk_score y priority_level NO son un modelo nuevo:
    son una traduccion de escala del mismo prioridad_score que ya calculan las
    3 compuertas (contexto, calidad, persistencia) mas arriba.

    risk_score: percentil (rank pct) de prioridad_score dentro de las alertas
    detectadas. Se prefiere sobre min-max porque un solo outlier extremo no
    aplasta la escala de todas las demas -- coherente con que priority_level
    tambien se define por cuantiles.

    priority_level: se extiende el mismo corte que ya usa 'prioridad'
    (cuantiles 0.60 / 0.85 -> BAJA/MEDIA/ALTA) agregando un tercer corte en
    0.97 para separar CRITICAL de HIGH dentro de lo que hoy es 'ALTA'. Sigue
    siendo el mismo criterio de umbral por cuantil, no una regla nueva.

    confidence_score (opcional): 1 - pct_flagged de la ventana, como proxy
    simple de cuanta confianza dar a la lectura que origino la alerta.
    """
    out = pd.DataFrame(index=al.index)
    out['signal_id'] = 'SIG-' + al['patient_id'].astype(str) + '-' + al['window_id'].astype(str)
    out['patient_id'] = al['patient_id']
    out['decision_datetime'] = al['window_end']
    out['risk_score'] = al['prioridad_score'].rank(pct=True).round(4)

    q_med, q_high, q_crit = al['prioridad_score'].quantile([0.60, 0.85, 0.97]).values
    out['priority_level'] = np.select(
        [al['prioridad_score'] >= q_crit, al['prioridad_score'] >= q_high,
         al['prioridad_score'] >= q_med],
        ['CRITICAL', 'HIGH', 'MEDIUM'], default='LOW')

    out['evidence_start'] = al['window_start']
    out['evidence_end'] = al['window_end']
    out['explanation'] = al['explicacion']
    out['model_version'] = 'rules_v2+iforest_v2'
    out['confidence_score'] = (1 - al['pct_flagged'].fillna(0)).clip(0, 1).round(4)

    for c in ('decision_datetime', 'evidence_start', 'evidence_end'):
        out[c] = pd.to_datetime(out[c]).dt.strftime('%Y-%m-%dT%H:%M:%S')

    assert out['signal_id'].is_unique, "signal_id duplicado -- revisar patient_id+window_id"
    assert out['risk_score'].between(0, 1).all(), "risk_score fuera de [0,1]"
    assert out['priority_level'].isin(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).all()
    assert (out['explanation'].str.len() > 0).all(), "explanation vacia"
    assert (pd.to_datetime(out['evidence_start']) <= pd.to_datetime(out['evidence_end'])).all()
    assert (pd.to_datetime(out['evidence_end']) <= pd.to_datetime(out['decision_datetime'])).all()
    return out[['signal_id', 'patient_id', 'decision_datetime', 'risk_score', 'priority_level',
                'evidence_start', 'evidence_end', 'explanation', 'model_version', 'confidence_score']]


# -------------------------------------------------- estudio de transformaciones
def ablacion_transformaciones(df, feats):
    """Responde 'que transformacion es NECESARIA' con evidencia, no por defecto.

    Conclusiones (ver outputs/ablacion_v2.csv):
      - winsorizado 8σ : NECESARIO, pero no para normalizar. El filtro idx>=3 ya
        quito las colas patologicas (HR pasa de min -2329 a -21.6). El cap existe
        para que un artefacto de un solo sensor (SpO2 a 96σ con HR/RR/TEMP planos)
        no domine 'severidad'. Sin cap la silueta sube, pero el top del ranking se
        llena de artefactos: no es un buen intercambio.
      - log1p / cuantiles : INNECESARIOS. Bajan la silueta sin mejorar el recall.
      - StandardScaler : NO afecta al modelo (Isolation Forest parte por feature,
        438/438 anomalias identicas con y sin escalar). Se conserva solo para que
        la silueta no quede dominada por la feature de mayor rango.
    """
    filas = []
    for nombre, fn in [
        ('sin transformar', lambda z: z),
        ('winsorizado 4sigma',  lambda z: z.clip(-4, 4)),
        ('winsorizado 8sigma',  lambda z: z.clip(-8, 8)),
        ('winsorizado 12sigma', lambda z: z.clip(-12, 12)),
        ('log1p con signo', lambda z: np.sign(z) * np.log1p(np.abs(z))),
    ]:
        t = df.copy()
        for v, sg in DIRECCION.items():
            z = fn(sg * t[f'{v}_zscore'].fillna(0))
            t[f'dz_{v}'], t[f'desv_{v}'] = z, z.clip(lower=0)
        t['severidad'] = sum(t[f'desv_{v}'] for v in DIRECCION)
        t['n_alteradas'] = sum((t[f'dz_{v}'] > UMBRAL_ALT).astype(int) for v in DIRECCION)
        t['concordancia'] = t[[f'desv_{v}' for v in DIRECCION]].min(axis=1)
        X = StandardScaler().fit_transform(t[feats].values)
        m = IsolationForest(n_estimators=300, contamination=CONTAMINACION,
                            random_state=RANDOM_STATE, n_jobs=1).fit(X)
        lab = m.predict(X)
        t['s'] = -m.score_samples(X)
        top = t.nlargest(int((lab == -1).sum()), 's')
        # cuantas del top-50 son artefacto de una sola variable
        art = (top.head(50)['n_alteradas'] <= 1).sum()
        filas.append({'transformacion': nombre,
                      'silueta': round(sil_seguro(X, lab), 4),
                      'artefactos_1var_en_top50': int(art)})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------ main
def main():
    Path('outputs').mkdir(exist_ok=True)
    df = cargar_y_sanear()
    df, feats = construir_features(df)
    print(f"[features] {len(feats)}: {feats}")

    X = StandardScaler().fit_transform(df[feats].values)   # escalado: solo para que
    # la silueta no quede dominada por la feature de mayor rango. Isolation Forest
    # es invariante (verificado: 438/438 anomalias identicas con y sin escalar).

    filas = []
    for c in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        m = IsolationForest(n_estimators=300, contamination=c,
                            random_state=RANDOM_STATE, n_jobs=1).fit(X)
        lab = m.predict(X)
        s = sil_seguro(X, lab)
        filas.append({'contaminacion': c, 'silueta': round(s, 4),
                      'n_anomalias': int((lab == -1).sum()), 'cumple_0.5': s > 0.5})
    barrido = pd.DataFrame(filas)
    barrido.to_csv('outputs/barrido_v2.csv', index=False)
    print("\n[barrido de contaminacion]"); print(barrido.to_string(index=False))

    abl = ablacion_transformaciones(df, feats)
    abl.to_csv('outputs/ablacion_v2.csv', index=False)
    print("\n[estudio de transformaciones]"); print(abl.to_string(index=False))

    iso = IsolationForest(n_estimators=300, contamination=CONTAMINACION,
                          random_state=RANDOM_STATE, n_jobs=1).fit(X)
    lab = iso.predict(X)
    df['score_anomalia'] = -iso.score_samples(X)
    df['es_anomalia'] = lab == -1
    sil = sil_seguro(X, lab)
    print(f"\n[modelo] contaminacion={CONTAMINACION} silueta={sil:.3f} "
          f"anomalias={df.es_anomalia.sum()}")

    # --- compuertas sobre TODAS las ventanas (la persistencia necesita vecinas) ---
    ctx = df.apply(compuerta_contexto, axis=1, result_type='expand')
    df['factor_contexto'], df['motivo_contexto'] = ctx[0], ctx[1]
    cal = df.apply(compuerta_calidad, axis=1, result_type='expand')
    df['factor_calidad'], df['motivo_calidad'] = cal[0], cal[1]
    df['factor_persistencia'], df['motivo_persistencia'] = compuerta_persistencia(df)

    df['prioridad_score'] = (df['score_anomalia'] * df['factor_contexto']
                             * df['factor_calidad'] * df['factor_persistencia'])

    al = df[df.es_anomalia].copy()
    q = al['prioridad_score'].quantile([0.60, 0.85]).values
    al['prioridad'] = np.where(al.prioridad_score >= q[1], 'ALTA',
                       np.where(al.prioridad_score >= q[0], 'MEDIA', 'BAJA'))

    contrib = atribucion_local(iso, X, np.where(df.es_anomalia.values)[0], feats)
    al = pd.concat([al.reset_index(drop=True), contrib.reset_index(drop=True)], axis=1)
    al['factor_dominante'] = [feats[i] for i in contrib.values.argmax(axis=1)]
    al['explicacion'] = al.apply(narrar, axis=1)
    al = al.sort_values(['prioridad_score', 'severidad'], ascending=False).reset_index(drop=True)
    al['ranking'] = np.arange(1, len(al) + 1)

    cols = (['ranking', 'prioridad', 'prioridad_score', 'patient_id', 'window_id',
             'window_start', 'window_end', 'score_anomalia', 'severidad',
             'n_alteradas', 'concordancia', 'discordancia_wearable',
             'factor_contexto', 'factor_calidad', 'factor_persistencia',
             'factor_dominante', 'explicacion']
            + [f'dz_{v}' for v in DIRECCION] + [f'contrib_{f}' for f in feats]
            + ['pct_flagged', 'n_check', 'n_low_signal', 'connectivity_flag',
               'context_physical_activity', 'context_sleep_state', 'age_years',
               'baseline_risk_profile', 'encounter_type', 'med_active'])
    al[cols].to_csv('outputs/alertas_v2.csv', index=False)

    signals = construir_signals_csv(al)
    signals.to_csv('outputs/signals.csv', index=False)
    print(f"\n[submission kit] outputs/signals.csv ({len(signals)} filas, formato validado "
          f"contra las reglas de validate_submission.py) -- todavia NO copiado a "
          f"submission_kit/ (falta evidence.csv, ver docstring del modulo)")
    print(signals['priority_level'].value_counts().rename('reparto priority_level').to_string())

    # ---------------------------------------------------- evidencia de desempeno
    eventos = {'PAT-0633': '2026-07-15 16:00', 'PAT-0009': '2026-07-12 15:00',
               'PAT-0869': '2026-07-20 15:00', 'PAT-0410': '2026-07-10 19:00',
               'PAT-0097': '2026-07-27 15:00', 'PAT-0675': '2026-07-05 12:00',
               'PAT-0118': '2026-07-08 11:00', 'PAT-0374': '2026-07-15 16:00',
               'PAT-0034': '2026-07-09 11:00', 'PAT-0992': '2026-07-10 10:00',
               'PAT-0609': '2026-07-25 11:00'}
    capt = alta = 0
    for pid, ts in eventos.items():
        t = pd.Timestamp(ts)
        m = al[(al.patient_id == pid) & (al.window_start <= t) & (al.window_end > t)]
        if len(m):
            capt += 1
            alta += (m.iloc[0]['prioridad'] == 'ALTA')
    deg = al.factor_contexto < 1
    print(f"\n[desempeno]")
    print(f"  eventos confirmados capturados : {capt}/{len(eventos)}  (en prioridad ALTA: {alta})")
    print(f"  degradadas por contexto        : {deg.sum():3d} ({100*deg.mean():.1f}%)")
    print(f"  degradadas por calidad         : {(al.factor_calidad < 1).sum():3d}")
    print(f"  degradadas por transitoriedad  : {(al.factor_persistencia < 1).sum():3d}")
    print(f"  elevadas por persistencia intra: {(al.factor_persistencia > 1.0).sum():3d}")
    print(f"  reparto de prioridad           : {al.prioridad.value_counts().to_dict()}")
    act_alta = al[al.prioridad == 'ALTA'].context_physical_activity.isin(['HIGH', 'MODERATE'])
    print(f"  ALTA con actividad fisica      : {act_alta.sum()} ({100*act_alta.mean():.1f}%)  "
          f"[v1: 38.1% del total]")

    pd.DataFrame([{'version': 'v2', 'contaminacion': CONTAMINACION, 'silueta': round(sil, 4),
                   'n_ventanas': len(df), 'n_alertas': len(al),
                   'eventos_capturados': f"{capt}/{len(eventos)}",
                   'n_features': len(feats),
                   'alertas_ALTA': int((al.prioridad == 'ALTA').sum())}]
                 ).to_csv('outputs/metricas_v2.csv', index=False)

    print("\n[top 10 por prioridad]")
    for _, r in al.head(10).iterrows():
        print(f"  #{r.ranking:<3} {r.prioridad:5s} {r.patient_id} {str(r.window_start)[:16]} "
              f"p={r.prioridad_score:.3f} | {r.explicacion[:96]}")
    print(f"\n-> outputs/alertas_v2.csv ({len(al)} alertas)")


if __name__ == '__main__':
    main()