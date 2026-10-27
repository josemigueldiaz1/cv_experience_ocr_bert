"""
Etapa 6 — Evaluacion, comparacion de modelos y analisis de errores.

Todo se calcula sobre el MISMO conjunto de test (que ningun modelo vio al entrenar ni al
ajustar hiperparametros) y la variante de texto principal, salvo donde se indica.

Salidas (output/tables/ y output/metrics/)
------------------------------------------
- model_comparison.csv      tabla del enunciado: MAE, RMSE, Acc ±1/2/3, bandas, R², tiempos
- bert_gain_bootstrap.csv   reduccion de MAE de BERT frente a cada baseline, IC 95 % pareado
- mae_by_band.csv, mae_by_occupation.csv, bias_by_band.csv
- confusion_<modelo>.csv    matriz de confusion por bandas
- bert_top_errors.csv       los 10 peores errores de BERT con su causa
- probe_results.csv         Pregunta 1: ¿la prediccion se mueve como la de un reclutador?
- ocr_noise_cost.csv        Pregunta 2: MAE con texto de referencia / OCR v6 / OCR rapido
- sensitivity.csv           Pregunta 3: MAE de cada modelo con cada variante de limpieza
- fairness.csv              Pregunta 4: MAE y error medio de BERT por sexo y por raza
- metrics/evaluation.json   resumen para el README y la app

Uso
---
    python src/evaluation.py
    python src/evaluation.py --bootstrap 5000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (CAP, METRICS, MODELS_BERT, NOMBRES_BANDAS, NOMBRES_MODELOS,  # noqa: E402
                       PREDICTIONS, PROCESSED, RAW, SEED, TABLES, VARIANTE_PRINCIPAL,
                       VARIANTES, banda, configurar_logging, guardar_json, leer_json,
                       leer_predicciones, recortar)

BASELINES = ["median", "tfidf", "embeddings"]
MODELOS = ["median", "tfidf", "embeddings", "bert", "rules"]

log = None


# ======================================================================== metricas
def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), recortar(p)
    e = p - y
    by, bp = [banda(v) for v in y], [banda(v) for v in p]
    return {
        "MAE (years)": float(np.mean(np.abs(e))),
        "RMSE (years)": float(np.sqrt(np.mean(e ** 2))),
        "Acc ±1": float(np.mean(np.abs(e) <= 1)),
        "Acc ±2": float(np.mean(np.abs(e) <= 2)),
        "Acc ±3": float(np.mean(np.abs(e) <= 3)),
        "Band accuracy": float(accuracy_score(by, bp)),
        "Band macro-F1": float(f1_score(by, bp, labels=NOMBRES_BANDAS, average="macro",
                                        zero_division=0)),
        "R²": float(r2_score(y, p)),
    }


def bootstrap_ganancia(y, p_bert, p_base, n: int, seed: int = SEED) -> dict:
    """
    Bootstrap PAREADO: en cada remuestreo se eligen los mismos CVs para los dos modelos,
    asi la variabilidad que se mide es la de la diferencia y no la de cada MAE por
    separado. Ganancia = MAE(baseline) - MAE(BERT); positiva = BERT se equivoca menos.
    """
    y, a, b = (np.asarray(v, float) for v in (y, p_bert, p_base))
    err_a, err_b = np.abs(a - y), np.abs(b - y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    difs = err_b[idx].mean(axis=1) - err_a[idx].mean(axis=1)
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {"gain": float(err_b.mean() - err_a.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "share_resamples_bert_not_better": float(np.mean(difs <= 0)),
            "significant": bool(lo > 0 or hi < 0)}


def _test(modelo: str, variante: str = VARIANTE_PRINCIPAL) -> pd.DataFrame | None:
    df = leer_predicciones(modelo, variante)
    return None if df is None else df[df.split == "test"].sort_values("id").reset_index(drop=True)


def _info(modelo: str, variante: str = VARIANTE_PRINCIPAL) -> dict:
    ruta = PREDICTIONS / f"{modelo}__{variante}.json"
    return leer_json(ruta) if ruta.exists() else {}


# ============================================================= tabla y desgloses
def tabla_comparativa() -> pd.DataFrame:
    filas = []
    for m in MODELOS:
        t = _test(m)
        if t is None:
            log.warning("  sin predicciones de %s", m)
            continue
        info = _info(m)
        filas.append({"Model": NOMBRES_MODELOS[m], **metricas(t.y, t.pred),
                      "Training time": _formato_tiempo(info.get("train_seconds", 0)),
                      "Inference time per CV": f"{1000 * info.get('inference_seconds_per_cv', 0):.2f} ms",
                      "key": m})
    df = pd.DataFrame(filas)
    df.drop(columns="key").round(4).to_csv(TABLES / "model_comparison.csv", index=False)
    return df


def _formato_tiempo(s: float) -> str:
    return f"{s:.2f} s" if s < 120 else f"{s / 60:.1f} min"


def desgloses(meta: pd.DataFrame) -> None:
    filas_banda, filas_ocup, sesgo = [], [], []
    for m in MODELOS:
        t = _test(m)
        if t is None:
            continue
        t = t.merge(meta[["id", "job"]], on="id")
        t["band"] = t.y.map(banda)
        t["abs_err"] = (t.pred - t.y).abs()
        t["err"] = t.pred - t.y
        for b, g in t.groupby("band"):
            filas_banda.append({"model": NOMBRES_MODELOS[m], "band": b, "n": len(g),
                                "mae": g.abs_err.mean()})
            sesgo.append({"model": NOMBRES_MODELOS[m], "band": b, "n": len(g),
                          "mean_error": g.err.mean(), "mean_true": g.y.mean(),
                          "mean_pred": g.pred.mean()})
        for j, g in t.groupby("job"):
            filas_ocup.append({"model": NOMBRES_MODELOS[m], "occupation": j, "n": len(g),
                               "mae": g.abs_err.mean()})
        cm = confusion_matrix([banda(v) for v in t.y], [banda(v) for v in t.pred],
                              labels=NOMBRES_BANDAS)
        pd.DataFrame(cm, index=[f"true {b}" for b in NOMBRES_BANDAS],
                     columns=[f"pred {b}" for b in NOMBRES_BANDAS]).to_csv(
            TABLES / f"confusion_{m}.csv")
    orden = {b: i for i, b in enumerate(NOMBRES_BANDAS)}
    pd.DataFrame(filas_banda).sort_values(["model", "band"], key=lambda s: s.map(orden)
                                          if s.name == "band" else s).round(4).to_csv(
        TABLES / "mae_by_band.csv", index=False)
    pd.DataFrame(filas_ocup).round(4).to_csv(TABLES / "mae_by_occupation.csv", index=False)
    pd.DataFrame(sesgo).round(4).to_csv(TABLES / "bias_by_band.csv", index=False)


def ganancias_bert(n_boot: int) -> pd.DataFrame:
    bert = _test("bert")
    filas = []
    for m in BASELINES + ["rules"]:
        t = _test(m)
        if t is None or bert is None:
            continue
        assert list(t.id) == list(bert.id), "los modelos no comparten el test"
        r = bootstrap_ganancia(bert.y, bert.pred, t.pred, n_boot)
        filas.append({"baseline": NOMBRES_MODELOS[m], "mae_baseline": float(np.mean(np.abs(t.pred - t.y))),
                      "mae_bert": float(np.mean(np.abs(bert.pred - bert.y))), **r,
                      "resamples": n_boot})
    df = pd.DataFrame(filas)
    df.round(4).to_csv(TABLES / "bert_gain_bootstrap.csv", index=False)
    return df


# ===================================================================== errores de BERT
def _intervalos(texto: str):
    from src.baseline_rules import intervalos
    return sorted(intervalos(texto))


def causa_error(fila) -> str:
    """
    Clasificacion de un error de BERT en las categorias del enunciado. Se comprueba en
    orden: primero lo que viene del OCR o del texto, despues lo que es del modelo.
    """
    from src.ocr_quality import rangos, rangos_cortados
    from src.baseline_rules import anios_experiencia

    if rangos(fila.text) != rangos(fila.reference):
        if rangos_cortados(fila.text):
            return CAUSAS["cortada"]
        return CAUSAS["ocr"]
    iv = _intervalos(fila.text)
    if not iv:
        return CAUSAS["sin_fechas"]
    if fila.y >= CAP:
        # La experiencia real pasa de 20 y se recorto: es el extremo menos visto en train.
        return CAUSAS["tope"]
    if fila.get("truncated", False):
        return CAUSAS["truncado"]
    # Orden de lectura: los rangos estan, pero alguno quedo separado de su puesto.
    lineas = fila.text.split("\n")
    if any(re.fullmatch(r"\s*\d{4}\.\d{2}-\d{4}\.\d{2}\s*", l) for l in lineas):
        return CAUSAS["orden"]
    solape = any(b[0] < a[1] for a, b in zip(iv, iv[1:]))
    if solape:
        return CAUSAS["solape"]
    hueco = sum(max(0.0, b[0] - a[1]) for a, b in zip(iv, iv[1:]))
    if hueco >= 1.0:
        return CAUSAS["hueco"]
    # Si la suma de fechas acierta, la informacion estaba completa y bien leida: el
    # fallo es del modelo (no sabe hacer la aritmetica en este rango de valores).
    if abs(anios_experiencia(fila.text) - fila.y) < 0.25:
        return CAUSAS["cuenta"]
    return CAUSAS["otro"]


# Las categorias del enunciado, en el texto que aparece en las tablas y en la app.
CAUSAS = {
    "cortada": "fecha cortada en la propia imagen (dataset)",
    "ocr": "el OCR leyó mal una fecha",
    "sin_fechas": "faltan fechas",
    "tope": "experiencia en el tope de 20 años (poco vista al entrenar)",
    "truncado": "truncado",
    "orden": "orden de lectura incorrecto",
    "solape": "empleos solapados",
    "hueco": "hueco entre empleos",
    "cuenta": "otro: fechas bien leídas, el modelo falló la cuenta",
    "otro": "otro",
}


def peores_errores(textos: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    bert = _test("bert")
    resumen = leer_json(MODELS_BERT / f"bert__{VARIANTE_PRINCIPAL}" / "training_summary.json")
    t = bert.merge(textos[["id", "job", VARIANTE_PRINCIPAL, "reference"]], on="id")
    t = t.rename(columns={VARIANTE_PRINCIPAL: "text"})
    t["truncated"] = False if resumen.get("cvs_truncated", 0) == 0 else None
    t["error"] = t.pred - t.y
    peores = t.reindex(t.error.abs().sort_values(ascending=False).index).head(k).copy()
    peores["n_jobs"] = peores.text.map(lambda s: len(_intervalos(s)))
    peores["gap_years"] = peores.text.map(
        lambda s: sum(max(0.0, b[0] - a[1]) for a, b in zip(_intervalos(s), _intervalos(s)[1:])))
    peores["cause"] = peores.apply(causa_error, axis=1)
    cols = ["id", "job", "y", "pred", "error", "n_jobs", "gap_years", "cause", "text"]
    peores[cols].round(3).to_csv(TABLES / "bert_top_errors.csv", index=False)
    return peores[cols]


# ======================================================================= predictores
def predictores(variante: str = VARIANTE_PRINCIPAL) -> dict:
    """Carga cada modelo entrenado como una funcion `predecir(textos)`."""
    from src import baseline_embeddings, baseline_median, baseline_rules, baseline_tfidf, train_bert
    return {"median": baseline_median.cargar(variante), "tfidf": baseline_tfidf.cargar(variante),
            "embeddings": baseline_embeddings.cargar(variante),
            "bert": train_bert.cargar(variante), "rules": baseline_rules.cargar(variante)}


# =========================================================== Pregunta 1: la sonda
_ANIO4 = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def desplazar_fechas(texto: str, anios: int = 5) -> str:
    """Suma `anios` a TODOS los anios del CV (empleos y estudios)."""
    return _ANIO4.sub(lambda m: str(int(m.group(1)) + anios), texto)


def quitar_empleo(texto: str) -> str:
    """Borra el empleo mas antiguo: su renglon con fechas y sus vinetas."""
    lineas = texto.split("\n")
    con_fecha = [i for i, l in enumerate(lineas) if re.search(r"\d{4}\.\d{2}-\d{4}\.\d{2}", l)]
    if len(con_fecha) < 2:
        return texto
    i = con_fecha[-1]
    j = i + 1
    while j < len(lineas) and lineas[j].lstrip().startswith("-"):
        j += 1
    return "\n".join(lineas[:i] + lineas[j:])


def adelantar_inicio(texto: str, anios: int = 3) -> str:
    """El empleo mas antiguo empieza `anios` antes: un reclutador sumaria `anios`."""
    rangos = list(re.finditer(r"(\d{4})(\.\d{2}-\d{4}\.\d{2})", texto))
    if not rangos:
        return texto
    m = rangos[-1]
    return texto[:m.start()] + str(int(m.group(1)) - anios) + m.group(2) + texto[m.end():]


def sonda(pred: dict, textos: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    20 CVs de test con al menos dos empleos. A cada uno se le aplican tres cambios con
    respuesta conocida, y se compara cuanto se mueve cada modelo con cuanto se moveria
    un reclutador (la suma de fechas, que en el texto original acierta con ±0.02 anios):

    - +5 anios a todas las fechas: la experiencia NO cambia (las duraciones son iguales).
    - Borrar el empleo mas antiguo: la experiencia baja en lo que duro ese empleo.
    - El empleo mas antiguo empieza 3 anios antes: la experiencia sube 3 anios.
    """
    from src.baseline_rules import anios_experiencia

    test_ids = set(_test("bert").id)
    cand = textos[textos.id.isin(test_ids)].copy()
    cand = cand[cand[VARIANTE_PRINCIPAL].map(lambda s: len(_intervalos(s)) >= 2)]
    muestra = cand.sample(n, random_state=SEED)
    orig = muestra[VARIANTE_PRINCIPAL].tolist()
    cambios = {SONDAS[0]: [desplazar_fechas(t) for t in orig],
               SONDAS[1]: [quitar_empleo(t) for t in orig],
               SONDAS[2]: [adelantar_inicio(t) for t in orig]}
    filas = []
    base = {m: f(orig) for m, f in pred.items()}
    for cambio, nuevos in cambios.items():
        esperado = [anios_experiencia(b) - anios_experiencia(a) for a, b in zip(orig, nuevos)]
        if cambio == SONDAS[0]:
            esperado = [0.0] * len(orig)
        for m, f in pred.items():
            delta = f(nuevos) - base[m]
            for k, cv in enumerate(muestra.id):
                filas.append({"probe": cambio, "model": NOMBRES_MODELOS[m], "id": cv,
                              "expected_change": esperado[k], "model_change": float(delta[k])})
    df = pd.DataFrame(filas)
    df.round(4).to_csv(TABLES / "probe_results_per_cv.csv", index=False)
    resumen = (df.assign(abs_dev=(df.model_change - df.expected_change).abs(),
                         same_direction=np.sign(df.model_change.round(2))
                         == np.sign(df.expected_change.round(2)))
               .groupby(["probe", "model"])
               .agg(expected_change=("expected_change", "mean"),
                    model_change=("model_change", "mean"),
                    mean_abs_deviation=("abs_dev", "mean"),
                    same_direction=("same_direction", "mean")).reset_index())
    resumen.round(3).to_csv(TABLES / "probe_results.csv", index=False)
    return resumen


# Nombres de las tres pruebas de la sonda, en el orden en que se aplican.
SONDAS = ("todas las fechas +5 años", "borrar el empleo más antiguo",
          "el empleo más antiguo empieza 3 años antes")


# =============================================================== Pregunta 2: ruido
# Las tres versiones del texto del test y la columna de `texts.parquet` de cada una.
FUENTES_RUIDO = {"referencia (sin ruido de OCR)": "reference",
                 "PaddleOCR v6 (el del entrenamiento)": VARIANTE_PRINCIPAL,
                 "PaddleOCR v5 mobile (con ruido)": "mobile"}


def costo_ruido(pred: dict, textos: pd.DataFrame) -> pd.DataFrame:
    test = _test("bert")[["id", "y"]].merge(textos, on=["id"], how="left",
                                            suffixes=("", "_t"))
    fuentes = FUENTES_RUIDO
    filas = []
    for m, f in pred.items():
        fila = {"model": NOMBRES_MODELOS[m]}
        for etiqueta, col in fuentes.items():
            if col not in test or test[col].isna().any():
                fila[etiqueta] = np.nan
                continue
            fila[etiqueta] = float(np.mean(np.abs(f(test[col].tolist()) - test.y)))
        fila["brecha v6 - referencia"] = fila[list(fuentes)[1]] - fila[list(fuentes)[0]]
        fila["brecha con ruido - referencia"] = fila[list(fuentes)[2]] - fila[list(fuentes)[0]]
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.round(4).to_csv(TABLES / "ocr_noise_cost.csv", index=False)
    return df


# =========================================================== Pregunta 3: sensibilidad
def sensibilidad() -> pd.DataFrame:
    filas = []
    for m in MODELOS:
        for v in VARIANTES:
            t = _test(m, v)
            if t is not None:
                filas.append({"model": NOMBRES_MODELOS[m], "setting": f"texto: {v}",
                              "test_mae": float(np.mean(np.abs(t.pred - t.y)))})
    for clave, m in (("tfidf_nodigits", "tfidf"), ("bert_huber", "bert")):
        t = _test(clave)
        if t is not None:
            ajuste = "sin dígitos" if "digits" in clave else "pérdida Huber"
            filas.append({"model": NOMBRES_MODELOS[m], "setting": f"texto: clean, {ajuste}",
                          "test_mae": float(np.mean(np.abs(t.pred - t.y)))})
    df = pd.DataFrame(filas)
    df.round(4).to_csv(TABLES / "sensitivity.csv", index=False)
    return df


# ============================================================ Pregunta 4: equidad
def equidad(meta: pd.DataFrame, n_boot: int, textos: pd.DataFrame) -> pd.DataFrame:
    """
    MAE y error medio (prediccion − real) por sexo y por raza, con IC 95 % bootstrap.

    Se anaden dos descriptores de la trayectoria de cada grupo (proporcion de CVs con un
    hueco de 1 anio o mas entre empleos, y numero medio de empleos) para poder explicar
    diferencias: sexo y raza NO son entradas, pero PopResume genera trayectorias a partir
    de estadisticas reales, que si difieren entre grupos.
    """
    filas = []
    rng = np.random.default_rng(SEED)
    huecos = textos.set_index("id")[VARIANTE_PRINCIPAL].map(
        lambda s: sum(max(0.0, b[0] - a[1]) for a, b in zip(_intervalos(s), _intervalos(s)[1:])))
    n_empleos = textos.set_index("id")[VARIANTE_PRINCIPAL].map(lambda s: len(_intervalos(s)))
    for m in ("bert", "tfidf", "rules"):
        t = _test(m)
        if t is None:
            continue
        t = t.merge(meta[["id", "sex", "race"]], on="id")
        t["err"] = t.pred - t.y
        t["gap"] = t.id.map(huecos)
        t["jobs"] = t.id.map(n_empleos)
        for atributo in ("sex", "race"):
            for grupo, g in t.groupby(atributo):
                e = g.err.values
                boots = rng.choice(e, size=(n_boot, len(e))).mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                filas.append({"model": NOMBRES_MODELOS[m], "attribute": atributo,
                              "group": grupo, "n": len(g), "true_mean": g.y.mean(),
                              "pred_mean": g.pred.mean(), "mae": np.abs(e).mean(),
                              "mean_error": e.mean(), "mean_error_ci_low": lo,
                              "mean_error_ci_high": hi,
                              "share_with_career_gap": float((g.gap >= 1).mean()),
                              "mean_jobs": float(g.jobs.mean())})
    df = pd.DataFrame(filas)
    df.round(4).to_csv(TABLES / "fairness.csv", index=False)
    return df


# ============================================================================== main
def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bootstrap", type=int, default=2000, help="Remuestreos (>= 1000)")
    p.add_argument("--sin-predictores", action="store_true",
                   help="Omite lo que necesita cargar los modelos (sonda y ruido)")
    args = p.parse_args()
    log = configurar_logging("evaluation")

    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    textos = pd.read_parquet(PROCESSED / "texts.parquet")

    comp = tabla_comparativa()
    log.info("Tabla comparativa (test, n=%d):", len(_test("median")))
    for _, r in comp.iterrows():
        log.info("  %-22s MAE %.3f | RMSE %.3f | ±2 %.1f%% | bandas %.1f%% | R² %.3f",
                 r.Model, r["MAE (years)"], r["RMSE (years)"], 100 * r["Acc ±2"],
                 100 * r["Band accuracy"], r["R²"])
    desgloses(meta)
    gan = ganancias_bert(args.bootstrap)
    for _, r in gan.iterrows():
        log.info("  BERT vs %-22s ganancia %.3f anios  IC95 [%.3f, %.3f]  %s", r.baseline,
                 r.gain, r.ci_low, r.ci_high, "significativa" if r.significant else "NO significativa")
    peores = peores_errores(textos)
    log.info("  Causas de los 10 peores errores de BERT: %s", peores.cause.value_counts().to_dict())
    sens = sensibilidad()
    fair = equidad(meta, args.bootstrap, textos)

    resumen = {"comparison": comp.drop(columns="key").to_dict(orient="records"),
               "bert_gain": gan.to_dict(orient="records"),
               "top_error_causes": peores.cause.value_counts().to_dict(),
               "sensitivity": sens.to_dict(orient="records"),
               "fairness_bert": fair[fair.model == NOMBRES_MODELOS["bert"]].to_dict(orient="records")}

    if not args.sin_predictores:
        pred = predictores()
        son = sonda(pred, textos)
        for _, r in son.iterrows():
            log.info("  sonda %-28s %-22s esperado %+.2f  modelo %+.2f", r.probe, r.model,
                     r.expected_change, r.model_change)
        ruido = costo_ruido(pred, textos)
        for _, r in ruido.iterrows():
            log.info("  ruido %-22s ref %.3f | v6 %.3f | movil %.3f", r.model,
                     *(r[c] for c in FUENTES_RUIDO))
        resumen["probe"] = son.to_dict(orient="records")
        resumen["ocr_noise_cost"] = ruido.to_dict(orient="records")

    guardar_json(resumen, METRICS / "evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
