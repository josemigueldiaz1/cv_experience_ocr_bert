"""
Etapa 2a — Calidad del OCR frente al texto de referencia.

Que mide
--------
- CER y WER (con `jiwer`), despues de normalizar el texto del OCR y el de referencia
  EXACTAMENTE de la misma manera.
- Anios leidos correctamente: de todos los anios (19xx/20xx) que hay en la referencia,
  que fraccion aparece en el OCR. Se cuenta como multiconjunto: si "2019" aparece dos
  veces en la referencia, el OCR tiene que tenerlo dos veces.
- Rangos de fechas leidos correctamente: que fraccion de los rangos completos
  "2017.12-2020.07" aparece intacta en el OCR. Es la metrica que mas importa en esta
  tarea, porque la experiencia es la suma de esos rangos (ver README): un
  digito mal leido, o un guion perdido, cambia la respuesta.

El texto del OCR se mide en orden de lectura reconstruido y SIN limpiar (variante `raw`):
lo que se quiere saber es cuanto se equivoca PaddleOCR, no cuanto arregla la limpieza.

Uso
---
    python src/ocr_quality.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Guiones que aparecen en los CVs o que el OCR puede devolver: en dash, em dash, menos...
GUIONES = "‐‑‒–—―−"
_TABLA_GUIONES = str.maketrans({c: "-" for c in GUIONES})
_VINIETA = re.compile(r"^\s*[-•·*]+\s*", re.M)
_ANIO = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_RANGO = re.compile(r"(?<!\d)((?:19|20)\d{2})\.(\d{2})\s*-\s*((?:19|20)\d{2})\.(\d{2})(?!\d)")


def normalizar(texto: str) -> str:
    """
    Normalizacion comun para comparar OCR y referencia.

    Lo que se iguala es lo que NO es un error de lectura: el estilo de vineta (la imagen
    dibuja "•" donde el texto de referencia tiene "-"), el tipo de guion, los espacios
    alrededor de "|" y las mayusculas. Lo que se conserva es todo lo demas, en particular
    cada digito y cada punto de las fechas.
    """
    t = texto.translate(_TABLA_GUIONES)
    t = _VINIETA.sub("", t)
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip().lower()


def anios(texto: str) -> Counter:
    return Counter(_ANIO.findall(texto))


def rangos(texto: str) -> Counter:
    """Rangos 'YYYY.MM-YYYY.MM' como cadenas canonicas."""
    t = texto.translate(_TABLA_GUIONES)
    return Counter(f"{a}.{b}-{c}.{d}" for a, b, c, d in _RANGO.findall(t))


# Rango cuyo final esta incompleto ("2020.02-2023." o "2021.04-2022.0"). En PopResume esto
# NO es un error del OCR: el generador del dataset corta las lineas demasiado largas al
# dibujar la imagen ("Elementary Middle School Teachers, <escuela>, 2020.02–2023."), y el
# OCR lee fielmente lo que hay. El lookahead (?![\d.]) evita contar los rangos completos.
_RANGO_CORTADO = re.compile(r"(?<!\d)(?:19|20)\d{2}\.\d{2}\s*-\s*(?:19|20)\d{2}(?:\.\d?)?(?![\d.])")


def rangos_cortados(texto: str) -> int:
    return len(_RANGO_CORTADO.findall(texto.translate(_TABLA_GUIONES)))


def recall_multiconjunto(ref: Counter, hyp: Counter) -> float | None:
    total = sum(ref.values())
    if total == 0:
        return None
    return sum(min(n, hyp.get(k, 0)) for k, n in ref.items()) / total


def medir(ocr_texto: str, ref_texto: str) -> dict:
    """Todas las metricas de calidad de un CV."""
    import jiwer

    r, h = normalizar(ref_texto), normalizar(ocr_texto)
    return {
        "cer": jiwer.cer(r, h),
        "wer": jiwer.wer(r, h),
        "years_ok": recall_multiconjunto(anios(ref_texto), anios(ocr_texto)),
        "ranges_ok": recall_multiconjunto(rangos(ref_texto), rangos(ocr_texto)),
        "n_years": sum(anios(ref_texto).values()),
        "n_ranges": sum(rangos(ref_texto).values()),
        "ranges_cut_in_image": rangos_cortados(ocr_texto),
    }


# ================================================================================ main
def main() -> int:
    from src.preprocessing import cargar_ocr, texto_en_orden
    from src.utils import METRICS, OCR, RAW, TABLES, configurar_logging, guardar_json

    log = configurar_logging("ocr_quality")
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref = pd.read_parquet(RAW / "reference_text.parquet").set_index("id").text

    filas = []
    faltan = 0
    for _, m in meta.iterrows():
        doc = cargar_ocr(m.id)
        if doc is None:
            faltan += 1
            continue
        texto = texto_en_orden(doc["lines"])
        conf = [l["score"] for l in doc["lines"]]
        filas.append({"id": m.id, "job": m.job,
                      "mean_conf": sum(conf) / len(conf) if conf else 0.0,
                      "min_conf": min(conf) if conf else 0.0,
                      **medir(texto, ref[m.id])})
    if faltan:
        log.warning("%d CVs sin salida de OCR en %s", faltan, OCR)

    df = pd.DataFrame(filas)
    df.to_csv(TABLES / "ocr_quality_per_cv.csv", index=False)

    por_ocup = df.groupby("job").agg(
        cvs=("id", "size"), cer_mean=("cer", "mean"), cer_median=("cer", "median"),
        wer_mean=("wer", "mean"), years_ok=("years_ok", "mean"),
        ranges_ok=("ranges_ok", "mean"), mean_conf=("mean_conf", "mean"),
        perfect_ranges=("ranges_ok", lambda s: (s == 1).mean()),
    ).reset_index()
    total = pd.DataFrame([{
        "job": "ALL", "cvs": len(df), "cer_mean": df.cer.mean(), "cer_median": df.cer.median(),
        "wer_mean": df.wer.mean(), "years_ok": df.years_ok.mean(),
        "ranges_ok": df.ranges_ok.mean(), "mean_conf": df.mean_conf.mean(),
        "perfect_ranges": (df.ranges_ok == 1).mean(),
    }])
    resumen = pd.concat([por_ocup, total], ignore_index=True)
    resumen.round(4).to_csv(TABLES / "ocr_quality_by_occupation.csv", index=False)

    # Anios: se agregan sobre el total de anios, no como media de proporciones por CV.
    tot_anios = int(df.n_years.sum())
    ok_anios = float((df.years_ok * df.n_years).sum())
    tot_rangos = int(df.n_ranges.sum())
    ok_rangos = float((df.ranges_ok.fillna(0) * df.n_ranges).sum())
    cortados = int(df.ranges_cut_in_image.sum())
    mal_leidos = tot_rangos - ok_rangos - cortados
    guardar_json({
        "cvs": len(df), "cer_mean": df.cer.mean(), "cer_median": df.cer.median(),
        "cer_p95": df.cer.quantile(.95), "wer_mean": df.wer.mean(),
        "wer_median": df.wer.median(), "years_total": tot_anios,
        "years_read_ok_pct": 100 * ok_anios / tot_anios,
        "ranges_total": tot_rangos, "ranges_read_ok_pct": 100 * ok_rangos / tot_rangos,
        "ranges_cut_in_image": cortados,
        "cvs_with_range_cut_in_image": int((df.ranges_cut_in_image > 0).sum()),
        "ranges_misread_by_ocr": int(round(mal_leidos)),
        "cvs_with_all_ranges_ok_pct": 100 * float((df.ranges_ok == 1).mean()),
        "mean_confidence": df.mean_conf.mean(),
    }, METRICS / "ocr_quality.json")

    for _, r in resumen.iterrows():
        log.info("  %-35s CER %.4f | WER %.4f | anios %.2f%% | rangos %.2f%%", r.job,
                 r.cer_mean, r.wer_mean, 100 * r.years_ok, 100 * r.ranges_ok)
    log.info("Anios leidos bien: %.0f de %d (%.2f %%). Rangos intactos: %.0f de %d (%.2f %%)",
             ok_anios, tot_anios, 100 * ok_anios / tot_anios, ok_rangos, tot_rangos,
             100 * ok_rangos / tot_rangos)
    log.info("De los %d rangos no intactos: %d vienen cortados en la propia imagen (%d CVs) "
             "y %d son errores de lectura del OCR", tot_rangos - ok_rangos, cortados,
             int((df.ranges_cut_in_image > 0).sum()), round(mal_leidos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
