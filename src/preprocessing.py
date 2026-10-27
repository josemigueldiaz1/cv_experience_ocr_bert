"""
Etapa 2b — Orden de lectura, limpieza, datos personales y enmascarado de respuestas.

Que hace, por cada CV
---------------------
1. ORDEN DE LECTURA. Reconstruye el texto a partir de las cajas de PaddleOCR (ver
   `texto_en_orden`): detecta si hay dos columnas, agrupa las cajas en renglones y lee
   de arriba abajo y de izquierda a derecha.
2. DATOS PERSONALES. Borra nombre, email, telefono y ubicacion ("CA, USA"). El estado es
   un atributo prohibido como feature, igual que el nombre.
3. RESPUESTAS DIRECTAS. Enmascara frases que dan la respuesta sin razonar: experiencia
   total ("12 years of experience"), edad ("age 35", "35 years old") y fecha de
   nacimiento. Las fechas de los empleos y de los estudios NO se tocan: son la evidencia.
4. LIMPIEZA. Unifica guiones y vinetas, junta palabras cortadas con guion al final de
   renglon y normaliza espacios.

Variantes que se guardan (para la Pregunta 3)
---------------------------------------------
- `raw`        : orden de lectura + pasos 2 y 3 (obligatorios). Sin limpieza.
- `clean`      : `raw` + paso 4. Es la variante principal.
- `clean_conf` : `clean` quitando los renglones con confianza del OCR < umbral.
- `reference`  : el texto de referencia con los mismos pasos 2-4. Solo para la Pregunta 2
                 (evaluar modelos ya entrenados sobre texto sin ruido de OCR).

Uso
---
    python src/preprocessing.py
    python src/preprocessing.py --umbral-conf 0.9
    python src/preprocessing.py --revision 50       # exporta 50 textos al azar para leer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (METRICS, OCR, PROCESSED, RAW, SEED, TABLES, configurar_logging,  # noqa: E402
                       guardar_json, leer_json)

# Umbral de confianza para la variante `clean_conf` (output/tables/ocr_confidence_lines.csv).
# Con PP-OCRv6 la confianza minima de los 57,813 renglones es 0.936: un umbral "tipico"
# de 0.90 no quitaria nada. Con 0.98 se quita el 2 % de los renglones (el percentil 2).
# Al revisarlos, estan BIEN leidos (p. ej. "Registered Nurses, ..., 2017.06-2023.01" con
# 0.969): baja confianza no significa error. Por eso la variante principal NO filtra, y
# `clean_conf` sirve para medir en la Pregunta 3 cuanto dana filtrar evidencia correcta.
UMBRAL_CONF = 0.98


# ======================================================================= orden de lectura
OCR_RUIDOSO = OCR.parent / "ocr_mobile"


def cargar_ocr(cv_id: str, carpeta: Path = OCR) -> dict | None:
    ruta = carpeta / f"{cv_id}.json"
    return leer_json(ruta) if ruta.exists() else None


def _renglones(lineas: list[dict]) -> list[list[dict]]:
    """
    Agrupa cajas en renglones: dos cajas estan en el mismo renglon si se solapan en
    vertical mas de la mitad de la altura de la mas baja. Asi, si el OCR parte una linea
    en dos cajas ("Registered Nurses, X," | "2017.12-2020.07"), vuelven a juntarse en el
    orden correcto y el puesto no se separa de sus fechas.
    """
    orden = sorted(lineas, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    filas: list[list[dict]] = []
    for l in orden:
        y1, y2 = l["box"][1], l["box"][3]
        if filas:
            ult = filas[-1]
            fy1 = min(c["box"][1] for c in ult)
            fy2 = max(c["box"][3] for c in ult)
            solape = min(y2, fy2) - max(y1, fy1)
            if solape > 0.5 * min(y2 - y1, fy2 - fy1):
                ult.append(l)
                continue
        filas.append([l])
    return [sorted(f, key=lambda c: c["box"][0]) for f in filas]


def _canal_central(lineas: list[dict], ancho: float) -> float | None:
    """
    Busca un canal vertical vacio entre el 30 % y el 70 % del ancho que separe dos
    columnas de texto que conviven a la misma altura. Devuelve su x, o None si el CV es
    de una sola columna.
    """
    if len(lineas) < 6:
        return None
    mejor = None
    for x in np.linspace(0.30 * ancho, 0.70 * ancho, 41):
        cruzan = sum(1 for l in lineas if l["box"][0] < x < l["box"][2])
        izq = [l for l in lineas if l["box"][2] <= x]
        der = [l for l in lineas if l["box"][0] >= x]
        if len(izq) < 3 or len(der) < 3:
            continue
        # Las dos columnas tienen que compartir altura; si "der" esta toda debajo de "izq"
        # no son columnas, es un bloque sangrado.
        y_der = (min(l["box"][1] for l in der), max(l["box"][3] for l in der))
        conviven = sum(1 for l in izq if l["box"][1] < y_der[1] and l["box"][3] > y_der[0])
        if conviven < 3 or cruzan > 0.2 * len(lineas):
            continue
        if mejor is None or cruzan < mejor[0]:
            mejor = (cruzan, x)
    return None if mejor is None else float(mejor[1])


def texto_en_orden(lineas: list[dict], ancho: float | None = None) -> str:
    """
    Texto del CV en orden de lectura.

    Una columna: renglones de arriba abajo, cajas de cada renglon de izquierda a derecha.

    Dos columnas: las cajas que cruzan el canal central (cabecera, titulos a todo lo
    ancho) parten la pagina en franjas horizontales. Dentro de cada franja se lee
    primero toda la columna izquierda y despues toda la derecha. Asi una barra lateral
    ("Skills", "Contact") no se intercala renglon a renglon con la experiencia.
    """
    if not lineas:
        return ""
    ancho = ancho or max(l["box"][2] for l in lineas)
    x = _canal_central(lineas, ancho)
    if x is None:
        bloques = [lineas]
    else:
        anchas = sorted([l for l in lineas if l["box"][0] < x < l["box"][2]],
                        key=lambda l: l["box"][1])
        resto = [l for l in lineas if not (l["box"][0] < x < l["box"][2])]
        cortes = [-np.inf] + [l["box"][1] for l in anchas] + [np.inf]
        bloques = []
        for i in range(len(cortes) - 1):
            franja = [l for l in resto if cortes[i] <= l["box"][1] < cortes[i + 1]]
            bloques.append([l for l in franja if l["box"][2] <= x])
            bloques.append([l for l in franja if l["box"][2] > x])
            if i < len(anchas):
                bloques.append([anchas[i]])
    salida = []
    for b in bloques:
        for fila in _renglones(b):
            salida.append(" ".join(c["text"] for c in fila))
    return "\n".join(salida)


# ================================================================ datos personales
_EMAIL = re.compile(r"[\w.+-]+\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+")
# Formato de telefono de EE. UU. (+1-611-270-1253, (611) 270 1253...). Se evito a
# proposito un patron generico del tipo "digitos, puntos y guiones": tambien capturaria
# "2017.12-2020.07" y borraria las fechas, que son justo la evidencia del modelo.
_TELEFONO = re.compile(r"(?<![\d.])(?:\+\s?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
                       r"(?![\d.])")
_UBICACION = re.compile(r"\b[A-Z]{2}\s*,\s*USA\b")
RENGLONES_CABECERA = 3


def borrar_datos_personales(texto: str, nombre: str, apellido: str) -> str:
    """
    Quita nombre, email, telefono y ubicacion; se reemplazan por una etiqueta fija.

    El nombre solo se busca en los primeros renglones (la cabecera). Buscarlo en todo el
    CV borraria tambien empresas o universidades que coinciden con un apellido
    ("Hill", "Wood", "Young"...).
    """
    t = _EMAIL.sub("[EMAIL]", texto)
    t = _TELEFONO.sub("[PHONE]", t)
    t = _UBICACION.sub("[LOCATION]", t)
    renglones = t.split("\n")
    cab = "\n".join(renglones[:RENGLONES_CABECERA])
    for parte in (nombre, apellido):
        if isinstance(parte, str) and len(parte) >= 2:
            cab = re.sub(rf"\b{re.escape(parte)}\b", "[NAME]", cab, flags=re.I)
    cab = re.sub(r"\[NAME\](?:[ \t]*\[NAME\])+", "[NAME]", cab)
    return "\n".join([cab] + renglones[RENGLONES_CABECERA:])


# ============================================================ respuestas directas
# Cada patron va con un nombre, para poder informar que se enmascaro y cuantas veces.
PATRONES_RESPUESTA = {
    "total_experience": re.compile(
        r"\b\d{1,2}(?:\.\d)?\s*\+?\s*(?:years?|yrs?)\.?\s+(?:of\s+)?(?:\w+\s+){0,2}"
        r"experience\b|\bexperience\s*(?:of|:)?\s*\d{1,2}\s*\+?\s*(?:years?|yrs?)\b", re.I),
    "age": re.compile(r"\bage\s*[:\-]?\s*\d{1,3}\b|\b\d{1,3}\s*(?:years?|yrs?)\s*old\b", re.I),
    # \bborn exige que "born" empiece palabra: "newborn care" (cursos de enfermeria) NO
    # coincide. Se comprobo con los 60,884 textos de referencia.
    "birth_date": re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|born(?:\s+(?:on|in))?)\b"
        r"\s*[:\-]?\s*[\w ,./-]{0,20}?\d{4}", re.I),
}


def enmascarar_respuestas(texto: str) -> tuple[str, dict[str, int]]:
    cuenta = {}
    for nombre, patron in PATRONES_RESPUESTA.items():
        texto, n = patron.subn("[MASKED]", texto)
        cuenta[nombre] = n
    return texto, cuenta


# ==================================================================== limpieza
_GUIONES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})


def limpiar(texto: str) -> str:
    t = texto.translate(_GUIONES)
    # Vinetas: "•", "·", "*" o "-" al principio de renglon → "- " uniforme.
    t = re.sub(r"^\s*[•·*-]+\s*", "- ", t, flags=re.M)
    # Palabra cortada con guion al final de renglon: "manage-\nment" → "management".
    t = re.sub(r"([a-z])-\n([a-z])", r"\1\2", t)
    # Rango de fechas al que el OCR le comio o le metio espacios: "2017.12 - 2020.07".
    t = re.sub(r"(\d{4}\.\d{2})\s*-\s*(\d{4}\.\d{2})", r"\1-\2", t)
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def procesar(lineas: list[dict], nombre: str, apellido: str, umbral: float,
             ancho: float | None = None) -> dict:
    """Las tres variantes de un CV a partir de sus lineas de OCR."""
    base = texto_en_orden(lineas, ancho)
    raw = borrar_datos_personales(base, nombre, apellido)
    raw, cuenta = enmascarar_respuestas(raw)

    confiables = [l for l in lineas if l["score"] >= umbral]
    base_conf = texto_en_orden(confiables, ancho)
    conf = borrar_datos_personales(base_conf, nombre, apellido)
    conf, _ = enmascarar_respuestas(conf)

    return {"raw": raw, "clean": limpiar(raw), "clean_conf": limpiar(conf),
            "lines_dropped": len(lineas) - len(confiables), "masked": cuenta}


# ======================================================================== main
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--umbral-conf", type=float, default=UMBRAL_CONF)
    p.add_argument("--revision", type=int, default=50,
                   help="Cuantos textos al azar exportar para revision manual")
    args = p.parse_args()
    log = configurar_logging("preprocessing")

    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref = pd.read_parquet(RAW / "reference_text.parquet").set_index("id").text

    from src.ocr_quality import rangos

    filas, mascaras, confianzas = [], [], []
    sin_ocr = rangos_perdidos = 0
    for _, m in meta.iterrows():
        doc = cargar_ocr(m.id)
        if doc is None or not doc.get("lines"):
            sin_ocr += 1
            continue
        out = procesar(doc["lines"], m.first_name, m.last_name, args.umbral_conf,
                       doc.get("width"))
        ref_txt = borrar_datos_personales(ref[m.id], m.first_name, m.last_name)
        ref_txt, cuenta_ref = enmascarar_respuestas(ref_txt)
        # Salvaguarda: borrar datos personales y enmascarar NUNCA debe tocar un rango de
        # fechas de empleo. Se comprueba sobre el texto de referencia, donde se sabe
        # cuantos rangos hay.
        if rangos(ref_txt) != rangos(ref[m.id]):
            rangos_perdidos += 1
        # Texto del OCR rapido (si se corrio), con la misma limpieza que `clean`. Solo se
        # usa para evaluar la robustez al ruido (Pregunta 2).
        doc_m = cargar_ocr(m.id, OCR_RUIDOSO)
        movil = (procesar(doc_m["lines"], m.first_name, m.last_name, args.umbral_conf,
                          doc_m.get("width"))["clean"] if doc_m and doc_m.get("lines")
                 else None)
        filas.append({"id": m.id, "job": m.job, "y": m.y,
                      "raw": out["raw"], "clean": out["clean"],
                      "clean_conf": out["clean_conf"], "reference": limpiar(ref_txt),
                      "mobile": movil, "lines_dropped": out["lines_dropped"]})
        mascaras.append({"id": m.id, "job": m.job, **out["masked"],
                         **{f"ref_{k}": v for k, v in cuenta_ref.items()}})
        confianzas.extend({"id": m.id, "score": l["score"], "text": l["text"]}
                          for l in doc["lines"])

    if sin_ocr:
        log.warning("%d CVs sin OCR (no se incluyen)", sin_ocr)
    if rangos_perdidos:
        log.error("La limpieza altero los rangos de fechas de %d CVs de referencia",
                  rangos_perdidos)
        return 1
    log.info("Salvaguarda OK: ningun rango de fechas se altero al borrar datos personales")
    df = pd.DataFrame(filas)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "texts.parquet", index=False)

    mk = pd.DataFrame(mascaras)
    mk.to_csv(TABLES / "masking_report.csv", index=False)
    cols = list(PATRONES_RESPUESTA)
    afectados = int((mk[cols].sum(axis=1) > 0).sum())

    cf = pd.DataFrame(confianzas)
    cf.to_csv(TABLES / "ocr_confidence_lines.csv", index=False)
    bajo = cf[cf.score < args.umbral_conf]

    # Revision manual: el enunciado pide leer al menos 50 textos de OCR al azar antes de
    # modelar. Se exportan con su variante raw para leerlos tal como los ve el modelo.
    rev = df.sample(min(args.revision, len(df)), random_state=SEED)[["id", "job", "y", "raw"]]
    rev.to_csv(TABLES / "manual_review_sample.csv", index=False)

    guardar_json({
        "cvs": len(df), "confidence_threshold": args.umbral_conf,
        "cvs_masked_direct_answer": afectados,
        "masked_by_pattern": {c: int(mk[c].sum()) for c in cols},
        "reference_masked_by_pattern": {c: int(mk[f"ref_{c}"].sum()) for c in cols},
        "lines_total": len(cf), "lines_below_threshold": len(bajo),
        "lines_below_threshold_pct": 100 * len(bajo) / max(len(cf), 1),
        "cvs_with_dropped_lines": int((df.lines_dropped > 0).sum()),
        "chars_mean": {v: float(df[v].str.len().mean()) for v in
                       ("raw", "clean", "clean_conf", "reference")},
    }, METRICS / "preprocessing.json")

    log.info("CVs procesados: %d", len(df))
    log.info("CVs con alguna respuesta directa enmascarada: %d %s", afectados,
             {c: int(mk[c].sum()) for c in cols})
    log.info("Renglones con confianza < %.2f: %d de %d (%.2f %%) en %d CVs",
             args.umbral_conf, len(bajo), len(cf), 100 * len(bajo) / max(len(cf), 1),
             int((df.lines_dropped > 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
