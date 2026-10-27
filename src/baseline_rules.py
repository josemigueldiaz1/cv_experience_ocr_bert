"""
Bonus — Modelo de reglas: suma de duraciones de los empleos.

Hace lo que haria un reclutador con un lapiz: busca cada rango de fechas de empleo
(`2017.12-2020.07`), calcula cuanto duro, fusiona los periodos que se solapan (dos
trabajos a la vez no cuentan doble) y suma.

No tiene ningun parametro y no se entrena. Es la prueba honesta que propone el
enunciado: si BERT no le gana a una resta, el transformer no esta aportando.

Lo que lo rompe
---------------
Solo puede fallar si el OCR estropea un rango (un digito mal leido, un guion perdido) o
si el CV usa un formato de fecha que la regex no reconoce. Por eso es tambien un
termometro directo de lo que cuesta el ruido del OCR (Pregunta 2).

Uso
---
    python src/baseline_rules.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import cargar_split  # noqa: E402
from src.utils import (VARIANTES, configurar_logging, cronometrar_inferencia,  # noqa: E402
                       guardar_predicciones, recortar)

# Anio.mes, separador (guion de cualquier tipo, "to", o nada si el OCR se lo comio),
# anio y, si esta, mes. "Present"/"Current" se cierra en la ultima fecha del dataset.
_RANGO = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*[./]\s*(\d{1,2})\s*(?:[-‐-―−~]|to)?\s*"
    r"(?:((?:19|20)\d{2})(?:\s*[./]\s*(\d{0,2}))?|(present|current|now))(?![\d.])", re.I)
FECHA_ACTUAL = 2023 + 0 / 12   # enero de 2023: fecha de fin mas reciente del dataset


def intervalos(texto: str) -> list[tuple[float, float]]:
    """Rangos de empleo validos como (inicio, fin) en anios decimales."""
    salida = []
    for a1, m1, a2, m2, actual in _RANGO.findall(texto):
        m1 = int(m1)
        if not 1 <= m1 <= 12:
            continue
        ini = int(a1) + (m1 - 1) / 12
        if actual:
            fin = FECHA_ACTUAL
        else:
            # Mes de fin ausente o cortado ("2023." o "2022.0"): en PopResume la imagen
            # corta las lineas largas (ver ocr_quality.py). Se asume enero, el valor mas
            # conservador; el error maximo es de 11 meses en vez de perder el empleo entero.
            m2 = int(m2) if m2 and len(m2) == 2 else 1
            if not 1 <= m2 <= 12:
                continue
            fin = int(a2) + (m2 - 1) / 12
        if fin >= ini:
            salida.append((ini, fin))
    return salida


def anios_experiencia(texto: str) -> float:
    """Suma de la union de los intervalos (los solapes se cuentan una vez)."""
    total, fin_actual = 0.0, -np.inf
    for ini, fin in sorted(intervalos(texto)):
        if ini > fin_actual:
            total += fin - ini
            fin_actual = fin
        elif fin > fin_actual:
            total += fin - fin_actual
            fin_actual = fin
    return total


def predecir(textos) -> np.ndarray:
    return recortar([anios_experiencia(t) for t in textos])


def cargar(variante: str = "clean"):
    return predecir


def main() -> int:
    log = configurar_logging("baseline_rules")
    for variante in VARIANTES:
        val, test = cargar_split("val", variante), cargar_split("test", variante)
        partes = [d.assign(split=s, pred=predecir(d.text)) for s, d in (("val", val), ("test", test))]
        df = pd.concat(partes)
        sin_fechas = int(sum(not intervalos(t) for t in test.text))
        guardar_predicciones("rules", variante, df, {
            "model": "rules", "variant": variante, "train_seconds": 0.0,
            "inference_seconds_per_cv": cronometrar_inferencia(predecir, test.text.tolist(), 5),
            "test_cvs_without_dates": sin_fechas,
        })
        v = df[df.split == "val"]
        log.info("  reglas %-10s MAE val=%.3f | CVs de test sin ningun rango: %d", variante,
                 float(np.mean(np.abs(v.y - v.pred))), sin_fechas)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
