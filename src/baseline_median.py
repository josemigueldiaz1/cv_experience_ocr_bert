"""
Baseline 0 — Predictor de la mediana.

Predice siempre la mediana de anios de experiencia del conjunto de entrenamiento, sin
mirar el CV. Es el piso: cualquier modelo que no le gane no aprendio nada del texto.

Se usa la mediana y no la media porque la metrica principal es el MAE, y la constante
que minimiza el error absoluto medio es justamente la mediana.

Uso
---
    python src/baseline_median.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import cargar_split  # noqa: E402
from src.utils import (MODELS_BASELINES, VARIANTES, configurar_logging, guardar_json,  # noqa: E402
                       guardar_predicciones, leer_json)


def cargar(variante: str = "clean"):
    """Devuelve `predecir(textos) -> np.ndarray`."""
    mediana = leer_json(MODELS_BASELINES / "median.json")["median"]
    return lambda textos: np.full(len(textos), mediana, dtype=float)


def main() -> int:
    log = configurar_logging("baseline_median")
    t0 = time.perf_counter()
    train = cargar_split("train")
    mediana = float(train.y.median())
    MODELS_BASELINES.mkdir(parents=True, exist_ok=True)
    guardar_json({"median": mediana, "train_size": len(train)},
                 MODELS_BASELINES / "median.json")
    entrenamiento = time.perf_counter() - t0

    # La mediana no lee el texto: las predicciones son iguales en todas las variantes,
    # pero se guardan para cada una para que la tabla de sensibilidad quede completa.
    for variante in VARIANTES:
        partes = []
        for nombre in ("val", "test"):
            df = cargar_split(nombre)
            partes.append(df.assign(split=nombre, pred=mediana))
        guardar_predicciones("median", variante, pd.concat(partes), {
            "model": "median", "variant": variante, "median": mediana,
            "train_seconds": entrenamiento, "inference_seconds_per_cv": 0.0,
        })
    log.info("Mediana de entrenamiento: %.3f anios (n=%d)", mediana, len(train))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
