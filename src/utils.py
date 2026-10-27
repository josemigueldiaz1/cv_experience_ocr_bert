"""
Utilidades compartidas: rutas, constantes del experimento, semillas y logging.

Todo lo que mas de un script necesita saber vive aqui, para que una decision (la semilla,
el tope de 20 anios, las bandas de experiencia) se cambie en un solo sitio y no quede
desincronizada entre el baseline y BERT.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ================================================================================ rutas
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = RAW / "images"
OCR = DATA / "ocr"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"

MODELS = ROOT / "models"
MODELS_BASELINES = MODELS / "baselines"
MODELS_BERT = MODELS / "bert"

OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
METRICS = OUTPUT / "metrics"
PREDICTIONS = OUTPUT / "predictions"

LOGS = ROOT / "logs"

# ======================================================================== experimento
SEED = 42

# Las cinco ocupaciones de PopResume, con el nombre exacto de sus shards en Hugging Face.
OCUPACIONES = [
    "registered_nurses",
    "elementary_middle_school_teachers",
    "software_developers",
    "accountants_auditors",
    "construction_laborers",
]
N_POR_OCUPACION = 800

# Tope de la variable objetivo: todo lo que pase de 20 se lee como "20 o mas".
CAP = 20

# Bandas de experiencia que pide el enunciado para la evaluacion como clasificacion.
BANDAS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, CAP, f"11-{CAP}")]
NOMBRES_BANDAS = [b[2] for b in BANDAS]

# Variantes del texto que se comparan en la Pregunta 3 (sensibilidad a la limpieza).
VARIANTES = ["raw", "clean", "clean_conf"]
VARIANTE_PRINCIPAL = "clean"


def banda(anios: float) -> str:
    """Convierte anios de experiencia (reales o predichos) a su banda."""
    y = int(round(min(max(float(anios), 0), CAP)))
    for lo, hi, nombre in BANDAS:
        if lo <= y <= hi:
            return nombre
    return NOMBRES_BANDAS[-1]


def recortar(valores):
    """Recorta predicciones al rango valido [0, CAP]."""
    return np.clip(np.asarray(valores, dtype=float), 0, CAP)


# ============================================================================ semillas
def fijar_semilla(seed: int = SEED) -> None:
    """Fija todas las fuentes de azar que usan los scripts."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================================= logging
def configurar_logging(nombre: str, verboso: bool = False) -> logging.Logger:
    """
    Log a consola y a `logs/<nombre>_<fecha>.log` a la vez.

    Cada script llama a esto una vez al empezar. Asi cada etapa deja su propio registro
    y, si algo falla en una corrida larga, el detalle no se pierde con la consola.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    archivo = LOGS / f"{nombre}_{datetime.now():%Y%m%d_%H%M%S}.log"

    raiz = logging.getLogger()
    raiz.setLevel(logging.DEBUG if verboso else logging.INFO)
    raiz.handlers.clear()

    fh = logging.FileHandler(archivo, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", "%H:%M:%S"))
    # El filtro va en los handlers y no en los loggers de cada libreria: algunas
    # (huggingface_hub, paddlex) vuelven a fijar su nivel al importarse y lo pisarian.
    for h in (fh, sh):
        h.addFilter(_SinRuido())
        raiz.addHandler(h)
    return logging.getLogger(nombre)


_RUIDOSOS = ("urllib3", "httpx", "httpcore", "filelock", "fsspec", "PIL",
             "huggingface_hub", "datasets", "sentence_transformers",
             "transformers", "matplotlib", "ppocr", "paddlex", "paddle", "streamlit")


class _SinRuido(logging.Filter):
    """Deja pasar avisos y errores de las librerias, pero no su INFO/DEBUG."""

    def filter(self, r: logging.LogRecord) -> bool:
        return r.levelno >= logging.WARNING or not r.name.startswith(_RUIDOSOS)


# ============================================================================== json io
def guardar_json(obj, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_a_json),
                    encoding="utf-8")


def leer_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _a_json(o):
    """Permite guardar tipos de numpy sin convertirlos a mano."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No se puede serializar {type(o)}")


# ========================================================================= predicciones
# Nombre interno → nombre en tablas y figuras. El orden es el de la tabla comparativa.
NOMBRES_MODELOS = {
    "median": "Mediana",
    "tfidf": "TF-IDF + Ridge",
    "embeddings": "Embeddings + Ridge",
    "bert": "BERT afinado",
    "rules": "Reglas de fechas (bonus)",
}

# Nombres legibles de las ocupaciones (figuras, tablas y app).
NOMBRES_OCUPACIONES = {
    "registered_nurses": "Enfermería",
    "elementary_middle_school_teachers": "Docentes",
    "software_developers": "Desarrollo de software",
    "accountants_auditors": "Contabilidad y auditoría",
    "construction_laborers": "Construcción",
}


def guardar_predicciones(modelo: str, variante: str, df: "pd.DataFrame", info: dict) -> None:
    """
    Formato comun de todos los modelos: una fila por CV de validacion y test con
    columnas id, split, y, pred. `info` (tiempos, hiperparametros) va a un JSON aparte.
    """
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    df = df[["id", "split", "y", "pred"]].copy()
    df["pred"] = recortar(df.pred)
    df.to_csv(PREDICTIONS / f"{modelo}__{variante}.csv", index=False)
    guardar_json(info, PREDICTIONS / f"{modelo}__{variante}.json")


def leer_predicciones(modelo: str, variante: str = VARIANTE_PRINCIPAL):
    import pandas as pd
    ruta = PREDICTIONS / f"{modelo}__{variante}.csv"
    return pd.read_csv(ruta, dtype={"id": str}) if ruta.exists() else None


def cronometrar_inferencia(predecir, textos: list[str], repeticiones: int = 1) -> float:
    """Segundos por CV de una funcion `predecir(textos)`, medidos sobre `textos`."""
    import time
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        predecir(textos)
    return (time.perf_counter() - t0) / (repeticiones * max(len(textos), 1))


def asegurar_carpetas() -> None:
    for d in (IMAGES, OCR, PROCESSED, SPLITS, MODELS_BASELINES, MODELS_BERT,
              FIGURES, TABLES, METRICS, PREDICTIONS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
