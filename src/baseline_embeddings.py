"""
Baseline 2 — Embeddings de oraciones (encoder congelado) + regresion Ridge.

Cada CV se convierte en UN vector de 384 numeros con un encoder preentrenado que NO se
ajusta (se usa tal cual viene). Despues, una Ridge aprende a pasar de ese vector a anios.

Que puede y que no puede aprender
---------------------------------
El vector resume "de que trata" el CV: ocupacion, seniority de los puestos, tipo de
tareas. Los encoders de oraciones estan entrenados para parecido semantico, no para
aritmetica: dos CVs con fechas distintas pero mismas tareas quedan casi en el mismo
punto. Es la prueba de si el "significado general" del CV basta.

Decisiones
----------
- Encoder: se prueban los tres que sugiere el enunciado (bge-small-en-v1.5,
  all-MiniLM-L6-v2, e5-small-v2) y se elige el de menor MAE en VALIDACION.
- CVs largos: si un CV supera el limite del encoder, se parte en trozos con solape, se
  codifica cada trozo y se promedian los vectores. Se registra cuantos CVs lo
  necesitaron (en este dataset, ninguno: los CVs tienen ~70 palabras).
- Vectores normalizados (longitud 1), y alpha de Ridge elegido en validacion.

Uso
---
    python src/baseline_embeddings.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import cargar_split  # noqa: E402
from src.utils import (MODELS_BASELINES, OUTPUT, SEED, TABLES, VARIANTE_PRINCIPAL,  # noqa: E402
                       VARIANTES, configurar_logging, cronometrar_inferencia, guardar_json,
                       guardar_predicciones, leer_json, recortar)

ENCODERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    # e5 se entreno con prefijos: los documentos van como "passage: ...".
    "e5-small": ("intfloat/e5-small-v2", "passage: "),
}
ALPHAS = np.logspace(-3, 3, 13)
SOLAPE = 64
CACHE = OUTPUT / "cache"

log = None


class Codificador:
    """Encoder congelado con troceo para textos que no caben."""

    def __init__(self, clave: str):
        from sentence_transformers import SentenceTransformer

        self.clave = clave
        self.nombre, self.prefijo = ENCODERS[clave]
        self.st = SentenceTransformer(self.nombre)
        self.max_tokens = self.st.max_seq_length
        self.trozados = 0

    def _trozos(self, texto: str) -> list[str]:
        tok = self.st.tokenizer
        ids = tok(self.prefijo + texto, add_special_tokens=False)["input_ids"]
        util = self.max_tokens - 2  # [CLS] y [SEP]
        if len(ids) <= util:
            return [self.prefijo + texto]
        self.trozados += 1
        paso = util - SOLAPE
        return [tok.decode(ids[i:i + util]) for i in range(0, len(ids), paso)]

    def __call__(self, textos: list[str]) -> np.ndarray:
        trozos, dueno = [], []
        for k, t in enumerate(textos):
            for c in self._trozos(t):
                trozos.append(c)
                dueno.append(k)
        v = self.st.encode(trozos, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
        dueno = np.array(dueno)
        salida = np.vstack([v[dueno == k].mean(axis=0) for k in range(len(textos))])
        return salida / np.linalg.norm(salida, axis=1, keepdims=True)


def _embeddings(cod: Codificador, df: pd.DataFrame, variante: str,
                split: str) -> tuple[np.ndarray, float]:
    """
    Embeddings con cache en disco: el encoder no cambia, no hace falta recalcularlos.
    Devuelve tambien los segundos que costo codificarlos (guardados en la cache), para
    que el tiempo de entrenamiento reportado no dependa de si habia cache o no.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    ruta = CACHE / f"emb_{cod.clave}__{variante}__{split}.npz"
    if ruta.exists():
        z = np.load(ruta, allow_pickle=False)
        if list(z["ids"]) == list(df.id) and "segundos" in z:
            return z["x"], float(z["segundos"])
    t0 = time.perf_counter()
    x = cod(df.text.tolist())
    segundos = time.perf_counter() - t0
    # Los ids como texto de numpy ('<U5'), no como objetos de Python: asi el archivo se
    # puede leer con allow_pickle=False, que no ejecuta codigo al cargar.
    np.savez(ruta, ids=np.array(df.id.astype(str).tolist()), x=x, segundos=segundos)
    return x, segundos


def _ajustar_ridge(xtr, ytr, xva, yva):
    mejor = None
    for a in ALPHAS:
        r = Ridge(alpha=a, random_state=SEED).fit(xtr, ytr)
        mae = mean_absolute_error(yva, recortar(r.predict(xva)))
        if mejor is None or mae < mejor[0]:
            mejor = (mae, a, r)
    return mejor


def elegir_encoder() -> str:
    """Compara los tres encoders en VALIDACION con la variante principal."""
    train, val = cargar_split("train", VARIANTE_PRINCIPAL), cargar_split("val", VARIANTE_PRINCIPAL)
    filas = []
    for clave in ENCODERS:
        cod = Codificador(clave)
        xtr, _ = _embeddings(cod, train, VARIANTE_PRINCIPAL, "train")
        xva, _ = _embeddings(cod, val, VARIANTE_PRINCIPAL, "val")
        mae, alpha, _ = _ajustar_ridge(xtr, train.y, xva, val.y)
        filas.append({"encoder": clave, "model": ENCODERS[clave][0], "val_mae": mae,
                      "alpha": alpha, "max_tokens": cod.max_tokens})
        log.info("  encoder %-10s MAE val=%.3f (alpha=%.3g)", clave, mae, alpha)
    tabla = pd.DataFrame(filas).sort_values("val_mae")
    tabla.to_csv(TABLES / "embeddings_encoder_selection.csv", index=False)
    return str(tabla.iloc[0].encoder)


def entrenar(variante: str, clave: str) -> None:
    cod = Codificador(clave)
    train, val, test = (cargar_split(s, variante) for s in ("train", "val", "test"))
    xtr, t_codif = _embeddings(cod, train, variante, "train")
    xva, _ = _embeddings(cod, val, variante, "val")
    xte, _ = _embeddings(cod, test, variante, "test")
    t1 = time.perf_counter()
    mae, alpha, ridge = _ajustar_ridge(xtr, train.y, xva, val.y)
    t_ridge = time.perf_counter() - t1

    joblib.dump(ridge, MODELS_BASELINES / f"embeddings__{variante}.joblib")
    guardar_json({"encoder": clave, "model": ENCODERS[clave][0]},
                 MODELS_BASELINES / "embeddings_encoder.json")

    partes = [val.assign(split="val", pred=recortar(ridge.predict(xva))),
              test.assign(split="test", pred=recortar(ridge.predict(xte)))]
    predecir = lambda textos: ridge.predict(cod(textos))  # noqa: E731
    guardar_predicciones("embeddings", variante, pd.concat(partes), {
        "model": "embeddings", "variant": variante, "encoder": ENCODERS[clave][0],
        "alpha": float(alpha), "val_mae": float(mae), "dim": int(xtr.shape[1]),
        "max_tokens": cod.max_tokens, "cvs_chunked": cod.trozados,
        # El "entrenamiento" es codificar train (el encoder no se ajusta) + la Ridge.
        "train_seconds": t_codif + t_ridge,
        "inference_seconds_per_cv": cronometrar_inferencia(predecir, test.text.tolist()),
    })
    log.info("  embeddings %-10s encoder=%s alpha=%.3g  MAE val=%.3f  (trozados: %d)",
             variante, clave, alpha, mae, cod.trozados)


def cargar(variante: str = VARIANTE_PRINCIPAL):
    clave = leer_json(MODELS_BASELINES / "embeddings_encoder.json")["encoder"]
    cod = Codificador(clave)
    ridge = joblib.load(MODELS_BASELINES / f"embeddings__{variante}.joblib")
    return lambda textos: recortar(ridge.predict(cod(list(textos))))


def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", choices=list(ENCODERS),
                   help="Salta la seleccion y usa este encoder")
    args = p.parse_args()
    log = configurar_logging("baseline_embeddings")
    MODELS_BASELINES.mkdir(parents=True, exist_ok=True)

    clave = args.encoder or elegir_encoder()
    log.info("Encoder elegido: %s", clave)
    for variante in VARIANTES:
        entrenar(variante, clave)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
