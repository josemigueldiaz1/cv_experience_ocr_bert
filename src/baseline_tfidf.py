"""
Baseline 1 — TF-IDF + regresion Ridge.

Convierte cada CV en un vector de pesos TF-IDF (que palabras y pares de palabras
aparecen, y cuanto destacan frente al resto de CVs) y ajusta una regresion lineal con
regularizacion L2 (Ridge).

Que puede y que no puede aprender
---------------------------------
Ve "2017", "2020" y "12" como fichas sueltas, sin orden ni distancia entre ellas. Puede
aprender asociaciones ("el token 2005 aparece en CVs con mucha experiencia", "cuantos mas
empleos, mas anios"), pero no puede restar 2020.07 − 2017.12. Por eso es el contraste
justo para BERT.

Decisiones (todas ajustadas en VALIDACION)
------------------------------------------
- n-gramas de palabras 1-2, y opcionalmente n-gramas de caracteres 2-5 (que ven
  fragmentos de numeros como "201" o "2.0"). Se elige con validacion.
- Numeros: se CONSERVAN como tokens en el modelo principal, porque las fechas son la
  unica evidencia que hay en el CV. La alternativa (borrar todos los digitos) se entrena
  igual y se reporta en la Pregunta 3: mide cuanto del acierto de TF-IDF viene de los
  anios y cuanto de las palabras (tipo de puesto, numero de empleos, titulo).
- Fuerza de regularizacion alpha: rejilla logaritmica de 0.01 a 100.

Uso
---
    python src/baseline_tfidf.py                 # las 3 variantes de texto + sin digitos
    python src/baseline_tfidf.py --variante clean
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import cargar_split  # noqa: E402
from src.utils import (MODELS_BASELINES, SEED, TABLES, VARIANTE_PRINCIPAL, VARIANTES,  # noqa: E402
                       configurar_logging, cronometrar_inferencia, guardar_predicciones,
                       recortar)

ALPHAS = np.logspace(-2, 2, 9)
CONFIGS_NGRAMAS = {"word12": False, "word12+char25": True}

log = None


def _sin_digitos(t: str) -> str:
    return re.sub(r"\d+", " ", t.lower())


class ModeloTfidf:
    """Vectorizadores + Ridge. Se guarda entero con joblib."""

    def __init__(self, con_chars: bool, con_digitos: bool, alpha: float):
        pre = None if con_digitos else _sin_digitos
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                    preprocessor=pre)
        self.char = (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                     sublinear_tf=True, preprocessor=pre, max_features=50000)
                     if con_chars else None)
        self.ridge = Ridge(alpha=alpha, random_state=SEED)

    def _x(self, textos, ajustar=False):
        f = (lambda v: v.fit_transform(textos)) if ajustar else (lambda v: v.transform(textos))
        partes = [f(self.word)] + ([f(self.char)] if self.char is not None else [])
        return hstack(partes).tocsr()

    def fit(self, textos, y):
        self.ridge.fit(self._x(textos, ajustar=True), y)
        return self

    def predict(self, textos):
        return recortar(self.ridge.predict(self._x(list(textos))))


# joblib guarda la RUTA de la clase. Ejecutado como script, seria `__main__.ModeloTfidf`
# y ni la evaluacion ni la app podrian cargar el modelo. Se registra este mismo modulo
# con su nombre importable y se fija esa ruta en la clase.
if __name__ == "__main__":
    sys.modules.setdefault("src.baseline_tfidf", sys.modules[__name__])
ModeloTfidf.__module__ = _sin_digitos.__module__ = "src.baseline_tfidf"


def ruta_modelo(variante: str, con_digitos: bool = True) -> Path:
    sufijo = "" if con_digitos else "_nodigits"
    return MODELS_BASELINES / f"tfidf{sufijo}__{variante}.joblib"


def cargar(variante: str = VARIANTE_PRINCIPAL, con_digitos: bool = True):
    modelo = joblib.load(ruta_modelo(variante, con_digitos))
    return lambda textos: modelo.predict(list(textos))


def entrenar(variante: str, con_digitos: bool = True) -> None:
    nombre = "tfidf" if con_digitos else "tfidf_nodigits"
    train, val, test = (cargar_split(s, variante) for s in ("train", "val", "test"))

    # Busqueda en rejilla sobre VALIDACION. Los vectorizadores se ajustan solo con train.
    t0 = time.perf_counter()
    resultados = []
    for (cfg, con_chars), alpha in product(CONFIGS_NGRAMAS.items(), ALPHAS):
        m = ModeloTfidf(con_chars, con_digitos, alpha).fit(train.text.tolist(), train.y)
        mae = mean_absolute_error(val.y, m.predict(val.text))
        resultados.append({"ngrams": cfg, "alpha": alpha, "val_mae": mae})
    rejilla = pd.DataFrame(resultados).sort_values("val_mae")
    mejor = rejilla.iloc[0]

    modelo = ModeloTfidf(CONFIGS_NGRAMAS[mejor.ngrams], con_digitos, mejor.alpha)
    t1 = time.perf_counter()
    modelo.fit(train.text.tolist(), train.y)
    t_entreno = time.perf_counter() - t1
    t_busqueda = time.perf_counter() - t0

    MODELS_BASELINES.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta_modelo(variante, con_digitos))
    rejilla.to_csv(TABLES / f"grid_{nombre}__{variante}.csv", index=False)

    partes = [d.assign(split=s, pred=modelo.predict(d.text))
              for s, d in (("val", val), ("test", test))]
    guardar_predicciones(nombre, variante, pd.concat(partes), {
        "model": nombre, "variant": variante, "ngrams": mejor.ngrams,
        "alpha": float(mejor.alpha), "keep_digits": con_digitos,
        "val_mae": float(mejor.val_mae), "n_features": int(modelo.ridge.coef_.size),
        "train_seconds": t_entreno, "grid_search_seconds": t_busqueda,
        "inference_seconds_per_cv": cronometrar_inferencia(modelo.predict, test.text.tolist()),
    })
    log.info("  %-15s %-10s ngramas=%s alpha=%.3g  MAE val=%.3f", nombre, variante,
             mejor.ngrams, mejor.alpha, mejor.val_mae)

    if variante == VARIANTE_PRINCIPAL and con_digitos:
        exportar_coeficientes(modelo)


def exportar_coeficientes(modelo: ModeloTfidf, k: int = 25) -> None:
    """
    Las fichas que mas suben y mas bajan la prediccion. Responde a "que usa el modelo
    para adivinar", que el enunciado pide mostrar para cada modelo.
    """
    nombres = list(modelo.word.get_feature_names_out())
    if modelo.char is not None:
        nombres += [f"char:'{c}'" for c in modelo.char.get_feature_names_out()]
    coef = modelo.ridge.coef_
    orden = np.argsort(coef)
    filas = ([{"direction": "lowers", "feature": nombres[i], "coef": coef[i]} for i in orden[:k]]
             + [{"direction": "raises", "feature": nombres[i], "coef": coef[i]}
                for i in orden[::-1][:k]])
    pd.DataFrame(filas).to_csv(TABLES / "tfidf_top_features.csv", index=False)


def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variante", choices=VARIANTES)
    args = p.parse_args()
    log = configurar_logging("baseline_tfidf")

    for variante in ([args.variante] if args.variante else VARIANTES):
        entrenar(variante, con_digitos=True)
    # Pregunta 3: misma receta, sin digitos (solo sobre la variante principal).
    if args.variante in (None, VARIANTE_PRINCIPAL):
        entrenar(VARIANTE_PRINCIPAL, con_digitos=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
