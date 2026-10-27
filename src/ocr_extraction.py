"""
Etapa 1 — Extraccion de texto con PaddleOCR.

Que hace
--------
Pasa cada imagen de `data/raw/images/` por PaddleOCR y guarda, por CV, un JSON en
`data/ocr/<id>.json` con cada renglon detectado: texto, confianza y caja
[x1, y1, x2, y2]. Tambien guarda el tiempo por CV y el hardware usado.

Es REANUDABLE: si el JSON de un CV ya existe con el mismo modelo, no se vuelve a
procesar. El OCR es la etapa lenta y se corre una sola vez.

Tres modos
----------
    python src/ocr_extraction.py                       # todos los CVs (modelo elegido)
    python src/ocr_extraction.py --benchmark 50        # compara 3 modelos en 50 CVs
    python src/ocr_extraction.py --imagen cv.png --salida out.json   # una imagen (app)

Por que este archivo oculta PyTorch
-----------------------------------
PaddlePaddle y PyTorch traen cada uno su propia copia de las DLL de CUDA (cuBLAS, cuDNN)
en versiones distintas (12.9 y 12.8). En Windows, la primera que se carga en un proceso
gana, y la segunda libreria falla con "The specified procedure could not be found".
PaddleOCR importa `modelscope`, y este importa torch si `find_spec("torch")` lo encuentra.
Este proceso nunca necesita torch, asi que se le responde que no existe (ver
`_ocultar_torch`). El resto del proyecto (BERT, embeddings) corre en procesos aparte.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _ocultar_torch() -> None:
    """Hace que `importlib.util.find_spec('torch')` devuelva None en este proceso."""
    original = importlib.util.find_spec

    def find_spec(name, package=None):
        if name == "torch" or name.startswith("torch."):
            return None
        return original(name, package)

    importlib.util.find_spec = find_spec


_ocultar_torch()
# Sin esto PaddleX comprueba en cada arranque la conexion con sus servidores de modelos.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (IMAGES, METRICS, OCR, RAW, SEED, TABLES,  # noqa: E402
                       configurar_logging, guardar_json, leer_json)

# Configuraciones comparadas. La elegida (MODELO) se justifica con el benchmark:
# output/tables/ocr_benchmark.csv y la seccion "OCR" del README.
MODELOS = {
    "v6_medium": dict(text_detection_model_name="PP-OCRv6_medium_det",
                      text_recognition_model_name="PP-OCRv6_medium_rec"),
    "v5_server": dict(text_detection_model_name="PP-OCRv5_server_det",
                      text_recognition_model_name="PP-OCRv5_server_rec"),
    "v5_mobile_en": dict(text_detection_model_name="PP-OCRv5_mobile_det",
                         text_recognition_model_name="en_PP-OCRv5_mobile_rec"),
}
MODELO = "v6_medium"

# Pregunta 2 ("cuanto cuesta el ruido del OCR"): con v6 el OCR es casi perfecto y la
# respuesta seria trivial. Se hace una segunda pasada con el modelo rapido, que rompe ~4
# de cada 10 rangos de fechas, para medir que modelo resiste mejor un OCR peor. Esa
# pasada SOLO se usa para evaluar; ningun modelo se entrena con ella.
MODELO_RUIDOSO = "v5_mobile_en"
OCR_RUIDOSO = OCR.parent / "ocr_mobile"

log = None


def crear_ocr(modelo: str = MODELO, dispositivo: str | None = None):
    from paddleocr import PaddleOCR
    import paddle

    if dispositivo is None:
        dispositivo = "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
    # Los tres modulos opcionales se apagan: las imagenes son paginas digitales rectas,
    # sin rotacion ni curvatura. Activarlos solo anade tiempo y riesgo de "corregir"
    # lo que ya esta bien.
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, device=dispositivo, **MODELOS[modelo])
    return ocr, dispositivo


def hardware(dispositivo: str) -> str:
    import platform
    import paddle

    if dispositivo.startswith("gpu"):
        return f"{paddle.device.cuda.get_device_name(0)} (CUDA, Paddle {paddle.__version__})"
    return f"CPU {platform.processor()} (Paddle {paddle.__version__})"


def ocr_imagen(ocr, ruta: Path) -> dict:
    """Corre el OCR sobre una imagen y devuelve el dict que se guarda como JSON."""
    t0 = time.perf_counter()
    r = ocr.predict(str(ruta))[0]
    dt = time.perf_counter() - t0
    alto, ancho = _tamano(ruta)
    lineas = [{"text": t, "score": round(float(s), 5), "box": [int(v) for v in b]}
              for t, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"])]
    return {"seconds": round(dt, 4), "width": int(ancho), "height": int(alto),
            "lines": lineas}


def _tamano(ruta: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(ruta) as im:
        return im.height, im.width


# ===================================================================== modos de uso
def extraer_todo(modelo: str, force: bool, destino: Path = OCR) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(RAW / "metadata.csv", dtype={"id": str}).id.tolist()
    pendientes = []
    for i in ids:
        ruta = destino / f"{i}.json"
        if force or not ruta.exists() or leer_json(ruta).get("model") != modelo:
            pendientes.append(i)
    log.info("CVs: %d | ya procesados: %d | pendientes: %d", len(ids),
             len(ids) - len(pendientes), len(pendientes))

    fallidos = []
    if pendientes:
        ocr, disp = crear_ocr(modelo)
        hw = hardware(disp)
        log.info("Modelo %s en %s", modelo, hw)
        ocr.predict(str(IMAGES / f"{pendientes[0]}.jpg"))  # calentamiento (no se mide)
        t0 = time.perf_counter()
        for k, i in enumerate(pendientes, 1):
            try:
                doc = ocr_imagen(ocr, IMAGES / f"{i}.jpg")
                if not doc["lines"]:
                    raise ValueError("el OCR no detecto ningun texto")
                guardar_json({"id": i, "model": modelo, "hardware": hw, **doc},
                             destino / f"{i}.json")
            except Exception as e:  # una imagen rota no debe tumbar 4,000
                fallidos.append({"id": i, "error": repr(e)})
                log.warning("  fallo %s: %s", i, e)
            if k % 250 == 0 or k == len(pendientes):
                dt = time.perf_counter() - t0
                log.info("  %d/%d  (%.3f s/CV, faltan ~%.1f min)", k, len(pendientes),
                         dt / k, dt / k * (len(pendientes) - k) / 60)

    resumir(ids, fallidos, destino)
    return 0


def resumir(ids: list[str], fallidos: list[dict], destino: Path = OCR) -> None:
    """Tiempo por CV, hardware, confianza media y fallos, a partir de los JSON."""
    docs = [leer_json(destino / f"{i}.json") for i in ids if (destino / f"{i}.json").exists()]
    seg = np.array([d["seconds"] for d in docs])
    conf = np.array([l["score"] for d in docs for l in d["lines"]])
    lin = np.array([len(d["lines"]) for d in docs])
    resumen = {
        "model": docs[0]["model"] if docs else None,
        "hardware": docs[0]["hardware"] if docs else None,
        "cvs_ok": len(docs), "cvs_failed": len(ids) - len(docs),
        "failed": fallidos,
        "seconds_per_cv_mean": float(seg.mean()), "seconds_per_cv_median": float(np.median(seg)),
        "total_minutes": float(seg.sum() / 60),
        "lines_per_cv_mean": float(lin.mean()),
        "mean_confidence": float(conf.mean()), "min_confidence": float(conf.min()),
        "lines_below_0.90": int((conf < 0.90).sum()), "lines_total": int(conf.size),
    }
    sufijo = "" if destino == OCR else f"_{destino.name}"
    guardar_json(resumen, METRICS / f"ocr_extraction{sufijo}.json")
    log.info("OCR: %d CVs ok, %d fallidos | %.3f s/CV | confianza media %.4f | %s",
             resumen["cvs_ok"], resumen["cvs_failed"], resumen["seconds_per_cv_mean"],
             resumen["mean_confidence"], resumen["hardware"])


def benchmark(n: int) -> int:
    """
    Compara las configuraciones de MODELOS sobre los mismos n CVs al azar (10 por
    ocupacion si n=50): tiempo, CER, WER, anios y rangos de fechas leidos bien.
    """
    from src.ocr_quality import medir
    from src.preprocessing import texto_en_orden

    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})
    ref = pd.read_parquet(RAW / "reference_text.parquet").set_index("id").text
    muestra = (meta.groupby("job", group_keys=False)
               .apply(lambda g: g.sample(max(1, n // meta.job.nunique()), random_state=SEED)))
    filas = []
    for modelo in MODELOS:
        ocr, disp = crear_ocr(modelo)
        ocr.predict(str(IMAGES / f"{muestra.id.iloc[0]}.jpg"))
        for i in muestra.id:
            doc = ocr_imagen(ocr, IMAGES / f"{i}.jpg")
            filas.append({"model": modelo, "id": i, "seconds": doc["seconds"],
                          "mean_conf": float(np.mean([l["score"] for l in doc["lines"]])),
                          **medir(texto_en_orden(doc["lines"], doc["width"]), ref[i])})
        log.info("  %s listo", modelo)
        del ocr
    df = pd.DataFrame(filas)
    tabla = df.groupby("model").agg(
        cvs=("id", "size"), seconds_per_cv=("seconds", "mean"), cer=("cer", "mean"),
        wer=("wer", "mean"), years_ok=("years_ok", "mean"), ranges_ok=("ranges_ok", "mean"),
        cvs_all_ranges_ok=("ranges_ok", lambda s: (s == 1).mean()),
        mean_conf=("mean_conf", "mean"),
    ).reset_index().sort_values("cer")
    tabla.round(4).to_csv(TABLES / "ocr_benchmark.csv", index=False)
    df.to_csv(TABLES / "ocr_benchmark_per_cv.csv", index=False)
    for _, r in tabla.iterrows():
        log.info("  %-13s %.3f s/CV | CER %.4f | WER %.4f | anios %.1f%% | rangos %.1f%% | "
                 "CVs con todos los rangos bien %.0f%%", r.model, r.seconds_per_cv, r.cer,
                 r.wer, 100 * r.years_ok, 100 * r.ranges_ok, 100 * r.cvs_all_ranges_ok)
    return 0


def una_imagen(ruta: Path, salida: Path, modelo: str) -> int:
    """Modo usado por la app de Streamlit: corre en un proceso separado del de PyTorch."""
    ocr, disp = crear_ocr(modelo)
    doc = ocr_imagen(ocr, ruta)
    salida.write_text(json.dumps({"model": modelo, "hardware": hardware(disp), **doc},
                                 ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", choices=list(MODELOS), default=MODELO)
    p.add_argument("--benchmark", type=int, metavar="N",
                   help="Compara los modelos en N CVs y termina")
    p.add_argument("--imagen", type=Path, help="Procesa una sola imagen")
    p.add_argument("--salida", type=Path, help="JSON de salida para --imagen")
    p.add_argument("--force", action="store_true", help="Reprocesa aunque exista el JSON")
    p.add_argument("--ruidoso", action="store_true",
                   help=f"Pasada extra con el modelo rapido ({MODELO_RUIDOSO}) a "
                        "data/ocr_mobile/, para la Pregunta 2")
    args = p.parse_args()

    if args.imagen:
        return una_imagen(args.imagen, args.salida or args.imagen.with_suffix(".json"),
                          args.modelo)
    log = configurar_logging("ocr_extraction")
    OCR.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        return benchmark(args.benchmark)
    if args.ruidoso:
        return extraer_todo(MODELO_RUIDOSO, args.force, OCR_RUIDOSO)
    return extraer_todo(args.modelo, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
