"""
cv_experience_ocr_bert — Genera TODA la tarea en una sola corrida.

Es el script para reproducir el proyecto entero en una maquina limpia, despues de
instalar las dependencias (ver README) y de iniciar sesion en Hugging Face:

    python run_all.py

Hace, en orden:

    0  comprueba el entorno (Python, librerias, GPU, sesion de Hugging Face)
    1  descarga la muestra estratificada de PopResume (4,000 CVs sin foto)
    2  PaddleOCR sobre las 4,000 imagenes (v6) y pasada extra con el modelo rapido
    3  calidad del OCR (CER, WER, anios y rangos de fechas bien leidos)
    4  orden de lectura, limpieza, borrado de datos personales y enmascarado
    5  particion train / validacion / test (70/15/15)
    6  baselines: mediana, TF-IDF + Ridge, embeddings + Ridge, y reglas (bonus)
    7  fine-tuning de BERT como regresor
    8  evaluacion, bootstrap, analisis de errores y las 4 preguntas
    9  figuras

Opciones utiles:

    python run_all.py --desde 6        retoma desde la etapa 6
    python run_all.py --solo 8         solo la evaluacion
    python run_all.py --n 400          muestra minima (400 CVs por ocupacion)

Es REANUDABLE: lo ya descargado no se vuelve a descargar, el OCR ya hecho no se repite
y los modelos BERT ya entrenados no se reentrenan. Una segunda corrida tarda minutos
(los baselines y la evaluacion si se recalculan, porque son rapidos).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src.utils import (METRICS, TABLES, configurar_logging, leer_json)  # noqa: E402

PY = sys.executable
log = logging.getLogger("run_all")


def banner(texto: str, char: str = "=") -> None:
    log.info("")
    log.info(char * 78)
    log.info(texto)
    log.info(char * 78)


def correr(descripcion: str, *args: str) -> int:
    """
    Ejecuta un script en un subproceso y transmite su salida.

    Subprocesos y no imports por una razon concreta: PaddlePaddle y PyTorch traen DLL de
    CUDA de versiones distintas y no pueden convivir en un mismo proceso (ver
    README, "Decisiones tecnicas"). Cada etapa en su propio proceso elimina el problema.
    """
    log.info("")
    log.info("-> %s", descripcion)
    log.info("   python %s", " ".join(args))
    t0 = time.perf_counter()
    p = subprocess.Popen([PY, *args], cwd=str(RAIZ), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace")
    ultimas = []
    for linea in p.stdout:
        linea = linea.rstrip()
        if not linea:
            continue
        ultimas = (ultimas + [linea])[-30:]
        # Solo se retransmiten las lineas de log de los scripts; el ruido de las
        # librerias (barras de progreso, avisos de Paddle) queda fuera de la consola.
        if "[INFO" in linea or "[WARN" in linea or "[ERROR" in linea:
            log.info("   | %s", linea[linea.find("]") + 2:][:220])
    p.wait()
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        for l in ultimas:
            log.error("   ! %s", l[:220])
    log.info("   %s en %s", "OK" if p.returncode == 0 else f"FALLO ({p.returncode})",
             f"{dt:.1f} s" if dt < 120 else f"{dt / 60:.1f} min")
    return p.returncode


# ======================================================================== comprobaciones
def comprobar_entorno() -> int:
    """Fallar aqui, con un mensaje claro, ahorra descubrirlo a mitad del OCR."""
    banner("0 · COMPROBACIONES PREVIAS")
    problemas = 0
    log.info("Python: %s", sys.version.split()[0])
    if sys.version_info < (3, 10):
        log.error("  se necesita Python 3.10 o superior")
        problemas += 1

    # find_spec comprueba que la libreria esta instalada SIN importarla (importar torch y
    # paddle aqui los haria chocar).
    import importlib.util
    faltan = [mod for mod in ("pandas", "sklearn", "scipy", "jiwer", "matplotlib", "seaborn",
                              "joblib", "datasets", "huggingface_hub", "transformers",
                              "sentence_transformers", "streamlit", "paddleocr", "paddle",
                              "torch")
              if importlib.util.find_spec(mod) is None]
    if faltan:
        log.error("  faltan dependencias: %s", ", ".join(faltan))
        log.error("  instalalas siguiendo el README (PyTorch y Paddle van con indice propio)")
        problemas += 1
    else:
        log.info("Dependencias: todas presentes")

    # GPU: se pregunta a cada libreria en su propio proceso (no pueden convivir).
    torch_gpu = subprocess.run([PY, "-c", "import torch;print(torch.cuda.is_available())"],
                               capture_output=True, text=True).stdout.strip()
    paddle_gpu = subprocess.run(
        [PY, "-c", "import paddle;print(paddle.device.cuda.device_count()>0)"],
        capture_output=True, text=True).stdout.strip().splitlines()
    log.info("GPU para PyTorch (BERT, embeddings): %s", torch_gpu or "?")
    log.info("GPU para PaddlePaddle (OCR):         %s", paddle_gpu[-1] if paddle_gpu else "?")
    if torch_gpu != "True":
        log.warning("  sin GPU para PyTorch: BERT entrenara en CPU (mucho mas lento)")

    from huggingface_hub import get_token
    if get_token():
        log.info("Hugging Face: sesion iniciada")
    else:
        log.warning("Hugging Face: SIN sesion. Si aun no hay datos descargados, ejecuta:")
        log.warning("    hf auth login")
        log.warning("  (y acepta las condiciones en huggingface.co/datasets/sumin-yu/PopResume)")
    return problemas


def resumen_final() -> None:
    banner("RESULTADOS")
    import pandas as pd

    comp = TABLES / "model_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        log.info("Tabla comparativa (test):")
        for _, r in df.iterrows():
            log.info("  %-22s MAE %.3f | ±2 anios %5.1f%% | bandas %5.1f%% | R² %6.3f | "
                     "entreno %s", r.Model, r["MAE (years)"], 100 * r["Acc ±2"],
                     100 * r["Band accuracy"], r["R²"], r["Training time"])
    gan = TABLES / "bert_gain_bootstrap.csv"
    if gan.exists():
        log.info("")
        log.info("Ganancia de BERT (reduccion de MAE, IC 95 por ciento, bootstrap pareado):")
        for _, r in pd.read_csv(gan).iterrows():
            log.info("  vs %-22s %+.3f anios [%+.3f, %+.3f] %s", r.baseline, r.gain, r.ci_low,
                     r.ci_high, "significativa" if r.significant else "no significativa")
    ocr = METRICS / "ocr_quality.json"
    if ocr.exists():
        q = leer_json(ocr)
        log.info("")
        log.info("OCR: CER medio %.4f%% | anios bien leidos %.2f%% | rangos intactos %.2f%%",
                 100 * q["cer_mean"], q["years_read_ok_pct"], q["ranges_read_ok_pct"])


# ================================================================================= main
ETAPAS = {
    1: ("DESCARGA DE DATOS", [("Muestreo estratificado y descarga", "src/download_data.py")]),
    2: ("OCR", [("PaddleOCR v6 sobre todos los CVs", "src/ocr_extraction.py"),
                ("Pasada con el OCR rapido (solo para la Pregunta 2)",
                 "src/ocr_extraction.py", "--ruidoso")]),
    3: ("CALIDAD DEL OCR", [("CER, WER, anios y rangos de fechas", "src/ocr_quality.py")]),
    4: ("PREPROCESAMIENTO", [("Orden de lectura, limpieza y enmascarado",
                              "src/preprocessing.py")]),
    5: ("PARTICION", [("Train / validacion / test", "src/split.py")]),
    6: ("BASELINES", [("Mediana", "src/baseline_median.py"),
                      ("TF-IDF + Ridge", "src/baseline_tfidf.py"),
                      ("Embeddings + Ridge", "src/baseline_embeddings.py"),
                      ("Reglas de fechas (bonus)", "src/baseline_rules.py")]),
    7: ("BERT", [("Fine-tuning como regresor", "src/train_bert.py")]),
    8: ("EVALUACION", [("Comparacion, bootstrap, errores y preguntas", "src/evaluation.py")]),
    9: ("FIGURAS", [("Figuras para el README y la app", "src/visualization.py")]),
}


def main() -> int:
    p = argparse.ArgumentParser(description="Genera todo el proyecto en una sola corrida",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--desde", type=int, default=1, choices=list(ETAPAS))
    p.add_argument("--solo", type=int, choices=list(ETAPAS))
    p.add_argument("--n", type=int, help="CVs por ocupacion (por defecto 800)")
    args = p.parse_args()

    configurar_logging("run_all")
    t0 = time.perf_counter()
    banner("cv_experience_ocr_bert — anios de experiencia desde imagenes de CV", "#")
    log.info("Jose Miguel Diaz · Data-Science-Python · HW_04")

    if comprobar_entorno():
        log.error("Hay problemas en el entorno. Corrigelos y vuelve a ejecutar.")
        return 1

    etapas = [args.solo] if args.solo else [e for e in ETAPAS if e >= args.desde]
    for e in etapas:
        titulo, pasos = ETAPAS[e]
        banner(f"{e} · {titulo}")
        for descripcion, *cmd in pasos:
            if e == 1 and args.n:
                cmd = [*cmd, "--n", str(args.n)]
            if correr(descripcion, *cmd) != 0:
                log.error("La etapa %d fallo. Corrige y retoma con:  python run_all.py --desde %d",
                          e, e)
                return 1

    resumen_final()
    banner(f"COMPLETADO en {(time.perf_counter() - t0) / 60:.1f} min", "#")
    log.info("Para abrir la aplicacion:   streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
