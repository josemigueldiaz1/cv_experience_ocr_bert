"""
Etapa 5 — Fine-tuning de un transformer como REGRESOR de anios de experiencia.

Configuracion
-------------
- `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`: el
  transformer lee el CV completo y una capa lineal sobre su representacion devuelve UN
  numero.
- Objetivo escalado: se entrena con y/20 (rango 0-1) y la prediccion se multiplica por
  20 y se recorta a [0, 20]. Con el objetivo en anios crudos, los primeros pasos producen
  gradientes enormes que desestabilizan el ajuste fino.
- Perdida: MSE (la del enunciado). Como alternativa (Pregunta 3) se entrena con Huber,
  que castiga los errores grandes de forma lineal y no cuadratica.
- Parada temprana: tras cada epoca se mide el MAE en VALIDACION; se guardan los pesos
  de la mejor epoca y se para si no mejora en `paciencia` epocas.
- Longitud: se mide con el tokenizador real cuantos tokens tienen los CVs. Si todos caben
  en `max_len`, no hay truncado y las fechas sobreviven siempre (se comprueba y se
  guarda en el JSON del modelo). Por eso basta el truncado por la cabeza.

Bucle propio en PyTorch (sin `Trainer`) para que cada paso sea visible: tokenizar,
forward, perdida, backward, AdamW, scheduler lineal con calentamiento, bf16 en GPU.

Uso
---
    python src/train_bert.py                           # variantes + Huber (lo de run_all)
    python src/train_bert.py --seleccion               # compara backbones en validacion
    python src/train_bert.py --variante clean --loss huber
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.split import cargar_split  # noqa: E402
from src.utils import (CAP, MODELS_BERT, PREDICTIONS, SEED, TABLES, VARIANTE_PRINCIPAL,  # noqa: E402
                       VARIANTES, configurar_logging, fijar_semilla, guardar_json,
                       guardar_predicciones, leer_json, recortar)

# Backbones candidatos (todos "ligeros", de la lista del enunciado). El principal se
# elige con `--seleccion` por MAE de validacion; el resultado queda en BACKBONE_JSON.
BACKBONES = {
    "modernbert": "answerdotai/ModernBERT-base",
    "distilbert": "distilbert-base-uncased",
    "deberta-small": "microsoft/deberta-v3-small",
}
BACKBONE_JSON = MODELS_BERT / "backbone.json"

# Paciencia 8 y no 4: la tasa de aprendizaje baja linealmente hasta 0 en la ultima epoca y
# el MAE de validacion oscila de una epoca a otra; ModernBERT se asienta recien en las
# ultimas epocas. Con paciencia 4 la parada temprana corto un entrenamiento en la epoca 11
# (MAE val 0.158) cuando la misma receta llega a ~0.11 si completa el calendario.
CONFIG = dict(lr=5e-5, batch=16, epocas=20, paciencia=8, max_len=256, warmup=0.06,
              weight_decay=0.01, huber_delta_anios=1.0, seed=SEED)

log = None


# ================================================================== datos y tokens
def tokenizar(tok, textos: list[str], max_len: int) -> dict:
    """
    Truncado por la cabeza con un max_len que cubre el 100 % de los CVs (se verifica en
    `estadisticas_tokens`). Con CVs de ~150 tokens, cabeza+cola o trozos darian
    exactamente lo mismo: no hay nada que cortar.
    """
    enc = tok(textos, truncation=True, max_length=max_len, padding=True,
              return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def estadisticas_tokens(tok, textos: list[str], max_len: int) -> dict:
    from src.ocr_quality import rangos

    largos = np.array([len(i) + 2 for i in tok(textos, add_special_tokens=False)["input_ids"]])
    # Las fechas sobreviven si el ultimo rango de fechas cae dentro de los tokens que
    # se conservan. Sin truncado, sobreviven siempre.
    truncados = int((largos > max_len).sum())
    con_rangos = sum(1 for t in textos if rangos(t))
    return {"tokens_mean": float(largos.mean()), "tokens_p95": float(np.percentile(largos, 95)),
            "tokens_max": int(largos.max()), "max_len": max_len, "cvs_truncated": truncados,
            "cvs_with_date_ranges": con_rangos,
            "dates_survive_truncation": truncados == 0}


# ============================================================================ modelo
def _dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predecir_lote(modelo, tok, textos: list[str], max_len: int, batch: int = 64) -> np.ndarray:
    import torch

    disp = next(modelo.parameters()).device
    modelo.eval()
    salida = []
    with torch.no_grad(), torch.autocast(device_type=disp.type, dtype=torch.bfloat16,
                                         enabled=disp.type == "cuda"):
        for i in range(0, len(textos), batch):
            enc = tokenizar(tok, textos[i:i + batch], max_len)
            enc = {k: v.to(disp) for k, v in enc.items()}
            salida.append(modelo(**enc).logits.float().squeeze(-1).cpu().numpy())
    return recortar(np.concatenate(salida) * CAP)


def entrenar(backbone: str, variante: str, loss: str = "mse", nombre: str | None = None,
             guardar: bool = True, **over) -> dict:
    """Entrena un modelo y devuelve su resumen. Guarda pesos y predicciones si `guardar`."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    cfg = {**CONFIG, **over}
    fijar_semilla(cfg["seed"])
    disp = _dispositivo()
    hf = BACKBONES[backbone]
    nombre = nombre or ("bert" if loss == "mse" else f"bert_{loss}")

    train, val, test = (cargar_split(s, variante) for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(hf)
    stats = estadisticas_tokens(tok, train.text.tolist() + val.text.tolist()
                                + test.text.tolist(), cfg["max_len"])
    # Pesos maestros SIEMPRE en fp32. Transformers 5 carga cada checkpoint en el tipo con
    # que se guardo, y deberta-v3-small esta guardado en fp16: AdamW sobre pesos fp16
    # produce NaN desde el primer paso (su epsilon de 1e-8 se redondea a 0 en fp16). El
    # calculo sigue en bf16 via autocast; solo los pesos que se actualizan van en fp32.
    modelo = AutoModelForSequenceClassification.from_pretrained(
        hf, num_labels=1, problem_type="regression", dtype=torch.float32).to(disp)

    if loss == "mse":
        criterio = torch.nn.MSELoss()
    else:
        criterio = torch.nn.HuberLoss(delta=cfg["huber_delta_anios"] / CAP)

    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    pasos_epoca = math.ceil(len(train) / cfg["batch"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * pasos_epoca * cfg["epocas"]),
                                            pasos_epoca * cfg["epocas"])

    textos = train.text.tolist()
    y = torch.tensor(train.y.values / CAP, dtype=torch.float32)
    gen = np.random.default_rng(cfg["seed"])
    mejor = (np.inf, None, -1)
    curva = []
    t0 = time.perf_counter()
    for epoca in range(1, cfg["epocas"] + 1):
        modelo.train()
        orden = gen.permutation(len(train))
        perdidas = []
        for i in range(0, len(orden), cfg["batch"]):
            idx = orden[i:i + cfg["batch"]]
            enc = tokenizar(tok, [textos[j] for j in idx], cfg["max_len"])
            enc = {k: v.to(disp) for k, v in enc.items()}
            with torch.autocast(device_type=disp.type, dtype=torch.bfloat16,
                                enabled=disp.type == "cuda"):
                pred = modelo(**enc).logits.squeeze(-1)
            perdida = criterio(pred.float(), y[idx].to(disp))
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            perdidas.append(perdida.item())

        p_val = predecir_lote(modelo, tok, val.text.tolist(), cfg["max_len"])
        mae_val = float(np.mean(np.abs(p_val - val.y.values)))
        curva.append({"epoch": epoca, "train_loss": float(np.mean(perdidas)),
                      "val_mae": mae_val, "seconds": time.perf_counter() - t0})
        log.info("    epoca %2d  perdida %.5f  MAE val %.3f", epoca, np.mean(perdidas), mae_val)
        if mae_val < mejor[0]:
            mejor = (mae_val, copy.deepcopy(modelo.state_dict()), epoca)
        elif epoca - mejor[2] >= cfg["paciencia"]:
            log.info("    parada temprana: %d epocas sin mejorar", cfg["paciencia"])
            break
    t_entreno = time.perf_counter() - t0
    modelo.load_state_dict(mejor[1])

    resumen = {"model": nombre, "variant": variante, "backbone": hf, "loss": loss,
               "best_epoch": mejor[2], "val_mae": mejor[0], "epochs_run": len(curva),
               "train_seconds": t_entreno, "hardware": _hardware(), **cfg, **stats}
    pd.DataFrame(curva).to_csv(TABLES / f"bert_curve__{nombre}__{variante}__{backbone}.csv",
                               index=False)
    if not guardar:
        return resumen

    destino = MODELS_BERT / f"{nombre}__{variante}"
    modelo.save_pretrained(destino)
    tok.save_pretrained(destino)
    guardar_json(resumen, destino / "training_summary.json")

    t1 = time.perf_counter()
    p_test = predecir_lote(modelo, tok, test.text.tolist(), cfg["max_len"])
    resumen["inference_seconds_per_cv"] = (time.perf_counter() - t1) / len(test)
    p_val = predecir_lote(modelo, tok, val.text.tolist(), cfg["max_len"])
    df = pd.concat([val.assign(split="val", pred=p_val), test.assign(split="test", pred=p_test)])
    guardar_predicciones(nombre, variante, df, resumen)
    return resumen


def _hardware() -> str:
    import torch
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})"
    return f"CPU (torch {torch.__version__})"


def cargar(variante: str = VARIANTE_PRINCIPAL, nombre: str = "bert"):
    """Devuelve `predecir(textos) -> np.ndarray` con el modelo guardado."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ruta = MODELS_BERT / f"{nombre}__{variante}"
    tok = AutoTokenizer.from_pretrained(ruta)
    modelo = AutoModelForSequenceClassification.from_pretrained(ruta).to(_dispositivo())
    max_len = leer_json(ruta / "training_summary.json")["max_len"]
    return lambda textos: predecir_lote(modelo, tok, list(textos), max_len)


# ============================================================================== main
SELECCION_CSV = TABLES / "bert_backbone_selection.csv"


def seleccion() -> str:
    """
    Compara los backbones en VALIDACION (variante principal, MSE) y elige el de menor MAE.

    - Reanudable: un backbone que ya tiene fila en la tabla de seleccion no se reentrena.
    - Cada candidato se entrena y se GUARDA como `sel_<backbone>`. El ganador se renombra
      a `bert__<variante principal>` y los pesos de los demas se borran: el entrenamiento
      del modelo principal no se repite despues de la seleccion.
    """
    import shutil

    previas = pd.read_csv(SELECCION_CSV) if SELECCION_CSV.exists() else pd.DataFrame()
    filas = previas.to_dict(orient="records")
    hechas = set(previas.backbone) if len(previas) else set()
    for clave in BACKBONES:
        if clave in hechas:
            log.info("  backbone %s ya evaluado: se reutiliza su fila", clave)
            continue
        log.info("  backbone %s", clave)
        try:
            r = entrenar(clave, VARIANTE_PRINCIPAL, nombre=f"sel_{clave}")
        except Exception as e:  # p. ej. un tokenizador que pide una dependencia extra
            log.warning("  %s no se pudo entrenar: %s", clave, e)
            continue
        filas.append({"backbone": clave, "hf_model": BACKBONES[clave], "val_mae": r["val_mae"],
                      "best_epoch": r["best_epoch"], "train_seconds": r["train_seconds"],
                      "tokens_mean": r["tokens_mean"], "tokens_max": r["tokens_max"]})
        pd.DataFrame(filas).sort_values("val_mae").to_csv(SELECCION_CSV, index=False)
    tabla = pd.DataFrame(filas).sort_values("val_mae")
    tabla.to_csv(SELECCION_CSV, index=False)
    ganador = str(tabla.iloc[0].backbone)
    guardar_json({"backbone": ganador, "hf_model": BACKBONES[ganador]}, BACKBONE_JSON)
    for _, r in tabla.iterrows():
        log.info("  %-14s MAE val %.3f (%.0f s)", r.backbone, r.val_mae, r.train_seconds)

    # El ganador pasa a ser el modelo principal, con sus predicciones y su curva.
    v = VARIANTE_PRINCIPAL
    origen, destino = MODELS_BERT / f"sel_{ganador}__{v}", MODELS_BERT / f"bert__{v}"
    if origen.exists():
        # Se mueven los ARCHIVOS dentro de `destino` en lugar de borrar la carpeta y
        # renombrar: en Windows, OneDrive o el explorador pueden tener abierta la carpeta
        # y borrarla falla con "Access is denied"; vaciarla y llenarla no.
        destino.mkdir(parents=True, exist_ok=True)
        for f in destino.iterdir():
            shutil.rmtree(f) if f.is_dir() else f.unlink()
        for f in origen.iterdir():
            shutil.move(str(f), str(destino / f.name))
        shutil.rmtree(origen, ignore_errors=True)
        for ruta in (destino / "training_summary.json", PREDICTIONS / f"sel_{ganador}__{v}.json"):
            if ruta.exists():
                guardar_json({**leer_json(ruta), "model": "bert"}, ruta)
        for ext in ("csv", "json"):
            fuente = PREDICTIONS / f"sel_{ganador}__{v}.{ext}"
            if fuente.exists():
                fuente.replace(PREDICTIONS / f"bert__{v}.{ext}")
        curva = TABLES / f"bert_curve__sel_{ganador}__{v}__{ganador}.csv"
        if curva.exists():
            curva.replace(TABLES / f"bert_curve__bert__{v}__{ganador}.csv")
    for clave in BACKBONES:
        sobra = MODELS_BERT / f"sel_{clave}__{v}"
        if sobra.exists():
            shutil.rmtree(sobra, ignore_errors=True)
            if sobra.exists():
                log.warning("  no se pudo borrar %s (abierta por otro programa); "
                            "puede borrarse a mano", sobra)
        for ext in ("csv", "json"):
            (PREDICTIONS / f"sel_{clave}__{v}.{ext}").unlink(missing_ok=True)
    return ganador


def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seleccion", action="store_true",
                   help="Compara backbones en validacion antes de entrenar")
    p.add_argument("--backbone", choices=list(BACKBONES))
    p.add_argument("--variante", choices=VARIANTES)
    p.add_argument("--loss", choices=["mse", "huber"])
    p.add_argument("--force", action="store_true", help="Reentrena aunque ya exista")
    args = p.parse_args()
    log = configurar_logging("train_bert")
    MODELS_BERT.mkdir(parents=True, exist_ok=True)

    if args.seleccion or (not args.backbone and not BACKBONE_JSON.exists()):
        backbone = seleccion()
    else:
        backbone = args.backbone or leer_json(BACKBONE_JSON)["backbone"]
    log.info("Backbone: %s (%s)", backbone, BACKBONES[backbone])

    # Lo que entrena run_all: las 3 variantes de texto con MSE + Huber sobre la principal.
    trabajos = [(v, "mse") for v in VARIANTES] + [(VARIANTE_PRINCIPAL, "huber")]
    if args.variante or args.loss:
        trabajos = [(args.variante or VARIANTE_PRINCIPAL, args.loss or "mse")]
    for variante, loss in trabajos:
        nombre = "bert" if loss == "mse" else f"bert_{loss}"
        resumen = MODELS_BERT / f"{nombre}__{variante}" / "training_summary.json"
        # Un modelo ya entrenado se reutiliza solo si es del backbone elegido: si la
        # seleccion cambia de ganador, los modelos del backbone anterior se rehacen.
        if resumen.exists() and not args.force \
                and leer_json(resumen).get("backbone") == BACKBONES[backbone]:
            log.info("  %s__%s ya entrenado (usa --force para repetir)", nombre, variante)
            continue
        log.info("  entrenando %s sobre '%s'", nombre, variante)
        r = entrenar(backbone, variante, loss)
        log.info("  -> MAE val %.3f en la epoca %d (%.0f s); tokens max %d, truncados %d",
                 r["val_mae"], r["best_epoch"], r["train_seconds"], r["tokens_max"],
                 r["cvs_truncated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
