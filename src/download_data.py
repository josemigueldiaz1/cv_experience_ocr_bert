"""
Etapa 0 — Muestreo estratificado y descarga de CVs desde PopResume.

Que hace
--------
1. Lee la configuracion `text` del dataset (5 parquet pequenos, ~13 MB) para conocer la
   poblacion completa: id, ocupacion, experiencia y atributos.
2. Para cada ocupacion elige, con semilla fija, bloques de 100 imagenes SIN foto y de ese
   conjunto sortea los N CVs de la muestra.
3. Guarda cada imagen tal cual (`data/raw/images/<id>.jpg`), los metadatos de la muestra
   y el texto de referencia, que solo se usa para medir la calidad del OCR.
4. Comprueba que la muestra reproduce la distribucion de experiencia de la poblacion.

Por que no se usa el ejemplo de streaming del enunciado
-------------------------------------------------------
Dentro de cada ocupacion, los shards de imagenes guardan primero TODAS las filas con foto
y despues todas las sin foto, ordenadas por id. Un `filter(has_photo == False)` en
streaming tiene que bajar y descartar primero la mitad con foto (~2 GB por ocupacion) y el
`shuffle(buffer_size=5000)` solo mezcla dentro de lo que ya leyo.

Leyendo solo las estadisticas de cada bloque (row group) se sabe, sin descargar imagenes,
que bloques son 100 % sin foto. Se eligen al azar los bloques justos para tener el doble
de la muestra, y de ese conjunto se sortean los N finales. Esto es valido porque el id
NO esta relacionado con la experiencia (Spearman entre id y exp_year < 0.03 en las cinco
ocupaciones; se comprobo antes de escribir este script). La comprobacion final de
representatividad (prueba KS contra la poblacion) queda en
`output/tables/sampling_check.csv`.

Resultado: ~1.2 GB de descarga en lugar de ~8 GB, y una muestra igual de aleatoria.

Uso
---
    python src/download_data.py                  # 800 CVs por ocupacion
    python src/download_data.py --n 400          # minimo que permite el enunciado
    python src/download_data.py --force          # vuelve a descargar aunque exista
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (CAP, IMAGES, N_POR_OCUPACION, OCUPACIONES, RAW, SEED, DATA,  # noqa: E402
                       TABLES, asegurar_carpetas, configurar_logging, guardar_json)

REPO = "datasets/sumin-yu/PopResume"
FILAS_POR_BLOQUE = 100

# Columnas de la configuracion `text` que se conservan. Los atributos sensibles se guardan
# solo en `data/raw/` (que no se sube) y solo para dos usos permitidos: la prueba de
# equidad de la Pregunta 4 (sex, race) y borrar el nombre del texto (first/last_name).
# NINGUNO se usa como feature.
COLUMNAS_META = ["id", "job", "exp_year", "sex", "race", "age", "state", "region",
                 "edu_level", "edu_group", "first_name", "last_name"]

ARCHIVO_IDS = DATA / "sampled_ids.csv"          # se sube a GitHub: es la muestra
ARCHIVO_META = RAW / "metadata.csv"             # atributos de la muestra (no se sube)
ARCHIVO_REF = RAW / "reference_text.parquet"    # texto de referencia (no se sube)

log = None


def leer_poblacion(fs) -> pd.DataFrame:
    """La configuracion `text` completa: una fila por CV de las 5 ocupaciones."""
    partes = []
    for job in OCUPACIONES:
        with fs.open(f"{REPO}/data/text/{job}.parquet") as f:
            partes.append(pd.read_parquet(f))
    pob = pd.concat(partes, ignore_index=True)
    log.info("Poblacion: %d CVs en %d ocupaciones", len(pob), pob.job.nunique())
    return pob


def bloques_sin_foto(fs, job: str) -> list[tuple[str, int]]:
    """
    Lista (shard, indice_de_bloque) de los bloques que son 100 % sin foto.

    Usa solo los metadatos del parquet (min/max de `has_photo` por bloque), asi que no
    descarga ninguna imagen. Los bloques mixtos de la frontera se ignoran.
    """
    import pyarrow.parquet as pq

    shards = sorted(fs.glob(f"{REPO}/data/images/{job}-*.parquet"))
    bloques = []
    for shard in shards:
        with fs.open(shard) as f:
            meta = pq.ParquetFile(f).metadata
            col = meta.schema.names.index("has_photo")
            for i in range(meta.num_row_groups):
                st = meta.row_group(i).column(col).statistics
                if st is not None and st.has_min_max and not st.min and not st.max:
                    bloques.append((shard, i))
    return bloques


def descargar_ocupacion(fs, job: str, n: int, rng: np.random.Generator,
                        force: bool) -> pd.DataFrame:
    """Sortea y descarga los n CVs de una ocupacion. Devuelve (id, job, archivo)."""
    import pyarrow.parquet as pq

    bloques = bloques_sin_foto(fs, job)
    necesarios = int(np.ceil(2 * n / FILAS_POR_BLOQUE))
    elegidos = sorted(rng.choice(len(bloques), size=min(necesarios, len(bloques)),
                                 replace=False))
    log.info("  %s: %d bloques sin foto disponibles, se leen %d", job, len(bloques),
             len(elegidos))

    # Primero solo los ids de los bloques elegidos (columna diminuta), para sortear la
    # muestra antes de tocar una sola imagen.
    candidatos = []
    por_shard: dict[str, list[int]] = {}
    for k in elegidos:
        shard, rg = bloques[k]
        por_shard.setdefault(shard, []).append(rg)
    for shard, rgs in por_shard.items():
        with fs.open(shard) as f:
            t = pq.ParquetFile(f).read_row_groups(rgs, columns=["id", "has_photo"])
        for id_, foto in zip(t.column("id").to_pylist(), t.column("has_photo").to_pylist()):
            if not foto:
                candidatos.append((id_, shard))
    candidatos.sort()
    idx = rng.choice(len(candidatos), size=n, replace=False)
    muestra = {candidatos[i][0] for i in idx}

    # Ahora si, las imagenes. Solo se escriben las de la muestra que aun no existen.
    filas = []
    for shard, rgs in por_shard.items():
        pendientes = [c[0] for c in candidatos if c[1] == shard and c[0] in muestra
                      and (force or not (IMAGES / f"{c[0]}.jpg").exists())]
        if pendientes:
            with fs.open(shard) as f:
                t = pq.ParquetFile(f).read_row_groups(rgs, columns=["id", "image"])
            for id_, img in zip(t.column("id").to_pylist(), t.column("image").to_pylist()):
                if id_ in pendientes:
                    (IMAGES / f"{id_}.jpg").write_bytes(img["bytes"])
    for id_ in sorted(muestra):
        filas.append({"id": id_, "job": job, "image": f"data/raw/images/{id_}.jpg"})
    return pd.DataFrame(filas)


def comprobar_representatividad(pob: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Compara la experiencia de la muestra con la de la poblacion, por ocupacion."""
    from scipy.stats import ks_2samp

    filas = []
    for job in OCUPACIONES:
        p = pob.loc[pob.job == job, "exp_year"]
        m = meta.loc[meta.job == job, "exp_year"]
        ks = ks_2samp(p, m)
        filas.append({
            "occupation": job, "population": len(p), "sample": len(m),
            "pop_mean": p.mean(), "sample_mean": m.mean(),
            "pop_median": p.median(), "sample_median": m.median(),
            "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
        })
    tabla = pd.DataFrame(filas).round(3)
    for _, r in tabla.iterrows():
        log.info("  %-35s media pob %.2f vs muestra %.2f | KS p=%.3f", r.occupation,
                 r.pop_mean, r.sample_mean, r.ks_pvalue)
    return tabla


def main() -> int:
    global log
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=N_POR_OCUPACION, help="CVs por ocupacion")
    p.add_argument("--force", action="store_true", help="Vuelve a descargar todo")
    args = p.parse_args()

    log = configurar_logging("download_data")
    asegurar_carpetas()

    # Reanudable: si la muestra ya existe con el mismo tamano y todas las imagenes estan
    # en disco, no hay nada que hacer.
    if ARCHIVO_IDS.exists() and ARCHIVO_META.exists() and not args.force:
        ids = pd.read_csv(ARCHIVO_IDS, dtype=str)
        completos = all((IMAGES / f"{i}.jpg").exists() for i in ids.id)
        if len(ids) == args.n * len(OCUPACIONES) and completos:
            log.info("Muestra ya descargada (%d CVs). Nada que hacer.", len(ids))
            return 0

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    t0 = time.perf_counter()
    try:
        pob = leer_poblacion(fs)
    except Exception as e:  # el dataset es "gated": sin login falla aqui
        log.error("No se pudo leer PopResume: %s", e)
        log.error("Acepta las condiciones en https://huggingface.co/datasets/sumin-yu/PopResume")
        log.error("y luego ejecuta:  hf auth login")
        return 1

    # Una semilla por ocupacion derivada de la global: anadir o quitar una ocupacion no
    # cambia la muestra de las demas.
    partes = []
    for k, job in enumerate(OCUPACIONES):
        rng = np.random.default_rng([SEED, k])
        partes.append(descargar_ocupacion(fs, job, args.n, rng, args.force))
    muestra = pd.concat(partes, ignore_index=True)

    sel = pob[pob.id.isin(muestra.id)]
    meta = muestra.merge(sel[COLUMNAS_META].drop(columns="job"), on="id", how="left")
    if meta.exp_year.isna().any():
        log.error("%d CVs de la muestra no aparecen en la configuracion text",
                  meta.exp_year.isna().sum())
        return 1

    # Variable objetivo: tope en CAP ("CAP o mas").
    meta["y"] = meta.exp_year.clip(0, CAP)
    en_tope = int((meta.exp_year >= CAP).sum())
    pct_tope = 100 * en_tope / len(meta)
    log.info("CVs en el tope de %d anios: %d (%.2f %%). Umbral del enunciado: 15 %%",
             CAP, en_tope, pct_tope)

    muestra[["id", "job"]].to_csv(ARCHIVO_IDS, index=False)
    meta.to_csv(ARCHIVO_META, index=False)
    sel[["id", "text"]].to_parquet(ARCHIVO_REF, index=False)

    TABLES.mkdir(parents=True, exist_ok=True)
    tabla = comprobar_representatividad(pob, meta)
    tabla.to_csv(TABLES / "sampling_check.csv", index=False)
    guardar_json({
        "seed": SEED, "n_per_occupation": args.n, "total": len(meta),
        "cap": CAP, "at_cap": en_tope, "at_cap_pct": round(pct_tope, 3),
        "population_at_cap_pct": round(100 * float((pob.exp_year >= CAP).mean()), 3),
        "download_minutes": round((time.perf_counter() - t0) / 60, 2),
    }, TABLES.parent / "metrics" / "sampling.json")

    log.info("Listo: %d imagenes en %s (%.1f min)", len(meta), IMAGES,
             (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
