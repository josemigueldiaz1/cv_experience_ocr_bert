"""
Etapa 3 — Particion train / validacion / test (70 / 15 / 15).

- Por id de CV: cada CV esta en una sola particion.
- Estratificada por ocupacion x banda de experiencia, para que las tres particiones
  tengan la misma mezcla de enfermeras juniors, contables seniors, etc. Sin esto, con
  pocos CVs en la banda 11-20, el test podria quedarse casi sin seniors.
- Semilla fija. Los ids se guardan en `data/splits/` (y SI se suben a GitHub): todos los
  modelos leen estos mismos archivos.

El test no se mira hasta la evaluacion final. Todo ajuste se hace con validacion.

Uso
---
    python src/split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (METRICS, PROCESSED, SEED, SPLITS, banda, configurar_logging,  # noqa: E402
                       guardar_json)


def cargar_split(nombre: str, variante: str | None = None) -> pd.DataFrame:
    """
    Devuelve una particion con su texto. Lo usan todos los modelos, para que ninguno
    pueda leer otra cosa que los ids guardados aqui.
    """
    ids = pd.read_csv(SPLITS / f"{nombre}.csv", dtype={"id": str})
    if variante is None:
        return ids
    textos = pd.read_parquet(PROCESSED / "texts.parquet")
    df = ids.merge(textos[["id", variante]], on="id", how="left")
    df = df.rename(columns={variante: "text"})
    if df.text.isna().any():
        raise ValueError(f"{df.text.isna().sum()} CVs de '{nombre}' sin texto '{variante}'")
    return df


def main() -> int:
    log = configurar_logging("split")
    df = pd.read_parquet(PROCESSED / "texts.parquet", columns=["id", "job", "y"])
    df["band"] = df.y.map(banda)
    df["strata"] = df.job + "|" + df.band

    train, resto = train_test_split(df, test_size=0.30, stratify=df.strata,
                                    random_state=SEED)
    val, test = train_test_split(resto, test_size=0.50, stratify=resto.strata,
                                 random_state=SEED)

    SPLITS.mkdir(parents=True, exist_ok=True)
    cols = ["id", "job", "y", "band"]
    for nombre, parte in (("train", train), ("val", val), ("test", test)):
        parte.sort_values("id")[cols].to_csv(SPLITS / f"{nombre}.csv", index=False)

    # Comprobaciones: sin solapes y misma composicion en las tres particiones.
    assert not (set(train.id) & set(val.id) or set(train.id) & set(test.id)
                or set(val.id) & set(test.id)), "hay CVs repetidos entre particiones"
    comp = (pd.concat([train.assign(split="train"), val.assign(split="val"),
                       test.assign(split="test")])
            .groupby(["split", "band"]).size().unstack(fill_value=0))
    log.info("Tamanos: train %d | val %d | test %d", len(train), len(val), len(test))
    log.info("CVs por banda y particion:\n%s", comp.to_string())
    guardar_json({"seed": SEED, "train": len(train), "val": len(val), "test": len(test),
                  "by_band": comp.to_dict(orient="index"),
                  "y_mean": {"train": train.y.mean(), "val": val.y.mean(),
                             "test": test.y.mean()}},
                 METRICS / "split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
