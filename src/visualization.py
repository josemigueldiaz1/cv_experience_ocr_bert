"""
Figuras del proyecto (output/figures/).

Lee las tablas que dejan las etapas anteriores y dibuja. No calcula metricas: si una
cifra aparece en una figura, viene de un CSV de output/tables/ y se puede comprobar.

Convenciones
------------
- Cada modelo tiene SIEMPRE el mismo color, en todas las figuras y en la app
  (`COLORES`). La mediana va en gris: es la referencia, no un competidor.
- Texto, ejes y leyendas en tinta neutra; el color solo lo llevan las marcas.
- Rejilla de un pixel, gris claro, sin trazos discontinuos. Barras finas.
- Las magnitudes ordenadas (nivel de ruido, variante de limpieza) usan tonos de un solo
  azul, de claro a oscuro.

Uso
---
    python src/visualization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (CAP, FIGURES, IMAGES, NOMBRES_BANDAS, NOMBRES_MODELOS,  # noqa: E402
                       NOMBRES_OCUPACIONES, OCR, OCUPACIONES, PREDICTIONS, RAW, TABLES,
                       VARIANTE_PRINCIPAL, configurar_logging, leer_json)

# ------------------------------------------------------------------ paleta y tinta
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

# Slots 1-4 de la paleta categorica de referencia, en su orden validado (los pares
# adyacentes superan los umbrales de daltonismo). La mediana, gris neutro.
COLORES = {
    NOMBRES_MODELOS["median"]: "#898781",
    NOMBRES_MODELOS["tfidf"]: "#2a78d6",
    NOMBRES_MODELOS["embeddings"]: "#eb6834",
    NOMBRES_MODELOS["bert"]: "#1baf7a",
    NOMBRES_MODELOS["rules"]: "#eda100",
}
ORDEN = list(COLORES)
AZUL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # rampa ordinal (claro → oscuro)

# Etiquetas cortas para ejes estrechos.
ETIQUETA_OCUP = {
    "registered_nurses": "Enfermería", "elementary_middle_school_teachers": "Docentes",
    "software_developers": "Software", "accountants_auditors": "Contabilidad",
    "construction_laborers": "Construcción",
}
GRUPOS = {"Female": "Mujeres", "Male": "Hombres", "White": "Blanca", "Black": "Negra",
          "Asian/Pacific Islander": "Asiática / isleña del Pacífico"}
ATRIBUTOS = {"sex": "sexo", "race": "raza"}

log = None


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE, "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10, "text.color": TINTA, "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2, "axes.grid": True, "grid.color": REJILLA,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.labelcolor": TINTA_2, "figure.dpi": 110,
        "axes.formatter.use_locale": False,
    })


def guardar(fig, nombre: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / nombre, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s", nombre)


def _tabla(nombre: str) -> pd.DataFrame | None:
    ruta = TABLES / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def barras_agrupadas(ax, df: pd.DataFrame, grupo: str, serie: str, valor: str,
                     orden_grupos, orden_series, colores, etiquetas: bool = False,
                     fmt: str = "{:.2f}") -> None:
    """Barras finas agrupadas, con 2 px de aire entre barras vecinas."""
    n = len(orden_series)
    ancho = min(0.8 / n, 0.16)
    x = np.arange(len(orden_grupos))
    for k, s in enumerate(orden_series):
        vals = [df.loc[(df[grupo] == g) & (df[serie] == s), valor].mean() for g in orden_grupos]
        pos = x + (k - (n - 1) / 2) * ancho
        ax.bar(pos, vals, width=ancho * 0.9, color=colores[k], label=s, linewidth=0)
        if etiquetas:
            for px, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(px, v, fmt.format(v), ha="center", va="bottom", fontsize=7,
                            color=TINTA_2)
    ax.set_xticks(x, orden_grupos)
    ax.grid(axis="x", visible=False)


# ============================================================================ figuras
def fig_distribucion() -> None:
    meta = pd.read_csv(RAW / "metadata.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    ax.hist(meta.y, bins=np.arange(0, CAP + 1.5, 1), color=AZUL[1], rwidth=0.85)
    en_tope = int((meta.exp_year >= CAP).sum())
    ax.set_title("Años de experiencia en los 4,000 CVs de la muestra")
    ax.set_xlabel("años de experiencia (tope en 20)")
    ax.set_ylabel("CVs")
    ax.annotate(f"{en_tope} CVs en el tope ({100 * en_tope / len(meta):.1f} %)",
                xy=(CAP + 0.5, max(en_tope, 5)), xytext=(12.5, ax.get_ylim()[1] * 0.45),
                color=TINTA_2, fontsize=9, arrowprops=dict(arrowstyle="-", color=TENUE))
    ax = axs[1]
    datos = [meta.loc[meta.job == j, "y"] for j in OCUPACIONES]
    bp = ax.boxplot(datos, orientation="horizontal", widths=0.45, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=TINTA, linewidth=1.5))
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=AZUL[1])
    for k in ("whiskers", "caps"):
        for w in bp[k]:
            w.set(color=AZUL[1])
    ax.set_yticks(range(1, len(OCUPACIONES) + 1), [ETIQUETA_OCUP[j] for j in OCUPACIONES])
    ax.set_title("Por ocupación")
    ax.set_xlabel("años")
    ax.grid(axis="y", visible=False)
    guardar(fig, "experience_distribution.png")


def fig_calidad_ocr() -> None:
    cv = _tabla("ocr_quality_per_cv.csv")
    oc = _tabla("ocr_quality_by_occupation.csv")
    bench = _tabla("ocr_benchmark.csv")
    if cv is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axs[0]
    ax.hist(cv.cer * 100, bins=30, color=AZUL[1], rwidth=0.85)
    ax.set_title("Tasa de error por carácter (CER) por CV")
    ax.set_xlabel("CER (%)")
    ax.set_ylabel("CVs")
    ax.text(0.97, 0.9, f"media {100 * cv.cer.mean():.3f} %\n{(cv.cer == 0).mean():.0%} de los CVs sin errores",
            transform=ax.transAxes, ha="right", va="top", color=TINTA_2, fontsize=9)
    ax = axs[1]
    oc = oc[oc.job != "ALL"]
    y = np.arange(len(oc))
    ax.barh(y, oc.ranges_ok * 100, color=AZUL[1], height=0.45)
    for yy, v in zip(y, oc.ranges_ok * 100):
        ax.text(v, yy, f" {v:.2f} %", va="center", fontsize=8, color=TINTA_2)
    ax.set_yticks(y, [ETIQUETA_OCUP.get(j, j) for j in oc.job])
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title("Rangos de fechas leídos exactos")
    ax.set_xlabel("% de rangos de fechas (AAAA.MM–AAAA.MM)")
    ax.grid(axis="y", visible=False)
    q = leer_json(TABLES.parent / "metrics" / "ocr_quality.json")
    if q.get("ranges_cut_in_image"):
        ax.text(0.02, -0.32, f"Los {q['ranges_cut_in_image']} rangos rotos vienen cortados en la "
                f"propia imagen (títulos\nlargos de docentes), no mal leídos: "
                f"{q['ranges_misread_by_ocr']} errores del OCR en total.",
                transform=ax.transAxes, fontsize=8, color=TINTA_2, va="top")
    ax = axs[2]
    if bench is not None:
        etiquetas = {"v6_medium": "PP-OCRv6 medium", "v5_server": "PP-OCRv5 server",
                     "v5_mobile_en": "PP-OCRv5 mobile"}
        bench = bench.sort_values("ranges_ok")
        y = np.arange(len(bench))
        ax.barh(y, bench.ranges_ok * 100, color=AZUL[1], height=0.45)
        for yy, (_, r) in zip(y, bench.iterrows()):
            ax.text(r.ranges_ok * 100, yy, f" {100 * r.ranges_ok:.0f} %  ·  {r.seconds_per_cv:.2f} s/CV",
                    va="center", fontsize=8, color=TINTA_2)
        ax.set_yticks(y, [etiquetas.get(m, m) for m in bench.model])
        ax.set_xlim(0, 150)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title("Comparación de modelos de OCR (50 CVs)")
        ax.set_xlabel("% de rangos de fechas intactos")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    guardar(fig, "ocr_quality.png")


def fig_cajas_ocr(cv_id: str | None = None) -> None:
    """Un CV con las cajas de PaddleOCR dibujadas y su texto al lado."""
    from PIL import Image, ImageDraw

    from src.preprocessing import texto_en_orden

    textos = pd.read_parquet(Path(RAW.parent / "processed" / "texts.parquet"))
    if cv_id is None:
        # Un CV con 4 empleos: el que mas fechas tiene que leer.
        n = textos.clean.str.count(r"\d{4}\.\d{2}-\d{4}\.\d{2}")
        cv_id = textos.loc[n == n.max(), "id"].iloc[0]
    doc = leer_json(OCR / f"{cv_id}.json")
    im = Image.open(IMAGES / f"{cv_id}.jpg").convert("RGB")
    dib = ImageDraw.Draw(im)
    for l in doc["lines"]:
        dib.rectangle(l["box"], outline=(42, 120, 214), width=3)
    alto = max(l["box"][3] for l in doc["lines"]) + 60
    ancho = max(l["box"][2] for l in doc["lines"]) + 60
    im = im.crop((0, 0, min(ancho, im.width), min(alto, im.height)))
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.1, 1]})
    axs[0].imshow(im)
    axs[0].set_title(f"CV {cv_id}: PaddleOCR detectó {len(doc['lines'])} líneas de texto")
    axs[0].axis("off")
    axs[1].axis("off")
    axs[1].set_title("Texto extraído, en el orden de lectura reconstruido")
    axs[1].text(0, 1, texto_en_orden(doc["lines"], doc["width"]), va="top", ha="left",
                family="monospace", fontsize=7.5, color=TINTA, transform=axs[1].transAxes,
                wrap=True)
    guardar(fig, "ocr_boxes_example.png")


def fig_pred_vs_real() -> None:
    modelos = [m for m in NOMBRES_MODELOS if (PREDICTIONS / f"{m}__{VARIANTE_PRINCIPAL}.csv").exists()]
    fig, axs = plt.subplots(1, len(modelos), figsize=(3.3 * len(modelos), 3.5), sharey=True)
    for ax, m in zip(np.atleast_1d(axs), modelos):
        df = pd.read_csv(PREDICTIONS / f"{m}__{VARIANTE_PRINCIPAL}.csv")
        df = df[df.split == "test"]
        nombre = NOMBRES_MODELOS[m]
        ax.plot([0, CAP], [0, CAP], color=EJE, linewidth=1, zorder=1)
        ax.scatter(df.y, df.pred, s=10, color=COLORES[nombre], alpha=0.55, linewidths=0,
                   zorder=2)
        # Media de la prediccion por anio real: si la curva se aplana hacia el centro,
        # el modelo "tira a la media" (sobreestima juniors, subestima seniors).
        g = df.assign(b=df.y.round()).groupby("b").pred.mean()
        ax.plot(g.index, g.values, color=TINTA, linewidth=1.5, zorder=3)
        mae = np.mean(np.abs(df.pred - df.y))
        ax.set_title(f"{nombre}\nMAE {mae:.2f} años", fontsize=10)
        ax.set_xlim(-0.5, CAP + 0.5)
        ax.set_ylim(-0.5, CAP + 0.5)
        ax.set_xlabel("años reales")
        ax.set_aspect("equal")
    np.atleast_1d(axs)[0].set_ylabel("años predichos")
    fig.suptitle("Predicho frente a real en el conjunto de test. Línea gris: predicción "
                 "perfecta; línea negra: predicción media por año real", x=0.01, ha="left",
                 fontsize=9, color=TINTA_2, y=0.02)
    guardar(fig, "pred_vs_true.png")


def fig_mae_desglose() -> None:
    for archivo, grupo, etiqueta, orden in (
            ("mae_by_band.csv", "band", "banda de experiencia real (años)", NOMBRES_BANDAS),
            ("mae_by_occupation.csv", "occupation", "ocupación", OCUPACIONES)):
        df = _tabla(archivo)
        if df is None:
            continue
        series = [m for m in ORDEN if m in set(df.model)]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        barras_agrupadas(ax, df, grupo, "model", "mae", orden, series,
                         [COLORES[s] for s in series])
        if grupo == "occupation":
            ax.set_xticks(range(len(orden)), [NOMBRES_OCUPACIONES[o] for o in orden])
        ax.set_ylabel("MAE (años)")
        ax.set_xlabel(etiqueta)
        ax.set_title(f"Error absoluto medio por {'banda de experiencia' if grupo == 'band' else 'ocupación'}")
        ax.legend(ncol=len(series), loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8)
        guardar(fig, archivo.replace(".csv", ".png"))


def fig_confusiones() -> None:
    for m, nombre in NOMBRES_MODELOS.items():
        df = _tabla(f"confusion_{m}.csv")
        if df is None:
            continue
        import seaborn as sns

        cm = df.set_index(df.columns[0]).values
        fig, ax = plt.subplots(figsize=(3.8, 3.3))
        # El color es la proporcion de cada fila (que parte de la banda real acaba en
        # cada banda predicha); el numero escrito es el conteo de CVs.
        filas = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(filas, annot=cm, fmt="d", cbar=False, vmin=0, vmax=1, ax=ax,
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        "azul", ["#f0efec", "#cde2fb", "#3987e5", "#0d366b"]),
                    linewidths=2, linecolor=SUPERFICIE, annot_kws={"fontsize": 9},
                    xticklabels=NOMBRES_BANDAS, yticklabels=NOMBRES_BANDAS)
        ax.set_xlabel("banda predicha")
        ax.set_ylabel("banda real")
        ax.set_title(nombre, fontsize=10)
        ax.grid(False)
        guardar(fig, f"confusion_{m}.png")


def fig_ganancia() -> None:
    df = _tabla("bert_gain_bootstrap.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(df)))
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color=EJE, linewidth=1)
    ax.hlines(y, df.ci_low, df.ci_high, color=COLORES[NOMBRES_MODELOS["bert"]], linewidth=2)
    ax.scatter(df.gain, y, s=40, color=COLORES[NOMBRES_MODELOS["bert"]], zorder=3,
               edgecolors=SUPERFICIE, linewidths=2)
    for yy, (_, r) in zip(y, df.iterrows()):
        ax.text(max(r.ci_high, 0), yy + 0.18, f"  {r.gain:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]",
                fontsize=8, color=TINTA_2, va="bottom")
    ax.set_yticks(y, [f"frente a {b}" for b in df.baseline])
    ax.set_xlabel("reducción del MAE lograda por BERT (años); positivo = BERT es mejor")
    ax.set_title("Ganancia de BERT frente a cada modelo, IC 95 % por bootstrap pareado")
    ax.grid(axis="y", visible=False)
    guardar(fig, "bert_gain.png")


def fig_ruido() -> None:
    from src.evaluation import FUENTES_RUIDO

    df = _tabla("ocr_noise_cost.csv")
    if df is None:
        return
    fuentes = list(FUENTES_RUIDO)
    largo = df.melt(id_vars="model", value_vars=fuentes, var_name="source", value_name="mae")
    modelos = [m for m in ORDEN if m in set(df.model)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    barras_agrupadas(ax, largo, "model", "source", "mae", modelos, fuentes, AZUL[:3],
                     etiquetas=True)
    ax.set_ylabel("MAE en test (años)")
    ax.set_title("¿Cuánto cuesta el ruido del OCR? Los mismos modelos, tres versiones del texto")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8)
    guardar(fig, "ocr_noise_cost.png")


def fig_sensibilidad() -> None:
    df = _tabla("sensitivity.csv")
    if df is None:
        return
    base = df[df.setting.str.match(r"^texto: \w+$")].copy()
    base["variant"] = base.setting.str.replace("texto: ", "", regex=False)
    modelos = [m for m in ORDEN if m in set(base.model)]
    variantes = ["raw", "clean", "clean_conf"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    barras_agrupadas(ax, base, "model", "variant", "test_mae", modelos, variantes, AZUL[:3],
                     etiquetas=True)
    ax.set_ylabel("MAE en test (años)")
    ax.set_title("Sensibilidad a la limpieza del OCR (cada modelo reentrenado con cada variante)")
    ax.legend(["OCR sin limpiar", "OCR limpio", "limpio y sin renglones de confianza baja"],
              ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8)
    guardar(fig, "sensitivity.png")


def fig_equidad() -> None:
    df = _tabla("fairness.csv")
    if df is None:
        return
    df = df[df.model == NOMBRES_MODELOS["bert"]]
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.45 * len(df)))
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color=EJE, linewidth=1)
    c = COLORES[NOMBRES_MODELOS["bert"]]
    ax.hlines(y, df.mean_error_ci_low, df.mean_error_ci_high, color=c, linewidth=2)
    ax.scatter(df.mean_error, y, s=40, color=c, zorder=3, edgecolors=SUPERFICIE, linewidths=2)
    ax.set_yticks(y, [f"{ATRIBUTOS[a]}: {GRUPOS.get(g, g)}  (n={n}, MAE {mae:.2f})"
                      for a, g, n, mae in zip(df.attribute, df.group, df.n, df.mae)])
    ax.set_xlabel("error medio, predicho − real (años); < 0 = subestima")
    ax.set_title("BERT afinado: error medio por grupo, IC 95 % por bootstrap")
    ax.grid(axis="y", visible=False)
    guardar(fig, "fairness.png")


def fig_sonda() -> None:
    from src.evaluation import SONDAS

    df = _tabla("probe_results.csv")
    if df is None:
        return
    titulos = {SONDAS[0]: "Todas las fechas +5 años\n(reclutador: no cambia)",
               SONDAS[1]: "Se borra el empleo más antiguo\n(reclutador: − su duración)",
               SONDAS[2]: "El empleo más antiguo empieza 3 años antes\n(reclutador: +3 años)"}
    modelos = [m for m in ORDEN if m in set(df.model)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, (p, titulo) in zip(axs, titulos.items()):
        d = df[df.probe == p]
        esperado = d.expected_change.mean()
        vals = [d.loc[d.model == m, "model_change"].mean() for m in modelos]
        x = np.arange(len(modelos))
        ax.bar(x, vals, width=0.5, color=[COLORES[m] for m in modelos])
        ax.axhline(esperado, color=TINTA, linewidth=1.2)
        # A la izquierda, sobre la mediana (que no se mueve): ahi nunca tapa una barra.
        # Debajo de la linea si es >= 0 (arriba esta el titulo) y encima si es negativa
        # (abajo esta el eje).
        ax.text(-0.45, esperado, f"reclutador {esperado:+.2f}", ha="left",
                va="top" if esperado >= 0 else "bottom", fontsize=8, color=TINTA)
        ax.axhline(0, color=EJE, linewidth=0.8)
        ax.set_xticks(x, [m.replace(" + ", "\n+ ").replace(" (bonus)", "") for m in modelos],
                      fontsize=7)
        ax.set_title(titulo, fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("cambio medio de la predicción (años)")
    fig.suptitle("Sonda en 20 CVs de test: ¿la predicción se mueve como la de un reclutador?",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    guardar(fig, "probe.png")


def fig_curva_bert() -> None:
    curvas = sorted(TABLES.glob(f"bert_curve__bert__{VARIANTE_PRINCIPAL}__*.csv"))
    if not curvas:
        return
    df = pd.read_csv(curvas[-1])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    c = COLORES[NOMBRES_MODELOS["bert"]]
    ax.plot(df.epoch, df.val_mae, color=c, linewidth=2, marker="o", markersize=4)
    mejor = df.loc[df.val_mae.idxmin()]
    ax.scatter([mejor.epoch], [mejor.val_mae], s=70, color=c, edgecolors=SUPERFICIE,
               linewidths=2, zorder=3)
    ax.annotate(f"mejor: época {int(mejor.epoch)}, MAE {mejor.val_mae:.3f}",
                (mejor.epoch, mejor.val_mae), textcoords="offset points", xytext=(-150, 25),
                fontsize=8, color=TINTA_2)
    ax.set_xlabel("época")
    ax.set_ylabel("MAE en validación (años)")
    ax.set_title("Ajuste fino de BERT: MAE de validación por época")
    guardar(fig, "bert_training_curve.png")


def main() -> int:
    global log
    log = configurar_logging("visualization")
    estilo()
    for f in (fig_distribucion, fig_calidad_ocr, fig_cajas_ocr, fig_pred_vs_real,
              fig_mae_desglose, fig_confusiones, fig_ganancia, fig_ruido, fig_sensibilidad,
              fig_equidad, fig_sonda, fig_curva_bert):
        try:
            f()
        except Exception as e:  # una figura rota no debe impedir las demas
            log.warning("  %s fallo: %s", f.__name__, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
