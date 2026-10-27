"""
App de Streamlit — 4 pestanas.

    streamlit run app.py

1. Problema y metodologia       objetivo, datos, OCR, modelos, particion, limitaciones
2. OCR y analisis exploratorio   distribucion de experiencia, calidad del OCR, ejemplo con cajas
3. Resultados de los modelos     tabla, dispersion, desgloses, confusiones, ganancia, preguntas
4. Estimador de experiencia      sube un CV (imagen o PDF) o explora los CVs de test

Todo lo que muestra sale de output/ (lo genera `python run_all.py`). La pestana 4 carga
los modelos entrenados de models/ y corre PaddleOCR en un PROCESO APARTE: Paddle y
PyTorch no pueden compartir proceso (ver README, "Decisiones tecnicas").
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src.utils import (CAP, FIGURES, IMAGES, METRICS, NOMBRES_BANDAS, NOMBRES_MODELOS,  # noqa: E402
                       NOMBRES_OCUPACIONES, OCR, PREDICTIONS, PROCESSED, RAW, TABLES,
                       VARIANTE_PRINCIPAL, banda, leer_json)

st.set_page_config(page_title="Estimador de experiencia en CVs", page_icon="📄", layout="wide")

COLORES = {NOMBRES_MODELOS["median"]: "#898781", NOMBRES_MODELOS["tfidf"]: "#2a78d6",
           NOMBRES_MODELOS["embeddings"]: "#eb6834", NOMBRES_MODELOS["bert"]: "#1baf7a",
           NOMBRES_MODELOS["rules"]: "#eda100"}
ORDEN = list(COLORES)
ESCALA = alt.Scale(domain=ORDEN, range=[COLORES[m] for m in ORDEN])
OCUP = NOMBRES_OCUPACIONES
GRUPOS = {"Female": "Mujeres", "Male": "Hombres", "White": "Blanca", "Black": "Negra",
          "Asian/Pacific Islander": "Asiática / isleña del Pacífico"}

# Columnas de las tablas de output/ traducidas para mostrarlas.
COLUMNAS = {
    "Model": "Modelo", "MAE (years)": "MAE (años)", "RMSE (years)": "RMSE (años)",
    "Acc ±1": "Acierto ±1", "Acc ±2": "Acierto ±2", "Acc ±3": "Acierto ±3",
    "Band accuracy": "Acierto de banda", "Band macro-F1": "F1 macro de banda",
    "Training time": "Tiempo de entrenamiento", "Inference time per CV": "Inferencia por CV",
    "model": "Modelo", "baseline": "Frente a", "mae_baseline": "MAE del otro modelo",
    "mae_bert": "MAE de BERT", "gain": "Ganancia (años)", "ci_low": "IC 95 % inferior",
    "ci_high": "IC 95 % superior", "share_resamples_bert_not_better":
        "Remuestreos donde BERT no mejora", "significant": "¿Significativa?",
    "resamples": "Remuestreos", "setting": "Configuración", "test_mae": "MAE en test",
    "attribute": "Atributo", "group": "Grupo", "n": "CVs", "true_mean": "Media real",
    "pred_mean": "Media predicha", "mae": "MAE", "mean_error": "Error medio",
    "mean_error_ci_low": "IC 95 % inferior", "mean_error_ci_high": "IC 95 % superior",
    "share_with_career_gap": "CVs con hueco ≥ 1 año", "mean_jobs": "Empleos (media)",
    "id": "CV", "job": "Ocupación", "y": "Años reales", "pred": "Predicción",
    "error": "Error", "n_jobs": "Empleos", "gap_years": "Años de hueco", "cause": "Causa",
    "text": "Texto", "cvs": "CVs", "cer_mean": "CER medio", "cer_median": "CER mediano",
    "wer_mean": "WER medio", "years_ok": "Años bien leídos", "ranges_ok": "Rangos intactos",
    "mean_conf": "Confianza media", "perfect_ranges": "CVs con todos los rangos bien",
}


def ver(df: pd.DataFrame | None, **kw) -> None:
    """Muestra una tabla con las columnas en espanol."""
    if df is None:
        return
    d = df.rename(columns=COLUMNAS)
    for c in ("Ocupación",):
        if c in d:
            d[c] = d[c].map(lambda v: OCUP.get(v, v))
    if "Grupo" in d:
        d["Grupo"] = d["Grupo"].map(lambda v: GRUPOS.get(v, v))
    if "Atributo" in d:
        d["Atributo"] = d["Atributo"].map({"sex": "sexo", "race": "raza"}).fillna(d["Atributo"])
    st.dataframe(d, hide_index=True, use_container_width=True, **kw)


# ================================================================== carga de datos
@st.cache_data
def tabla(nombre: str) -> pd.DataFrame | None:
    ruta = TABLES / nombre
    return pd.read_csv(ruta, dtype={"id": str}) if ruta.exists() else None


@st.cache_data
def metrica(nombre: str) -> dict:
    ruta = METRICS / nombre
    return leer_json(ruta) if ruta.exists() else {}


@st.cache_data
def predicciones_test() -> pd.DataFrame | None:
    """Una fila por CV de test con la prediccion de cada modelo."""
    base = None
    for m, nombre in NOMBRES_MODELOS.items():
        ruta = PREDICTIONS / f"{m}__{VARIANTE_PRINCIPAL}.csv"
        if not ruta.exists():
            continue
        df = pd.read_csv(ruta, dtype={"id": str})
        df = df[df.split == "test"][["id", "y", "pred"]].rename(columns={"pred": nombre})
        base = df if base is None else base.merge(df.drop(columns="y"), on="id")
    if base is None:
        return None
    meta = pd.read_csv(RAW / "metadata.csv", dtype={"id": str})[["id", "job"]]
    return base.merge(meta, on="id")


@st.cache_data
def textos() -> pd.DataFrame | None:
    ruta = PROCESSED / "texts.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


@st.cache_resource(show_spinner="Cargando los modelos entrenados...")
def modelos() -> dict:
    from src.evaluation import predictores
    return predictores(VARIANTE_PRINCIPAL)


def figura(nombre: str, pie: str | None = None) -> None:
    ruta = FIGURES / nombre
    if ruta.exists():
        st.image(str(ruta), use_container_width=True)
        if pie:
            st.caption(pie)
    else:
        st.info(f"No se encontró la figura `{nombre}`. Ejecuta primero `python run_all.py`.")


def cajas(cv_id: str):
    """Imagen del CV con las cajas de PaddleOCR."""
    from PIL import Image, ImageDraw
    doc = leer_json(OCR / f"{cv_id}.json")
    im = Image.open(IMAGES / f"{cv_id}.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    for l in doc["lines"]:
        d.rectangle(l["box"], outline=(42, 120, 214), width=3)
    alto = max(l["box"][3] for l in doc["lines"]) + 60
    return im.crop((0, 0, im.width, min(alto, im.height))), doc


# ============================================================================ cabecera
st.title("¿Cuántos años de experiencia? Leer imágenes de CV con PaddleOCR y BERT")
st.caption("José Miguel Díaz · Data-Science-Python · HW_04 · PopResume (EMNLP 2026, CC BY 4.0)")

if not (TABLES / "model_comparison.csv").exists():
    st.warning("Todavía no hay resultados. Ejecuta `python run_all.py` para generar datos, "
               "modelos y tablas.")

t1, t2, t3, t4 = st.tabs(["1 · Problema y metodología", "2 · OCR y análisis exploratorio",
                          "3 · Resultados de los modelos", "4 · Estimador de experiencia"])

# ================================================================== 1. metodologia
with t1:
    mu = metrica("sampling.json")
    oq = metrica("ocr_quality.json")
    oe = metrica("ocr_extraction.json")
    pr = metrica("preprocessing.json")
    sp = metrica("split.json")

    st.header("El problema")
    st.markdown(
        "Un reclutador mira un CV y estima en segundos los años de experiencia del "
        "candidato. Ese número casi nunca está escrito: hay que **deducirlo** de las fechas "
        "de cada empleo, de cuántos puestos hay y de cuándo terminaron los estudios. Este "
        "proyecto construye un sistema que lo hace a partir de la **imagen** del CV: "
        "PaddleOCR lee el texto y cuatro modelos de complejidad creciente predicen los años "
        "(de 0 a 20).")
    st.markdown(
        "**Lo difícil aquí no es encontrar la evidencia, sino usarla.** En PopResume cada "
        "empleo trae su rango de fechas (`2017.12–2020.07`), y sumar las duraciones "
        "reproduce la etiqueta con un error medio de **0.02 años**. Un modelo necesita (1) "
        "recibir los dígitos intactos del OCR y (2) hacer aritmética con ellos. Los modelos "
        "basados en palabras solo aprenden asociaciones; la pregunta es cuánto se acerca un "
        "transformer a la resta que un reclutador hace de cabeza.")

    c = st.columns(4)
    c[0].metric("CVs de la muestra", f"{mu.get('total', 0):,}")
    c[1].metric("Rangos de fechas bien leídos", f"{oq.get('ranges_read_ok_pct', 0):.2f} %")
    c[2].metric("Tiempo de OCR por CV", f"{oe.get('seconds_per_cv_mean', 0):.2f} s")
    c[3].metric("CVs de test", f"{sp.get('test', 0):,}")

    st.header("Datos")
    st.markdown(
        f"- **PopResume**: 60,884 CVs sintéticos de 5 ocupaciones de EE. UU., generados a "
        f"partir de estadísticas de población (no hay datos de personas reales), cada uno "
        f"con sus años reales de experiencia.\n"
        f"- **Muestra**: {mu.get('n_per_occupation', 800)} CVs por ocupación, solo imágenes "
        f"**sin foto**, semilla {mu.get('seed', 42)}. Los ids están en "
        f"`data/sampled_ids.csv`. El muestreo por bloques y la prueba KS contra la población "
        f"se explican en el README.\n"
        f"- **Variable objetivo**: `exp_year` recortado a 0–{CAP}. Solo {mu.get('at_cap', 0)} "
        f"CVs ({mu.get('at_cap_pct', 0):.2f} %) están en el tope, muy por debajo del 15 % que "
        f"justificaría subirlo.\n"
        f"- **Regla de enmascarado**: las frases que dan la respuesta (experiencia total, "
        f"edad, fecha de nacimiento) se enmascaran antes de modelar. Tras leer 50 textos de "
        f"OCR al azar y aplicar las regex a todos los CVs: **{pr.get('cvs_masked_direct_answer', 0)} "
        f"CVs afectados**. Las fechas de empleo y de graduación se conservan: son la evidencia.\n"
        f"- **Nunca se usan como entradas**: edad, sexo, raza, nombres, estado ni región. "
        f"Nombre, email, teléfono y ubicación se borran del texto.")

    st.header("OCR y orden de lectura")
    st.markdown(
        f"- **PaddleOCR 3.7, PP-OCRv6 medium** (detección + reconocimiento) en una "
        f"{oe.get('hardware', 'GPU')}. Elegido con una comparación sobre 50 CVs: el modelo "
        f"rápido (*mobile*) tiene un CER de solo 0.2 %, pero rompe el 42 % de los rangos de "
        f"fechas.\n"
        f"- Un JSON por CV en `data/ocr/` con el texto, la confianza y la caja de cada línea.\n"
        f"- **Orden de lectura**: las cajas se agrupan en renglones (solape vertical mayor a "
        f"la mitad de la altura de la más baja) y se leen de arriba abajo y de izquierda a "
        f"derecha, así un puesto y sus fechas quedan en la misma línea aunque el OCR los "
        f"haya separado. Un detector de dos columnas busca un canal vertical vacío entre el "
        f"30 % y el 70 % del ancho; si lo encuentra, las líneas a todo lo ancho parten la "
        f"página en franjas y en cada franja se lee primero la columna izquierda. Los CVs de "
        f"PopResume son de una columna, así que el detector nunca se activa; está para los "
        f"CVs que se suban en la pestaña 4.")

    st.header("Cuatro modelos, una sola partición")
    st.markdown(
        f"Train / validación / test = **{sp.get('train', 0)} / {sp.get('val', 0)} / "
        f"{sp.get('test', 0)}** CVs, estratificada por ocupación × banda de experiencia, "
        f"semilla 42. Todos los modelos leen los mismos archivos de `data/splits/`; todo "
        f"ajuste se hace con validación.")
    st.markdown(
        "| Modelo | Intuición |\n|---|---|\n"
        "| **Mediana** | Siempre dice el valor típico. Es el piso: un modelo que no le gana no aprendió nada. |\n"
        "| **TF-IDF + Ridge** | Cuenta qué palabras y pares de palabras aparecen. Ve `2017` y `2020` como fichas sueltas, sin relación entre sí. |\n"
        "| **Embeddings + Ridge** | Un codificador congelado resume *de qué trata* el CV en un vector. |\n"
        "| **BERT afinado** | Un transformer lee el CV completo y se entrena de punta a punta para devolver los años. |\n"
        "| *Reglas de fechas (bonus)* | Extrae cada rango de fechas y suma las duraciones. Sin entrenamiento. |")

    st.header("Limitaciones")
    st.markdown(
        "- **CVs sintéticos y limpios**: un solo formato, una sola forma de escribir fechas, "
        "imágenes digitales. CVs reales (escaneos, fotos, 'Jan 2019 – Present', dos "
        "columnas) serían mucho más difíciles.\n"
        "- **Solo 5 ocupaciones, estadísticas de EE. UU., solo inglés.**\n"
        "- **La etiqueta es la suma de los empleos listados**, lo que hace la tarea más fácil "
        "que en la vida real (empleos no listados, medias jornadas).\n"
        "- **Un estimador de experiencia nunca debe decidir a quién se contrata.** Puede "
        "prellenar un perfil que una persona revisa; ver la prueba de equidad en la pestaña 3.")

# ================================================================== 2. OCR y EDA
with t2:
    meta = pd.read_csv(RAW / "metadata.csv") if (RAW / "metadata.csv").exists() else None
    st.header("Años de experiencia")
    if meta is not None:
        meta["Ocupación"] = meta.job.map(OCUP)
        hist = alt.Chart(meta).mark_bar(color="#3987e5", cornerRadiusTopLeft=3,
                                        cornerRadiusTopRight=3).encode(
            x=alt.X("y:Q", bin=alt.Bin(step=1, extent=[0, CAP + 1]),
                    title="años (tope en 20)"),
            y=alt.Y("count():Q", title="CVs"),
            tooltip=[alt.Tooltip("count():Q", title="CVs")]).properties(height=260)
        caja = alt.Chart(meta).mark_boxplot(color="#3987e5", size=18, extent="min-max").encode(
            x=alt.X("y:Q", title="años"), y=alt.Y("Ocupación:N", title=None)).properties(height=260)
        c1, c2 = st.columns([1.2, 1])
        c1.altair_chart(hist, use_container_width=True)
        c2.altair_chart(caja, use_container_width=True)
        en_tope = int((meta.exp_year >= CAP).sum())
        st.caption(
            f"Por qué estos gráficos: el histograma muestra la forma que los modelos deben "
            f"aprender (una cola larga a la derecha; la mayoría tiene menos de 10 años), y "
            f"las cajas muestran que las ocupaciones se parecen bastante, así que la ocupación "
            f"por sí sola no predice mucho. {en_tope} CVs ({100 * en_tope / len(meta):.2f} %) "
            f"están en el tope de 20 años.")

    st.header("Calidad del OCR")
    oq = metrica("ocr_quality.json")
    c = st.columns(4)
    c[0].metric("CER medio", f"{100 * oq.get('cer_mean', 0):.3f} %")
    c[1].metric("WER medio", f"{100 * oq.get('wer_mean', 0):.3f} %")
    c[2].metric("Años bien leídos", f"{oq.get('years_read_ok_pct', 0):.2f} %")
    c[3].metric("Rangos de fechas intactos", f"{oq.get('ranges_read_ok_pct', 0):.2f} %")
    figura("ocr_quality.png",
           "Por qué estos gráficos: el CER solo esconde lo que importa. El panel del medio "
           "mide lo único de lo que depende la etiqueta, los rangos de fechas, por ocupación. "
           "El de la derecha es la comparación con la que se eligió el modelo de OCR: el más "
           "rápido parece excelente por su CER, pero rompe 4 de cada 10 rangos de fechas.")
    ver(tabla("ocr_quality_by_occupation.csv"))

    st.header("Ejemplo: lo que ve PaddleOCR")
    tx = textos()
    if tx is not None:
        ids = tx.id.tolist()
        cv = st.selectbox("CV", ids, index=0, key="eda_cv")
        try:
            im, doc = cajas(cv)
            c1, c2 = st.columns([1.1, 1])
            c1.image(im, caption=f"{len(doc['lines'])} líneas de texto detectadas (cajas azules)",
                     use_container_width=True)
            fila = tx[tx.id == cv].iloc[0]
            c2.markdown("**Texto que reciben los modelos** (limpio, sin datos personales)")
            c2.code(fila.clean, language=None)
            c2.markdown(f"Experiencia real: **{fila.y:.2f} años**")
        except FileNotFoundError:
            st.info("Las imágenes no están en esta copia del repositorio (no se suben a "
                    "GitHub). Ejecuta `python run_all.py` para descargarlas.")

# ================================================================== 3. resultados
with t3:
    st.header("Comparación de modelos (conjunto de test)")
    comp = tabla("model_comparison.csv")
    if comp is not None:
        ver(comp.round(3))
        st.caption("MAE = cuántos años se equivoca en promedio. Acierto ±k = proporción de "
                   "CVs a k años o menos del valor real. Bandas: 0–2, 3–5, 6–10, 11–20. El "
                   "tiempo de entrenamiento de BERT incluye todas sus épocas en la GPU.")

    pt = predicciones_test()
    st.header("Predicho frente a real")
    if pt is not None:
        modelo_sel = st.selectbox("Modelo", [m for m in ORDEN if m in pt.columns],
                                  index=ORDEN.index(NOMBRES_MODELOS["bert"]))
        d = pt.assign(pred=pt[modelo_sel], error=pt[modelo_sel] - pt.y,
                      Ocupación=pt.job.map(OCUP))
        puntos = alt.Chart(d).mark_circle(size=40, opacity=0.6, color=COLORES[modelo_sel],
                                          stroke="#fcfcfb", strokeWidth=1).encode(
            x=alt.X("y:Q", title="años reales", scale=alt.Scale(domain=[0, CAP])),
            y=alt.Y("pred:Q", title="años predichos", scale=alt.Scale(domain=[0, CAP])),
            tooltip=[alt.Tooltip("id:N", title="CV"), "Ocupación:N",
                     alt.Tooltip("y:Q", title="real", format=".2f"),
                     alt.Tooltip("pred:Q", title="predicho", format=".2f"),
                     alt.Tooltip("error:Q", title="error", format="+.2f")])
        diag = alt.Chart(pd.DataFrame({"x": [0, CAP], "y": [0, CAP]})).mark_line(
            color="#c3c2b7", strokeWidth=1).encode(x="x:Q", y="y:Q")
        st.altair_chart((diag + puntos).properties(height=420), use_container_width=True)
        st.caption("Línea gris: predicción perfecta. Los puntos por encima son "
                   "sobreestimaciones. Una nube que se aplana hacia el centro indica que el "
                   "modelo tira hacia la media (sobreestima a los juniors y subestima a los "
                   "seniors).")
        figura("pred_vs_true.png")

    st.header("Error por banda de experiencia y por ocupación")
    for archivo, grupo, titulo in (("mae_by_band.csv", "band", "banda real"),
                                   ("mae_by_occupation.csv", "occupation", "ocupación")):
        df = tabla(archivo)
        if df is None:
            continue
        if grupo == "occupation":
            df = df.assign(occupation=df.occupation.map(OCUP))
        orden_x = NOMBRES_BANDAS if grupo == "band" else list(OCUP.values())
        ch = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{grupo}:N", sort=orden_x, title=titulo),
            xOffset=alt.XOffset("model:N", sort=ORDEN),
            y=alt.Y("mae:Q", title="MAE (años)"),
            color=alt.Color("model:N", scale=ESCALA, sort=ORDEN, title=None,
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("model:N", title="modelo"),
                     alt.Tooltip(f"{grupo}:N", title=titulo), alt.Tooltip("n:Q", title="CVs"),
                     alt.Tooltip("mae:Q", title="MAE", format=".3f")]
        ).properties(height=300)
        st.altair_chart(ch, use_container_width=True)

    st.header("Matrices de confusión por bandas")
    cols = st.columns(len(NOMBRES_MODELOS))
    for c, (m, nombre) in zip(cols, NOMBRES_MODELOS.items()):
        with c:
            figura(f"confusion_{m}.png")

    st.header("¿La ganancia de BERT es real? Bootstrap pareado, IC 95 %")
    figura("bert_gain.png")
    ver(tabla("bert_gain_bootstrap.csv"))

    st.header("Pregunta 1 — ¿El modelo lee las fechas?")
    figura("probe.png", "20 CVs de test modificados de tres maneras con respuesta conocida. "
                        "Un modelo que lee las fechas se mueve como la línea negra; uno que "
                        "lee vocabulario, no.")
    st.header("Pregunta 2 — ¿Cuánto cuesta el ruido del OCR?")
    figura("ocr_noise_cost.png")
    ver(tabla("ocr_noise_cost.csv"))
    st.header("Pregunta 3 — Sensibilidad a las decisiones metodológicas")
    figura("sensitivity.png")
    ver(tabla("sensitivity.csv"))
    st.header("Pregunta 4 — Prueba de equidad")
    figura("fairness.png", "El sexo y la raza NO son entradas del modelo; se usan solo aquí, "
                           "para revisar si los errores difieren entre grupos.")
    ver(tabla("fairness.csv"))
    st.header("Los 10 errores más grandes de BERT")
    ver(tabla("bert_top_errors.csv"))

# ================================================================== 4. estimador
with t4:
    st.header("Sube un CV")
    st.caption("PNG/JPG o PDF (primera página). PaddleOCR corre en un proceso aparte; luego "
               "se aplica la misma limpieza que en el entrenamiento y después cada modelo.")
    sub = st.file_uploader("Imagen o PDF del CV", type=["png", "jpg", "jpeg", "pdf"])
    if sub is not None:
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / sub.name
            ruta.write_bytes(sub.getvalue())
            if ruta.suffix.lower() == ".pdf":
                import fitz  # PyMuPDF
                pag = fitz.open(ruta)[0]
                ruta = ruta.with_suffix(".png")
                pag.get_pixmap(dpi=150).save(ruta)
            salida = Path(tmp) / "ocr.json"
            with st.spinner("Ejecutando PaddleOCR..."):
                r = subprocess.run([sys.executable, "src/ocr_extraction.py", "--imagen",
                                    str(ruta), "--salida", str(salida)], cwd=str(RAIZ),
                                   capture_output=True, text=True)
            if r.returncode != 0 or not salida.exists():
                st.error("Falló el OCR:\n" + (r.stderr or "")[-1500:])
            else:
                from src.preprocessing import procesar
                doc = json.loads(salida.read_text(encoding="utf-8"))
                # Sin metadatos: el nombre se toma del primer renglon (la cabecera).
                primero = min(doc["lines"], key=lambda l: l["box"][1])["text"].split()
                out = procesar(doc["lines"], primero[0] if primero else "",
                               primero[-1] if len(primero) > 1 else "", 0.9, doc["width"])
                c1, c2 = st.columns([1, 1])
                c1.image(str(ruta), use_container_width=True)
                c2.markdown(f"**Texto extraído** ({len(doc['lines'])} líneas, "
                            f"{doc['seconds']:.2f} s en {doc['hardware']})")
                c2.code(out["clean"], language=None)
                preds = {NOMBRES_MODELOS[m]: float(f([out["clean"]])[0])
                         for m, f in modelos().items()}
                st.subheader("Años de experiencia predichos")
                cs = st.columns(len(preds))
                for c, (m, v) in zip(cs, preds.items()):
                    c.metric(m, f"{v:.1f} años", help=f"banda {banda(v)}")

    st.header("Explora los CVs de test")
    pt = predicciones_test()
    tx = textos()
    if pt is not None and tx is not None:
        c1, c2, c3 = st.columns(3)
        ocup = c1.multiselect("Ocupación", list(OCUP.values()), default=list(OCUP.values()))
        rango = c2.slider("Años reales", 0.0, float(CAP), (0.0, float(CAP)), step=0.5)
        err_min = c3.slider("Error de BERT de al menos (años)", 0.0, 5.0, 0.0, step=0.25)
        bert = NOMBRES_MODELOS["bert"]
        d = pt.assign(Ocupación=pt.job.map(OCUP), **{"Error de BERT": (pt[bert] - pt.y).abs()})
        d = d[d["Ocupación"].isin(ocup) & d.y.between(*rango) & (d["Error de BERT"] >= err_min)]
        cols = ["id", "Ocupación", "y"] + [m for m in ORDEN if m in d.columns] + ["Error de BERT"]
        st.dataframe(d[cols].rename(columns={"id": "CV", "y": "Años reales"})
                     .sort_values("Error de BERT", ascending=False).round(2),
                     hide_index=True, use_container_width=True, height=260)
        if len(d):
            cv = st.selectbox("Abrir un CV", d.sort_values("Error de BERT", ascending=False).id)
            fila = tx[tx.id == cv].iloc[0]
            c1, c2 = st.columns([1.1, 1])
            try:
                im, _ = cajas(cv)
                c1.image(im, use_container_width=True)
            except FileNotFoundError:
                c1.info("La imagen no está disponible en esta copia.")
            c2.code(fila.clean, language=None)
            r = d[d.id == cv].iloc[0]
            c2.markdown(f"Real: **{r.y:.2f} años** · " + " · ".join(
                f"{m}: **{r[m]:.2f}**" for m in ORDEN if m in r.index))

    st.header("Buenas y malas predicciones, explicadas")
    e = tabla("bert_top_errors.csv")
    if pt is not None and e is not None:
        bert = NOMBRES_MODELOS["bert"]
        # Entre 3 y 15 anios: en el tope (20) el acierto es trivial, porque las
        # predicciones se recortan a 20.
        mejor = (pt[pt.y.between(3, 15)].assign(err=lambda d: (d[bert] - d.y).abs())
                 .sort_values("err").iloc[0])
        st.markdown(f"**Buena** — CV `{mejor.id}`: {mejor.y:.2f} años reales, BERT "
                    f"{mejor[bert]:.2f}. Sus fechas se leyeron bien y el total cae en la zona "
                    f"más poblada de los datos de entrenamiento, donde BERT vio muchos ejemplos "
                    f"parecidos.")
        con_hueco = e[e.cause == "hueco entre empleos"]
        peor = con_hueco.iloc[0] if len(con_hueco) else e.iloc[0]
        st.markdown(f"**Mala** — CV `{peor.id}`: {peor.y:.2f} años reales, BERT {peor.pred:.2f} "
                    f"({peor.error:+.2f}). Causa: *{peor.cause}*: hay {peor.gap_years:.1f} "
                    f"años sin trabajar entre dos empleos, y BERT no los descuenta bien. La "
                    f"lista completa de los 10 errores más grandes está en la pestaña 3.")
        st.code(peor.text, language=None)
