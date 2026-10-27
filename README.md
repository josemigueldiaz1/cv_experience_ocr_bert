# cv_experience_ocr_bert

**José Miguel Díaz** · Data-Science-Python · [HW_04_202602 (issue #191)](https://github.com/d2cml-ai/Data-Science-Python/issues/191)

Sistema que lee la **imagen** de un CV con **PaddleOCR**, reconstruye su texto y predice
los **años de experiencia laboral** (0 a 20) con un **BERT afinado como regresor**,
comparado con tres baselines (mediana, TF-IDF + Ridge, embeddings + Ridge) y con un
modelo de reglas que suma las fechas.

### Resultados en una mirada (600 CVs de test)

| Modelo | MAE (años) | Dentro de ±2 años | R² |
|---|---|---|---|
| Mediana | 3.54 | 34.5 % | −0.04 |
| TF-IDF + Ridge | 0.82 | 95.5 % | 0.941 |
| Embeddings + Ridge | 1.06 | 87.3 % | 0.903 |
| **BERT afinado (ModernBERT-base)** | **0.11** | **100 %** | **0.999** |
| *Reglas: sumar las fechas (bonus)* | *0.02* | *100 %* | *1.000* |

1. **PaddleOCR leyó todas las fechas.** 100 % de los años y 0 rangos de fechas mal
   leídos en 4,000 CVs. Los únicos rangos rotos vienen cortados en la propia imagen.
2. **BERT es claramente el mejor modelo entrenado**: se equivoca ~1.3 meses de media y
   le gana a TF-IDF por 0.71 años (IC 95 % [0.65, 0.76]).
3. **Pero no aprendió a restar fechas.** Si un empleo empieza 3 años antes, BERT sube su
   predicción 0.31 años; un reclutador sumaría 3. Reacciona a la estructura del CV
   (cuántos empleos hay), no a los valores de las fechas.
4. **Una regla de tres líneas le gana** (0.02 años, en milisegundos y sin GPU), porque en
   PopResume la experiencia es literalmente la suma de los rangos de fechas.
5. **BERT es el más frágil al ruido del OCR**: con un OCR peor su error se multiplica por
   5.6, mientras TF-IDF y los embeddings empeoran menos de un 10 %.
6. **Equidad:** sin usar sexo ni raza, BERT subestima levemente a todos los grupos (5 a
   18 días), algo más al grupo con más huecos entre empleos.

**¿Vale la pena BERT?** Frente a los baselines de texto, sí: es 7 veces más preciso. Frente
a sumar las fechas, no: en CVs con este formato la regla es más precisa, más rápida, más
robusta y explicable. BERT tendría sentido para CVs reales y desordenados, donde una regla
se rompe; esa es la prueba que este dataset no permite hacer.

---

## 1. Qué hace y para quién

Un reclutador mira un CV y estima en segundos cuántos años de experiencia tiene el
candidato. El número casi nunca está escrito: se **deduce** de las fechas de cada
empleo, de cuántos puestos hay y de cuándo terminaron los estudios. Este sistema hace lo
mismo a partir de la **imagen** del CV, para quien procesa CVs a escala: agencias de
selección, bolsas de trabajo, servicios públicos de empleo o equipos de RR. HH. que
reciben miles de postulaciones. El uso razonable es **prellenar el perfil** del candidato
para que una persona lo revise, nunca decidir a quién se entrevista (ver Pregunta 4).

## 2. Datos y muestreo

[PopResume](https://huggingface.co/datasets/sumin-yu/PopResume) (Yu, Park & Moon, EMNLP
2026; CC BY 4.0): 60,884 CVs sintéticos de 5 ocupaciones, generados a partir de
estadísticas de población de EE. UU. Cada uno trae sus años reales de experiencia.

- **Muestra:** 800 CVs por ocupación (4,000), **solo imágenes sin foto**, semilla 42.
  Los ids están en [`data/sampled_ids.csv`](data/sampled_ids.csv).
- **Cómo se muestreó.** Dentro de cada ocupación, las imágenes con foto están todas
  primero y las sin foto después, ordenadas por id. El ejemplo de *streaming* del
  enunciado tendría que descargar y descartar la mitad con foto (~2 GB por ocupación). En
  su lugar se leen solo los metadatos del parquet: el mínimo y el máximo de `has_photo`
  en cada bloque de 100 filas dicen qué bloques son 100 % sin foto, sin bajar una imagen.
  Se eligen al azar los bloques justos para tener 1,600 candidatos por ocupación y de ahí
  se sortean los 800. Es válido porque el id no está relacionado con la experiencia
  (Spearman entre −0.02 y 0.00 en las cinco ocupaciones). Resultado: ~1.2 GB de descarga
  en vez de ~8 GB, en 3.3 minutos.
- **Representatividad:** prueba KS de la experiencia de la muestra contra la población,
  por ocupación: p = 0.15, 0.98, 0.09, 0.94 y 0.11. Ninguna diferencia significativa
  ([`sampling_check.csv`](output/tables/sampling_check.csv)).
- **Variable objetivo:** `exp_year` recortado a 0–20. Solo **27 CVs (0.68 %)** están en
  el tope, muy por debajo del 15 % que justificaría subirlo: **se mantiene en 20**.
- **Distribución:** media de 6 a 7 años según la ocupación; la mayoría tiene menos de 10
  ([figura](output/figures/experience_distribution.png)).

**Lo que se vio al explorar, antes de modelar.** Todos los CVs tienen la misma
estructura (nombre, contacto, `WORK HISTORY` con 1 a 4 empleos, `EDUCATION`), y cada
empleo lleva su rango de fechas (`2017.12–2020.07`). Sumando las duraciones de los
rangos se obtiene `exp_year` con un error medio de **0.021 años** en los 60,884 textos de
referencia. Es decir: **la respuesta está entera en las fechas**. Un modelo no necesita
"intuir"; necesita recibir los dígitos intactos del OCR y hacer una resta. Esto cambia la
pregunta de la tarea: no es si se puede saber, sino cuánto se acerca cada modelo a una
aritmética que un reclutador hace de cabeza.

## 3. OCR con PaddleOCR

- **Modelo: PP-OCRv6 medium** (detección + reconocimiento; el predeterminado de
  PaddleOCR 3.7), con los tres módulos opcionales apagados (orientación de documento,
  corrección de curvatura y orientación de línea): las imágenes son páginas digitales
  rectas, y activarlos solo añade tiempo y riesgo.
- **Por qué ese modelo:** benchmark sobre los mismos 50 CVs (10 por ocupación), con la
  primera pasada descartada como calentamiento
  ([`ocr_benchmark.csv`](output/tables/ocr_benchmark.csv)):

  | Modelo | s/CV | CER | Años bien | Rangos de fechas intactos |
  |---|---|---|---|---|
  | **PP-OCRv6 medium** | 0.30 | **0.01 %** | **100 %** | **100 %** |
  | PP-OCRv5 server | 0.48 | 0.08 % | 98 % | 96 % |
  | PP-OCRv5 mobile (inglés) | 0.14 | 0.20 % | 83 % | 58 % |

  El modelo *mobile* tiene un CER de solo 0.2 %, pero rompe el 42 % de los rangos de
  fechas: se come el guion largo y pega las dos fechas (`2015.05–2017.12` →
  `2015.052017.12`). **Un CER bajo no garantiza que se lean bien los dígitos que
  importan**; por eso se mide también el rango de fechas completo.
- **Salida:** un JSON por CV en `data/ocr/` con cada renglón: texto, confianza y caja
  `[x1, y1, x2, y2]`, más el tiempo y el hardware.
- **Ejecución:** 4,000 CVs en una NVIDIA RTX 5060 Laptop (8 GB), **0.34 s por CV**
  (~23 min), **0 imágenes fallidas**, confianza media **0.995**.
- **Orden de lectura** (`preprocessing.texto_en_orden`). Las cajas se agrupan en
  renglones (dos cajas van juntas si se solapan en vertical más de la mitad de la altura
  de la más baja) y cada renglón se lee de izquierda a derecha: si el OCR partiera
  "Registered Nurses, X," y "2017.12–2020.07" en dos cajas, vuelven a quedar juntas y el
  puesto no se separa de sus fechas. Para **dos columnas** se busca un canal vertical
  vacío entre el 30 % y el 70 % del ancho que separe texto que convive a la misma
  altura; si existe, las líneas que lo cruzan (cabecera, títulos a todo lo ancho) parten
  la página en franjas, y en cada franja se lee primero la columna izquierda entera y
  luego la derecha. Los CVs de PopResume son de una columna (el detector nunca se
  activa), pero está para los CVs que se suban a la app.

## 4. Calidad del OCR y preprocesamiento

**Calidad** ([`ocr_quality_by_occupation.csv`](output/tables/ocr_quality_by_occupation.csv),
[figura](output/figures/ocr_quality.png)), con OCR y referencia normalizados igual
(guiones, viñetas, espacios alrededor de `|`, minúsculas; los dígitos no se tocan):

| | Valor |
|---|---|
| CER medio | 0.005 % (98 % de los CVs sin un solo error) |
| WER medio | 0.04 % |
| Años (19xx/20xx) bien leídos | **24,645 de 24,645 (100 %)** |
| Rangos de fechas intactos | 9,724 de 9,758 (99.65 %) |
| Rangos mal leídos por el OCR | **0** |

Los 34 rangos no intactos **no son errores del OCR**: vienen cortados en la propia
imagen. Todos son de profesores: "Elementary Middle School Teachers" es el título más
largo, y el generador de PopResume corta la línea al dibujarla
("…, Harmony Meadow Academy, 2020.02–2023."). PaddleOCR leyó fielmente lo que había.
En esos 34 CVs (0.85 %) la evidencia llega incompleta a cualquier modelo.

**Limpieza** (variante `clean`, la principal): guiones y viñetas unificados (el OCR
devuelve a veces `–` y a veces `-` en el mismo CV), palabras cortadas con guion al final
de renglón unidas, espacios normalizados.

**Confianza baja ≠ error.** La confianza mínima de los 57,813 renglones es 0.936: un
umbral típico de 0.90 no quitaría nada. Los renglones con menos confianza son nombres
cortos en mayúsculas ("BETH VO", "KARA YU"), y entre 0.97 y 0.98 hay puestos con fechas
perfectamente leídos. Con este OCR, **filtrar por confianza solo borra evidencia
correcta**: la variante principal no filtra. La variante `clean_conf` quita los renglones
con confianza < 0.98 (el 2 % de los renglones, el percentil 2) para medir en la
Pregunta 3 cuánto daña hacerlo.

**Datos personales.** En todas las variantes se borran nombre (en la cabecera; buscarlo
en todo el CV borraría empresas como "Hill" o "Wood"), email, teléfono y ubicación
("CA, USA"), que se reemplazan por `[NAME]`, `[EMAIL]`, `[PHONE]`, `[LOCATION]`. La regex
de teléfono se escribió para el formato real de EE. UU.: un patrón genérico de "dígitos,
puntos y guiones" también habría capturado `2017.12-2020.07`. Una salvaguarda comprueba
en los 4,000 textos de referencia que estos pasos **no alteran ningún rango de fechas**.

**Respuestas directas.** Se leyeron 50 textos de OCR al azar
([`manual_review_sample.csv`](output/tables/manual_review_sample.csv)) y se aplicaron a los
4,000 tres regex: experiencia total ("12 years of experience"), edad ("age 35",
"35 years old") y fecha de nacimiento. **CVs afectados: 0.** PopResume no escribe esas
frases (las 58 coincidencias de "born" en el dataset son *newborn*, en cursos de
enfermería, y la regex exige que "born" empiece palabra). Las fechas de empleo y de
graduación se conservan: son la evidencia.

Lo que mostró la lectura de los 50 textos: un único error de letra ("CI pipeline" →
"Cl pipeline"); guiones mezclados (justifica la limpieza); **pistas sin aritmética** (los
puestos recientes de los seniors usan "Led", "Directed", "Designed … frameworks"; los de
los juniors, "Assisted with", "Maintained"; y el número de empleos va de 1 a 4); y huecos
entre empleos, que hacen que la experiencia sea la suma de los empleos y no
"último año − primer año".

## 5. Partición

Por id de CV, **70 / 15 / 15 = 2,800 / 600 / 600**, estratificada por ocupación × banda
de experiencia (0–2, 3–5, 6–10, 11–20), semilla 42. Los ids están en
[`data/splits/`](data/splits/) y **todos los modelos leen esos mismos archivos**. Todo
ajuste (hiperparámetros, encoder, backbone, época) se hizo con validación; el test solo
se usó en la evaluación final.

## 6. Modelos

| Modelo | Cómo se construyó | Decisiones (validación) |
|---|---|---|
| **Mediana** | predice siempre la mediana de train: **5.70 años** | la constante que minimiza el MAE es la mediana |
| **TF-IDF + Ridge** | n-gramas de palabras 1–2 (+ caracteres 2–5), `min_df=2`, `sublinear_tf` | rejilla: con o sin n-gramas de caracteres × α en [0.01, 100]. Gana palabras + caracteres, α = 0.1. **Los números se conservan** como tokens: son la única evidencia; la alternativa sin dígitos va a la Pregunta 3 |
| **Embeddings + Ridge** | encoder congelado → un vector por CV → Ridge | se compararon `bge-small-en-v1.5` (MAE val 1.15), `all-MiniLM-L6-v2` (1.22) y **`e5-small-v2` (1.10)**. CVs más largos que el encoder: trozos con solape y promedio de vectores (no hizo falta en ninguno) |
| **BERT afinado** | ver abajo | |
| *Reglas (bonus)* | regex de rangos de fechas → fusionar solapes → sumar | sin parámetros; si falta el mes de fin (imagen cortada), asume enero |

**BERT como regresor** (`src/train_bert.py`):
`AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`. El
objetivo se escala a y/20 y la predicción se vuelve a años y se recorta a [0, 20]. Bucle
de entrenamiento propio en PyTorch (tokenizar, forward, pérdida, backward, AdamW,
scheduler lineal con 6 % de calentamiento), cálculo en bf16 con pesos maestros en fp32,
recorte de gradiente a 1.0. Parada temprana sobre el MAE de validación (paciencia 8,
máximo 20 épocas), quedándose con los pesos de la mejor época. La paciencia empezó en 4 y
se subió a 8 al ver las curvas: la tasa de aprendizaje baja linealmente hasta cero en la
época 20, el MAE de validación oscila de una época a otra y ModernBERT se asienta en las
últimas épocas; con paciencia 4 un entrenamiento se cortó en la época 11 (MAE val 0.158)
cuando la misma receta llega a ~0.11. Todos los modelos BERT de este repositorio
(selección incluida) se entrenaron con la receta final.

**Qué transformer.** Se compararon tres backbones "ligeros" de la lista del enunciado con
la misma receta, sobre la variante principal y midiendo en validación
([`bert_backbone_selection.csv`](output/tables/bert_backbone_selection.csv)):

| Backbone | Parámetros | MAE val | Mejor época | Tiempo |
|---|---|---|---|---|
| **ModernBERT-base** | 149 M | **0.110** | 20 | 13.0 min |
| DistilBERT (base, uncased) | 66 M | 0.138 | 20 | 4.1 min |
| DeBERTa-v3-small | 141 M | 0.369 | 3 | 5.3 min (parada en la época 11) |

Gana **ModernBERT-base**, el modelo visto en la sesión 9. DeBERTa fue inestable con esta
receta: el MAE de validación saltaba entre 0.37 y 1.06 de una época a otra. Hiperparámetros
del modelo final: lr 5e-5, lote 16, hasta 20 épocas, paciencia 8, `max_len` 256, semilla 42,
bf16, NVIDIA RTX 5060 Laptop; **13.0 minutos** de entrenamiento, mejor época 20
([`training_summary.json`](models/bert/bert__clean/training_summary.json),
[curva](output/figures/bert_training_curve.png)).

**CVs largos.** Con el tokenizador real, el CV más largo tiene **221 tokens** (media
142). Con `max_len = 256` **ningún CV se trunca** y todas las fechas sobreviven siempre;
cabeza, cabeza + cola o trozos darían exactamente lo mismo. Se usa truncado por la
cabeza, y la comprobación queda en el `training_summary.json` de cada modelo.

## 7. Evaluación (600 CVs de test)

Tabla completa en [`model_comparison.csv`](output/tables/model_comparison.csv). Tiempo de
entrenamiento = ajuste del modelo final (la búsqueda de hiperparámetros va aparte);
inferencia medida sobre los 600 CVs de test en la RTX 5060.

| Modelo | MAE (años) | RMSE (años) | Acc ±1 | Acc ±2 | Acc ±3 | Acc. bandas | F1 macro bandas | R² | Entrenamiento | Inferencia por CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Mediana | 3.543 | 4.571 | 18.3 % | 34.5 % | 49.8 % | 33.0 % | 0.124 | −0.038 | 0.01 s | 0.00 ms |
| TF-IDF + Ridge | 0.820 | 1.094 | 67.2 % | 95.5 % | 98.7 % | 84.3 % | 0.849 | 0.941 | 2.4 s | 0.61 ms |
| Embeddings + Ridge | 1.063 | 1.400 | 55.7 % | 87.3 % | 96.5 % | 83.5 % | 0.839 | 0.903 | 9.6 s | 1.37 ms |
| **BERT afinado** | **0.112** | **0.151** | **100 %** | **100 %** | **100 %** | **97.7 %** | **0.976** | **0.999** | **13.0 min** | **4.15 ms** |
| *Reglas (bonus)* | *0.021* | *0.026* | *100 %* | *100 %* | *100 %* | *97.8 %* | *0.978* | *1.000* | *—* | *0.01 ms* |

Bandas: 0–2, 3–5, 6–10, 11–20 años (sobre los años redondeados). Incluso BERT y la regla
fallan alguna banda: un CV con 2.48 años que se predice 2.52 cambia de banda con un error
de dos semanas.

**La ganancia de BERT es real** (bootstrap pareado, 2,000 remuestreos del test;
[`bert_gain_bootstrap.csv`](output/tables/bert_gain_bootstrap.csv),
[figura](output/figures/bert_gain.png)):

| BERT frente a | Reducción de MAE (años) | IC 95 % | ¿Significativa? |
|---|---|---|---|
| Mediana | 3.43 | [3.21, 3.66] | sí |
| TF-IDF + Ridge | **0.71** | [0.65, 0.76] | sí |
| Embeddings + Ridge | **0.95** | [0.88, 1.02] | sí |
| Reglas (bonus) | **−0.09** | [−0.10, −0.08] | sí: **la regla es mejor** |

**Predicho frente a real** ([figura](output/figures/pred_vs_true.png)). BERT y la regla
quedan sobre la diagonal. TF-IDF y los embeddings **tiran hacia la media**: subestiman a
los seniors (error medio −0.74 y −0.83 años en la banda 11–20) y sobreestiman la banda
3–5 (+0.24 y +0.07). BERT casi no lo hace (−0.06 en 11–20;
[`bias_by_band.csv`](output/tables/bias_by_band.csv)).

**Por banda y por ocupación** ([`mae_by_band.csv`](output/tables/mae_by_band.csv),
[`mae_by_occupation.csv`](output/tables/mae_by_occupation.csv)). El error de TF-IDF y de
los embeddings crece en los seniors (11–20: 1.16 y 1.57 años) y el de BERT casi no cambia
entre bandas (0.10–0.14). Entre ocupaciones las diferencias son pequeñas para todos los
modelos. Las matrices de confusión por bandas están en `output/figures/confusion_*.png`.

**Los 10 errores más grandes de BERT**
([`bert_top_errors.csv`](output/tables/bert_top_errors.csv)), clasificados con las
categorías del enunciado (se comprueban en orden: OCR → texto → modelo):

| Causa | CVs | Lectura |
|---|---|---|
| **Hueco entre empleos** (≥ 1 año) | 6 | la experiencia no es "fin − inicio"; BERT resta mal los huecos |
| Fechas bien leídas, BERT falla la cuenta | 3 | dos CVs con un solo empleo de menos de 1 año, y uno de 16 años |
| Experiencia en el tope de 20 años | 1 | 23 años reales recortados a 20; BERT dijo 19.3 |
| OCR leyó mal una fecha / orden de lectura / truncado / fechas faltantes / solapes | 0 | el OCR y el orden de lectura no causaron ningún error grande |

El mayor error de BERT es de **0.70 años**. Ninguno de los 10 viene del OCR.

**Qué usa cada modelo para adivinar.** TF-IDF: los tokens que más suben la predicción son
años antiguos (`2007`, `2008`, el n-grama `-200` de un rango que termina en 200x) y los que
más la bajan son años recientes (`2020`, `2021`, `2022`): aprendió **cuándo empezó la
carrera**, no a restar ([`tfidf_top_features.csv`](output/tables/tfidf_top_features.csv)).
Sin dígitos su MAE pasa de 0.82 a 1.71: la mitad de lo que sabe viene de los números, y
el resto de las palabras (cuántos empleos, verbos de liderazgo frente a "Assisted with").
Los embeddings captan sobre todo el "tipo" de CV; BERT, la estructura (ver la sonda).

**Mejoras posibles.** (1) Dar a BERT las duraciones ya calculadas (o pedirle que
prediga cada rango y sumar), en lugar de pedirle la aritmética. (2) Entrenar con
aumentación de fechas (desplazar y estirar rangos) para que aprenda a leer los valores y
no la estructura. (3) Un modelo híbrido: la regla cuando encuentra rangos completos y BERT
cuando el formato no es reconocible.

## 8. Las cuatro preguntas

### Pregunta 1 — ¿Qué aprende BERT que los baselines no pueden?

**Cuánto y dónde.** BERT reduce el MAE en 0.71 años frente a TF-IDF y 0.95 frente a los
embeddings, con intervalos que no tocan el cero. La ganancia viene de **todas las bandas**,
pero sobre todo de los **seniors** (11–20): ahí TF-IDF falla 1.16 años y BERT 0.14, porque
los baselines tiran hacia la media y BERT no.

**La sonda** ([`probe_results.csv`](output/tables/probe_results.csv),
[figura](output/figures/probe.png)). A 20 CVs de test con al menos dos empleos se les
aplicaron tres cambios con respuesta conocida. "Reclutador" es la suma de fechas, que en
el texto original acierta con ±0.02 años:

| Cambio | Reclutador | BERT | TF-IDF | Embeddings |
|---|---|---|---|---|
| Borrar el empleo más antiguo | −2.83 | **−2.83** | −1.79 | −2.36 |
| El empleo más antiguo empieza 3 años antes | +3.00 | **+0.31** | +0.36 | +0.12 |
| Todas las fechas +5 años | 0.00 | **−2.09** | −1.31 | −0.07 |

**Lectura.** BERT reacciona perfecto cuando cambia la **estructura** (desaparece un empleo
entero), pero casi nada cuando cambia el **valor** de una fecha (+0.31 en vez de +3). Y si
todas las fechas pasan a años que nunca vio (2028), su predicción baja 2 años cuando no
debería moverse. **BERT no aprendió a restar fechas**: aprendió patrones que, en
PopResume, van casi siempre juntos con la experiencia (número de empleos, en qué años
empiezan, qué verbos usan los puestos). Por eso le gana a TF-IDF dentro de la
distribución, y por eso falla fuera de ella.

### Pregunta 2 — ¿Cuánto cuesta el ruido del OCR?

**Calidad del OCR (v6):** CER 0.005 %, WER 0.04 %, 100 % de años bien leídos y **cero**
rangos de fechas mal leídos (los 34 rotos vienen cortados en la imagen). Donde falla un
OCR de esta familia lo mostró el modelo rápido: se come el guion largo entre fechas.

**Cada modelo entrenado, evaluado sobre tres versiones del texto de los mismos 600 CVs**
([`ocr_noise_cost.csv`](output/tables/ocr_noise_cost.csv),
[figura](output/figures/ocr_noise_cost.png)):

| Modelo | Texto de referencia | OCR v6 (el de entrenamiento) | OCR rápido (v5 mobile) |
|---|---|---|---|
| Mediana | 3.543 | 3.543 | 3.543 |
| TF-IDF + Ridge | 0.820 | 0.820 | 0.896 (+9 %) |
| Embeddings + Ridge | 1.064 | 1.063 | 1.152 (+8 %) |
| **BERT** | 0.113 | 0.112 | **0.632 (×5.6)** |
| Reglas | 0.020 | 0.021 | 0.021 |

- **Con el OCR elegido, el ruido no cuesta nada**: la diferencia con el texto de
  referencia es ≤ 0.001 años en todos los modelos.
- **Con un OCR peor, BERT es el modelo menos robusto**: su error se multiplica por 5.6,
  mientras TF-IDF y los embeddings empeoran menos de un 10 %. El OCR rápido pega las
  fechas (`2015.052017.12`): para TF-IDF y los embeddings es un token raro más entre
  muchos; para BERT cambia la tokenización de la única evidencia precisa que usa. El mismo
  efecto aparece con un solo carácter: el BERT entrenado con la variante `raw` (guiones
  largos `–`, tal como los devuelve el OCR) tiene un MAE de test de 0.118, que sube a
  **0.564** si recibe los mismos CVs con guiones cortos (`-`). **La regla es la más
  robusta** porque su regex acepta rangos sin guion.
- Consecuencia práctica: el OCR y la limpieza de producción tienen que ser **idénticos**
  a los de entrenamiento, y el benchmark del OCR tiene que medir los rangos de fechas, no
  solo el CER.

### Pregunta 3 — Sensibilidad metodológica

Se compararon **tres** de las opciones del enunciado
([`sensitivity.csv`](output/tables/sensitivity.csv),
[figura](output/figures/sensitivity.png)). En la principal, **cada modelo se reentrenó con
cada variante** del texto y se evaluó sobre la misma variante.

**Limpieza del OCR** (MAE de test, años):

| Modelo | OCR sin limpiar (`raw`) | OCR limpio (`clean`, principal) | Limpio y sin renglones de confianza < 0.98 (`clean_conf`) |
|---|---|---|---|
| Mediana | 3.543 | 3.543 | 3.543 |
| TF-IDF + Ridge | 0.823 | **0.820** | 0.884 (+8 %) |
| Embeddings + Ridge | 1.072 | **1.063** | 1.148 (+8 %) |
| BERT | 0.118 | **0.112** | 0.183 (+63 %) |
| Reglas | 0.021 | **0.021** | 0.250 (×12) |

- **Limpiar ayuda poco pero en todos los modelos** (BERT −5 %): unificar los guiones le
  da a cada rango de fechas siempre la misma tokenización.
- **Quitar los renglones de confianza baja daña a todos.** Quita el 2 % de los
  renglones, pero entre ellos 358 puestos con sus fechas, **todos bien leídos**. La regla
  pierde empleos enteros (×12) y BERT pierde un 63 % de precisión. Con un OCR bueno, la
  confianza no indica errores: filtrar por ella solo borra evidencia.

**Números en TF-IDF:** sin dígitos, su MAE pasa de **0.820 a 1.706** (×2.1). La mitad de
lo que sabe viene de los años; el resto, de las palabras.

**Pérdida de BERT:** MSE **0.112** frente a Huber (δ = 1 año) **0.121**. Huber está
pensada para que los errores grandes no dominen el ajuste, pero aquí casi todos los
errores son menores de 1 año, y en esa zona Huber es exactamente la mitad del MSE: mismo
óptimo, la mitad del gradiente. Con la misma tasa de aprendizaje converge un poco menos.
Se mantiene MSE.

**Estrategia para textos largos:** no se compara porque no hay nada que comparar: el CV
más largo tiene 221 tokens y ninguno se trunca (cabeza, cabeza + cola y trozos dan el
mismo texto).

**Resumen.** La elección que más pesa no es del modelo sino del texto: limpiar ayuda un
poco, y filtrar por confianza daña mucho. En el modelo, la elección que más pesa es
conservar los números en TF-IDF (×2.1 de error si se quitan).

### Pregunta 4 — Valor de negocio, equidad y ética

**Quién lo usaría y para qué.** Agencias de selección, bolsas de trabajo y servicios
públicos de empleo que reciben miles de CVs: para **prellenar** el campo "años de
experiencia" del perfil, ordenar una bandeja o detectar inconsistencias (el candidato
declara 10 años y el CV suma 4). Prellenar un perfil que una persona revisa es muy
distinto de decidir quién pasa a entrevista: lo primero es razonable con este error; lo
segundo, no.

**Qué cuesta un error.** ±2 años es la diferencia entre "junior" y "semi-senior" en muchos
avisos; ±5 años cambia la categoría salarial. Es **peor subestimar**: un filtro del tipo
"≥ 5 años" descarta en silencio a quien sí cumple, y esa persona nunca se entera ni puede
corregirlo; sobreestimar se corrige en la entrevista. Aquí BERT acierta dentro de ±1 año
en el 100 % del test, pero eso vale **para CVs con este formato**; en CVs reales el error
sería mayor (ver limitaciones).

**Prueba de equidad** ([`fairness.csv`](output/tables/fairness.csv),
[figura](output/figures/fairness.png)). Sexo y raza **no** son entradas del modelo; se
usan solo aquí. Error medio de BERT (predicción − real), con IC 95 % bootstrap:

| Grupo | n | MAE | Error medio (años) | IC 95 % | CVs con hueco ≥ 1 año |
|---|---|---|---|---|---|
| Mujeres | 296 | 0.114 | −0.026 | [−0.043, −0.008] | 32 % |
| Hombres | 304 | 0.110 | −0.017 | [−0.033, −0.001] | 36 % |
| Asian/Pacific Islander | 111 | 0.136 | −0.050 | [−0.081, −0.018] | 42 % |
| Black | 51 | 0.097 | −0.019 | [−0.053, +0.016] | 22 % |
| White | 438 | 0.108 | −0.014 | [−0.028, −0.001] | 33 % |

BERT **subestima levemente a todos los grupos**, entre 5 y 18 días; las diferencias entre
grupos son de días y los intervalos se solapan. El grupo más subestimado
(Asian/Pacific Islander) es también el que tiene **más huecos entre empleos** (42 %), y
BERT resta peor los huecos (error medio −0.029 con hueco frente a −0.017 sin hueco). Es
el mecanismo típico de un sesgo sin variables protegidas: PopResume genera las
trayectorias a partir de estadísticas reales que difieren entre grupos, y un modelo que
maneja peor un tipo de trayectoria traslada ese error al grupo donde es más frecuente.
Aquí el efecto es de días, pero el mecanismo es el que hay que vigilar con CVs reales.

**Límites y por qué nunca debe decidir a quién se contrata.**
- **CVs sintéticos y limpios**: un formato, una forma de escribir fechas, imágenes
  digitales. Con escaneos, fotos, "Jan 2019 – Present", dos columnas o empleos sin fechas,
  el OCR, la regla y BERT empeorarían; ninguno de estos números se traslada a CVs reales.
- **Solo 5 ocupaciones, estadísticas de EE. UU., solo inglés.**
- **La etiqueta es la suma de los empleos listados**: no ve trabajo informal, a tiempo
  parcial ni empleos omitidos, que pesan distinto según el grupo social.
- La experiencia es un mal sustituto de la idoneidad, y un error silencioso no se puede
  apelar. El sistema puede **sugerir** un número que una persona confirma; nunca debe ser
  un filtro automático.

## Decisiones técnicas

- **PaddlePaddle y PyTorch no pueden compartir proceso en Windows.** Cada uno trae su
  copia de las DLL de CUDA (12.9 y 12.8), y la primera que se carga gana. PaddleOCR,
  además, importa `modelscope`, que importa torch si lo encuentra. Por eso
  `ocr_extraction.py` hace que `importlib.util.find_spec("torch")` devuelva `None` dentro
  de su proceso, y todo lo demás corre en procesos aparte: `run_all.py` lanza cada etapa
  como subproceso y la app llama al OCR como subproceso.
- **Pesos en fp32.** Transformers 5 carga cada checkpoint en el tipo con que se guardó;
  `deberta-v3-small` está en fp16, y AdamW sobre pesos fp16 da NaN desde el primer paso.
  Se cargan siempre en fp32 y el cálculo va en bf16.
- **GPU serie 50.** Necesitan CUDA ≥ 12.8: PyTorch `cu128` y PaddlePaddle `cu129`.

---

## Instalación (Windows, Python 3.12)

```powershell
# 1. Entorno virtual (fuera de OneDrive, para no sincronizar miles de archivos)
py -3.12 -m venv C:\Users\<usuario>\.venvs\cv_experience_ocr_bert
C:\Users\<usuario>\.venvs\cv_experience_ocr_bert\Scripts\activate
python -m pip install --upgrade pip

# 2. Dependencias (PyTorch y PaddlePaddle vienen de sus propios índices; están en el archivo)
pip install -r requirements.txt

# 3. Acceso al dataset: acepta las condiciones en
#    https://huggingface.co/datasets/sumin-yu/PopResume  y luego:
hf auth login          # (antes "huggingface-cli login"); pega un token de tipo Read
```

**GPU de la serie 50 (RTX 50xx).** Necesitan CUDA ≥ 12.8. Por eso `requirements.txt`
instala `torch==2.11.0+cu128` y `paddlepaddle-gpu==3.4.0` desde el índice `cu129`: las
versiones por defecto de PyPI no reconocen estas GPU.

**Sin GPU.** En `requirements.txt`, cambia `cu128` por `cpu` y `paddlepaddle-gpu` por
`paddlepaddle==3.4.0`. Todo funciona, pero el OCR (~0.3 s/CV en GPU) y BERT tardan
bastante más.

## Cómo reproducir todo

```powershell
python run_all.py
```

Un solo comando, de la descarga a las figuras. Comprueba el entorno al empezar y es
**reanudable**: lo ya descargado, el OCR ya hecho y los BERT ya entrenados no se repiten.

| Etapa | Script | Qué produce | Tiempo (RTX 5060) |
|---|---|---|---|
| 1 | `src/download_data.py` | 4,000 imágenes + metadatos + texto de referencia | ~3 min |
| 2 | `src/ocr_extraction.py` | un JSON por CV en `data/ocr/` (y `data/ocr_mobile/`) | ~23 + 11 min |
| 3 | `src/ocr_quality.py` | CER, WER, años y rangos de fechas bien leídos | segundos |
| 4 | `src/preprocessing.py` | `data/processed/texts.parquet` (5 variantes de texto) | segundos |
| 5 | `src/split.py` | `data/splits/{train,val,test}.csv` | segundos |
| 6 | `src/baseline_*.py` | mediana, TF-IDF, embeddings, reglas | ~3 min |
| 7 | `src/train_bert.py` | selección entre 3 backbones (el ganador queda como modelo principal) + 3 entrenamientos más (`raw`, `clean_conf`, Huber) | ~1 h |
| 8 | `src/evaluation.py` | tablas de `output/tables/`, bootstrap, preguntas | ~1 min |
| 9 | `src/visualization.py` | figuras de `output/figures/` | segundos |

Cada script también se puede correr suelto (`python src/<script>.py --help`).

**Lo que no está en GitHub y cómo regenerarlo.** Las imágenes (`data/raw/`), el OCR
(`data/ocr/`), los textos procesados (`data/processed/`) y los pesos de los modelos
(`models/`) están en `.gitignore`. `python run_all.py` los regenera. Lo que sí está
versionado: la lista de CVs muestreados (`data/sampled_ids.csv`), la partición
(`data/splits/`), todas las tablas, métricas, predicciones y figuras de `output/`.

## La app

```powershell
streamlit run app.py
```

Cuatro pestañas, que corresponden a las cuatro del enunciado:

| Pestaña | Enunciado | Qué tiene |
|---|---|---|
| 1 · Problema y metodología | Problem & Methodology | objetivo, datos, regla de enmascarado, OCR y orden de lectura, modelos, partición, limitaciones |
| 2 · OCR y análisis exploratorio | OCR & Exploratory Analysis | distribución de la experiencia (y el tope), calidad del OCR, un CV con sus cajas junto a su texto |
| 3 · Resultados de los modelos | Model Results | tabla comparativa, predicho frente a real por modelo, MAE por banda y ocupación, matrices de confusión, ganancia de BERT con IC y las cuatro preguntas |
| 4 · Estimador de experiencia | Interactive Experience Estimator | sube la imagen o el PDF de un CV (corre PaddleOCR y los cinco modelos), explora los CVs de test y ve un acierto y un error explicados |

Para abrirla hay que activar antes el entorno virtual
(`C:\Users\<usuario>\.venvs\cv_experience_ocr_bert\Scripts\Activate.ps1`).

## Estructura

```
cv_experience_ocr_bert/
├── app.py                     app de Streamlit (4 pestañas)
├── run_all.py                 todo el pipeline en una corrida
├── requirements.txt           versiones exactas (PaddlePaddle, PaddleOCR, PyTorch...)
├── src/
│   ├── download_data.py       muestreo estratificado y descarga desde PopResume
│   ├── ocr_extraction.py      PaddleOCR por imagen, caché, confianza, benchmark
│   ├── ocr_quality.py         CER / WER / años / rangos frente al texto de referencia
│   ├── preprocessing.py       orden de lectura, limpieza, datos personales, enmascarado
│   ├── split.py               partición estratificada 70/15/15
│   ├── baseline_median.py     mediana
│   ├── baseline_tfidf.py      TF-IDF + Ridge
│   ├── baseline_embeddings.py embeddings congelados + Ridge
│   ├── baseline_rules.py      bonus: suma de duraciones con regex
│   ├── train_bert.py          BERT afinado como regresor
│   ├── evaluation.py          métricas, bootstrap, errores, las 4 preguntas
│   ├── visualization.py       figuras
│   └── utils.py               semillas, rutas, constantes, logging
├── data/
│   ├── sampled_ids.csv        los 4,000 CVs de la muestra (versionado)
│   ├── splits/                train.csv, val.csv, test.csv (versionado)
│   ├── raw/  ocr/  ocr_mobile/  processed/      (no versionados; los crea run_all)
├── models/baselines/  models/bert/              (pesos no versionados)
├── output/
│   ├── figures/  tables/  metrics/  predictions/
└── video/link.txt
```

## Flujo de trabajo en GitHub

Salvo el commit inicial (`.gitignore` y un README vacío), nada se escribió directamente
en `main`. Cada etapa vive en su rama y entró por Pull Request: `feature/data-sampling`, `feature/paddleocr-extraction`,
`feature/ocr-quality`, `feature/baselines`, `feature/bert-regression`,
`feature/evaluation`, `feature/streamlit-dashboard`, `docs/readme`.
