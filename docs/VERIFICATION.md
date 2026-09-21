# Verificación — U2T01 contra el enunciado

Cada punto de `U2T01.pdf` tiene una implementación y, donde se puede automatizar, una comprobación que se vuelve a
correr con un comando. Las comprobaciones de datos y de checkpoints se generan solas:

| Suite | Comando | Resultado |
|---|---|---|
| Alineación sub-palabra / etiqueta | `python scripts/sanity_check.py` | ✅ → [`sanity_check.txt`](sanity_check.txt) |
| Integridad de datos | `python scripts/verify_data.py` | ✅ 13/13 → [`data_checks.md`](data_checks.md) |
| Los checkpoints reproducen su métrica | `python scripts/verify_models.py --all` | → [`model_checks.md`](model_checks.md) |
| Parrilla de experimentos | `python scripts/run_experiments.py` | → `results/<tarea>/<run>.json` |

## Matriz requisito → implementación → evidencia

### Parte 1 — Entorno de la tarea

| # | Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|---|
| 1.1 | Adaptar BERT a las cuatro tareas, una cabeza distinta sobre el mismo cuerpo | `src/train.py::build_model` — `SequenceClassification`, `TokenClassification` (×2) y `QuestionAnswering` sobre `bert-base-uncased` | `results/{agnews,ner,pos,qa}/` | ✅ |
| 1.2 | Clasificación de tópicos: AG News, 4 clases | `fancyzhx/ag_news`; 20 000 de entrenamiento y 5 000 de dev por muestreo con semilla, test oficial de 7 600 intacto | `data_checks.md`: test balanceado 1 900 por clase, fuga entre splits = 0 | ✅ |
| 1.3 | NER: CoNLL-2003, una etiqueta por token | `lhoestq/conll2003`, 14 041 / 3 250 / 3 453 | `data_checks.md`: 0 desajustes en 20 744 oraciones | ✅ |
| 1.4 | POS: UD English-EWT, 17 etiquetas | `universal-dependencies/universal_dependencies`, config `en_ewt`, 12 544 / 2 001 / 2 077 | `data_checks.md`: exactamente 17 UPOS; tokens multipalabra ya expandidos | ✅ |
| 1.5 | QA extractivo: SQuAD v1.1, submuestra de ~15 k | `rajpurkar/squad`, 15 000 de entrenamiento con semilla; evaluación sobre las 10 570 preguntas de validación (el test de SQuAD no es público) | `results/qa/*.json` | ✅ |
| 1.6 | La etiqueta va en el primer sub-token; `-100` en el resto | `src/encoding.py::align_labels`, usado por NER, POS y —en su forma de span— por QA | `sanity_check.txt`; `data_checks.md`: invariantes en 1 000 oraciones por tarea | ✅ |
| 1.7 | Imprimir un batch con tokens junto a sus etiquetas antes de entrenar | `scripts/sanity_check.py`, ejecutado antes de la parrilla | `sanity_check.txt` | ✅ |
| 1.8 | `bert-base` como cuerpo de los modelos entregados | `configs/experiments.py::BODY = "bert-base-uncased"` | los 20 runs de `GRID` usan ese cuerpo | ✅ |
| 1.9 | *(Opcional)* DistilBERT / BERT-large como benchmark de tamaño | `configs/experiments.py::OPTIONAL`, `run_experiments.py --optional` | `results/*/…distilbert…json` | — no ejecutado (el enunciado lo marca como opcional; el tiempo de GPU se destinó a cubrir los tres escalones en las cuatro tareas). `run_experiments.py --optional` lo corre. |
| 1.10 | HuggingFace Transformers, Tokenizers, Datasets y `Trainer` | `datasets.load_dataset`, `AutoTokenizer` rápido (los `word_ids()` son la base de la alineación), `Trainer` con subclases propias | `requirements.txt` con versiones fijas | ✅ |

### Parte 2 — Experimentos

| # | Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|---|
| 2.1 | Un modelo entregado por tarea | `configs/delivery.json` (por defecto, el mejor medido) | sección 1 del reporte | ✅ |
| 2.2 | El modelo entregado solo cuenta junto a una alternativa entrenada por nosotros, con números | todos los runs de la tarea quedan en `results/` y salen en el reporte **y en la model card** | tablas de la sección 6 del reporte | ✅ |
| 2.3 | Al menos dos métodos de adaptación por tarea | los **tres** escalones en las **cuatro** tareas | `run_experiments.py`; tabla tareas × métodos | ✅ |
| 2.4 | Cada método de adaptación aparece en algún lado | feature-based, partial FT y full FT | ídem | ✅ |
| 2.5 | Feature-based puede comparar varios consumidores de la representación congelada | regresión logística (CLS y mean-pooling), SVM lineal, random forest, y probes lineal y MLP en PyTorch | `src/feature_based.py`, `results/agnews/frozen_*` | ✅ |
| 2.6 | Cuidar los learning rates: cabeza ~1e-3, encoder ~2e-5, **dos grupos de parámetros** | `src/train.py::TwoGroupTrainer.create_optimizer` — cuatro grupos en realidad: cabeza y cuerpo × con/sin weight decay | verificado: la razón cabeza/cuerpo se mantiene durante todo el schedule lineal con 10 % de warmup | ✅ |
| 2.7 | Un run por configuración | sin repeticiones ni búsqueda de hiperparámetros | `results/` tiene un archivo por configuración | ✅ |
| 2.8 | Reconocer el ruido de ±1–3 puntos en el reporte en vez de re-correr | recuadro «On noise» en la sección 8 | reporte | ✅ |
| 2.9 | Generar hipótesis y ajustarlas conforme se avanza | cinco hipótesis registradas antes de la parrilla, con su veredicto medido | sección 3 del reporte | ✅ |

### Parte 3 — Métricas y evaluación

| # | Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|---|
| 3.1 | Métricas adecuadas por tarea, justificadas | accuracy + macro-F1 (AG News), F1 de entidad con seqeval + accuracy de token (NER), accuracy de token + macro-F1 sobre 17 etiquetas (POS), EM + F1 con la normalización oficial (SQuAD) | `src/metrics.py`; tabla de justificación en la sección 5 del reporte | ✅ |
| 3.2 | Métricas de entrenamiento para depurar el avance: loss, scores por época, segundos por paso/época | `Trainer` con `logging_steps=50` y `eval_strategy="epoch"`; `EpochTimer` mide segundos por época y por paso | cada `results/*.json` lleva `train_log`, `epoch_times`, `train_seconds`, `eval_seconds`; figura 3 y tabla de tiempos del reporte | ✅ |
| 3.3 | Los resultados se probarán contra el modelo y contra una réplica del profesor | semillas fijas (Python, NumPy, torch, `seed` y `data_seed` de `TrainingArguments`), submuestras derivadas de shuffles con semilla, dependencias fijadas, y un verificador que recarga cada checkpoint y recalcula su métrica | `model_checks.md`; sección 10 del reporte | ✅ |

### Parte 4 — Reporte

| # | Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|---|
| 4.1 | Documentar el proceso de los experimentos | secciones 2 a 7 | `report/U2T01_report.pdf` | ✅ |
| 4.2 | Análisis de los métodos y justificación de las decisiones técnicas, por modelo y tarea | una subsección por tarea en la sección 6, más la sección 8 | reporte | ✅ |
| 4.3 | Tabla comparando tareas contra métodos | `build_report.py::table_matrix` | sección 6 del reporte | ✅ |
| 4.4 | Referencias de datasets, código y papers | sección 11, y la licencia y cita de cada dataset en su model card | reporte y `docs/model_cards/` | ✅ |

### Parte 5 — Publicación

| # | Requisito | Implementación | Evidencia | Estado |
|---|---|---|---|---|
| 5.1 | Publicar los cuatro modelos finales y sus tokenizers en el Hub | `scripts/push_to_hub.py` sube la carpeta del checkpoint, que incluye el tokenizer | 4 repos públicos bajo `joseeangel`, verificados descargándolos de vuelta del Hub: [agnews](https://huggingface.co/joseeangel/bert-base-uncased-agnews-topic) · [ner](https://huggingface.co/joseeangel/bert-base-uncased-conll2003-ner) · [pos](https://huggingface.co/joseeangel/bert-base-uncased-ud-ewt-pos) · [qa](https://huggingface.co/joseeangel/bert-base-uncased-squad-qa) | ✅ |
| 5.2 | Model card con datos de entrenamiento, métricas, uso previsto, limitaciones y referencias | `push_to_hub.py::card`, generada desde el mismo `results/*.json` que produjo el reporte | `docs/model_cards/` | ✅ |
| 5.3 | Si es privado, dar acceso a `Dexterg83` | no aplica: los cuatro repos son **públicos**, cualquiera entra con el enlace | 4 repos públicos bajo `joseeangel`, verificados descargándolos de vuelta del Hub: [agnews](https://huggingface.co/joseeangel/bert-base-uncased-agnews-topic) · [ner](https://huggingface.co/joseeangel/bert-base-uncased-conll2003-ner) · [pos](https://huggingface.co/joseeangel/bert-base-uncased-ud-ewt-pos) · [qa](https://huggingface.co/joseeangel/bert-base-uncased-squad-qa) | ✅ |

### Entregables

| # | Requisito | Dónde | Estado |
|---|---|---|---|
| E.1 | Código: scripts, notebooks, configs, requirements | `src/`, `scripts/`, `notebooks/U2T01_colab.ipynb`, `configs/`, `requirements.txt` con versiones exactas | ✅ |
| E.2 | Semillas fijas y documentación para replicar el proceso | semilla 42 en todos lados; `README.md` § *Reproduce it* y sección 10 del reporte | ✅ |
| E.3 | Reporte en PDF | `report/U2T01_report.pdf` | ✅ |
| E.4 | Repos en el HuggingFace Hub | 4 repos públicos bajo `joseeangel`, verificados descargándolos de vuelta del Hub: [agnews](https://huggingface.co/joseeangel/bert-base-uncased-agnews-topic) · [ner](https://huggingface.co/joseeangel/bert-base-uncased-conll2003-ner) · [pos](https://huggingface.co/joseeangel/bert-base-uncased-ud-ewt-pos) · [qa](https://huggingface.co/joseeangel/bert-base-uncased-squad-qa) | ✅ |

## Decisiones que se apartan del camino obvio, y por qué

1. **El cuerpo congelado se fija en modo `eval`.** Si no, su dropout sigue activo y el «extractor de features» deja
   de ser determinista: el probe de PyTorch y el estimador de scikit-learn estarían consumiendo representaciones
   distintas de la misma oración y la comparación entre ambos no querría decir nada.
2. **El *pooler* de BERT se mide en las dos formas.** `BertForSequenceClassification` no clasifica el `[CLS]`
   crudo sino `tanh(dense([CLS]))`, una proyección preentrenada para *next-sentence prediction*. Congelarla hace
   que un «probe lineal» lea un vector saturado por la tanh en vez de la representación de la oración. Se reportan
   las dos variantes porque la diferencia es grande y es un resultado sobre la arquitectura, no sobre el método.
3. **El mapa de etiquetas de CoNLL está fijado y verificado al cargar.** `lhoestq/conll2003` entrega `ner_tags`
   como enteros sin nombres; `src/data.py::verify_conll_labels` lo contrasta contra la oración de ejemplo del
   shared task. Un cambio silencioso en ese mapa sería indistinguible de un modelo malo.
4. **La cabeza MLP se reconstruye al recargar.** `from_pretrained` vuelve a crear `classifier` como un `Linear` y
   descarta los pesos de una cabeza no estándar sin quejarse, dejando un modelo con cabeza aleatoria. El tipo de
   cabeza vive en el config y `src/train.py::load_run_model` la rehace antes de restaurar los pesos. Lo encontró
   `scripts/verify_models.py`, que es justamente para lo que sirve.
5. **La evaluación de QA hace una sola pasada.** Las ventanas de validación llevan las posiciones de oro además de
   los offsets, así que `Trainer` llama a `compute_metrics` en la propia pasada de evaluación y de ahí salen EM,
   F1 y la loss; con dos pasadas se pagaba el doble por época sin obtener nada más.
6. **Atención *eager* en MPS.** El kernel fusionado de atención de MPS no soporta dropout; en CUDA el mismo código
   conserva la ruta rápida. `bfloat16` está activo en todos los runs y duplica el throughput en este equipo.
