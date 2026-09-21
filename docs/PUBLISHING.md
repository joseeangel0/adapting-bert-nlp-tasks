# Publicar los cuatro modelos en el HuggingFace Hub (Parte 5)

Todo el material ya está generado. Falta una cuenta y un token; el resto es un comando.

## 1. Crear la cuenta (2 minutos, gratis)

1. Entra a <https://huggingface.co/join> y crea la cuenta.
2. Confirma el correo.
3. Anota tu **usuario** (aparece en `huggingface.co/<usuario>`). Ese es el `--user` del comando.

## 2. Crear un token de escritura

1. <https://huggingface.co/settings/tokens> → **Create new token**.
2. Tipo: **Write**. Nombre: `u2t01`. Créalo y cópialo (empieza con `hf_…`; solo se muestra una vez).

## 3. Iniciar sesión en esta máquina

```bash
.venv/bin/hf auth login          # pega el token cuando lo pida; responde "n" a git credentials
.venv/bin/hf auth whoami         # debe imprimir tu usuario
```

El token queda en `~/.cache/huggingface/token`, fuera del repositorio. **Nunca lo pegues en un
archivo del proyecto ni en un commit.**

## 4. Revisar antes de subir

```bash
# Escribe las cuatro model cards en docs/model_cards/ sin tocar el Hub
.venv/bin/python scripts/push_to_hub.py --user <tu-usuario> --dry-run
```

Revisa `docs/model_cards/*.md`: datos de entrenamiento y licencia, método de adaptación con sus
hiperparámetros, métricas en el split de prueba, la tabla completa de métodos que probamos para esa
tarea, uso previsto, limitaciones y referencias.

## 5. Subir

```bash
.venv/bin/python scripts/push_to_hub.py --user <tu-usuario>              # públicos
.venv/bin/python scripts/push_to_hub.py --user <tu-usuario> --private    # privados
```

Se crean cuatro repositorios, cada uno con el modelo, su tokenizer y su model card:

| Tarea | Repositorio |
|---|---|
| Clasificación de tópicos | `<usuario>/bert-base-uncased-agnews-topic` |
| NER | `<usuario>/bert-base-uncased-conll2003-ner` |
| POS | `<usuario>/bert-base-uncased-ud-ewt-pos` |
| QA extractivo | `<usuario>/bert-base-uncased-squad-qa` |

## 6. Si los dejas privados

El enunciado pide dar acceso a **`Dexterg83`**. En cada repositorio:
**Settings → Collaborators → Add collaborator → `Dexterg83`** (rol *read* basta para evaluar).

## 7. Actualizar el reporte con las URLs

```bash
# pon tu usuario en configs/report.json -> "hf_user"
.venv/bin/python scripts/build_report.py
```

La sección 9 del reporte pasa de mostrar los nombres de los repositorios a mostrar los enlaces reales.

## Comprobar que quedó bien

```python
from transformers import pipeline

pipeline("text-classification", model="<usuario>/bert-base-uncased-agnews-topic")(
    "Wall Street closed lower after the Federal Reserve raised rates")

pipeline("token-classification", model="<usuario>/bert-base-uncased-conll2003-ner",
         aggregation_strategy="simple")("Angela Merkel visited Microsoft in Berlin")

pipeline("question-answering", model="<usuario>/bert-base-uncased-squad-qa")(
    question="Where did she go?", context="Angela Merkel visited Microsoft in Berlin in 2019.")
```

Si un modelo devuelve predicciones sin sentido, el checkpoint se subió mal; `scripts/verify_models.py`
recarga cada checkpoint local y recalcula su métrica, y `docs/model_checks.md` guarda el resultado.
