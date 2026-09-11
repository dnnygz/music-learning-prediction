# Music Learning Prediction - Yousician Open Dataset

Proyecto de Machine Learning para predecir si un estudiante de música mejorará su desempeño a partir de sus primeras sesiones de práctica en la plataforma Yousician.

## Problema

El problema que buscamos resolver es **predecir si un estudiante de música logrará mejorar su desempeño durante su proceso de aprendizaje utilizando únicamente la información obtenida durante sus primeras sesiones de práctica**.

Actualmente, las plataformas de aprendizaje musical registran una gran cantidad de interacciones, pero identificar tempranamente qué estudiantes están progresando y cuáles podrían necesitar apoyo adicional resulta difícil. Por ello, el objetivo es construir un modelo de clasificación que permita predecir la variable **`Performance_Improvement`** (mejora futura del estudiante), analizando factores iniciales como frecuencia de práctica, tiempo dedicado, precisión en notas y acordes, dificultad de los ejercicios, nivel de participación y comportamiento durante las sesiones.

De esta manera, una plataforma de aprendizaje musical podría detectar estudiantes con riesgo de estancamiento y recomendar estrategias de aprendizaje más personalizadas para mejorar su progreso musical.

## Fuente de datos

El proyecto utiliza el **Yousician Open Dataset**, disponible en:

[https://yousician.com/open-data](https://yousician.com/open-data)

El dataset contiene registros de interacciones de estudiantes dentro de una plataforma de aprendizaje musical. Cada registro representa una actividad realizada por un usuario durante la práctica de una canción o ejercicio.

## Metodología

El trabajo sigue la metodología **CRISP-DM**:

1. **Business Understanding:** definición del problema predictivo y su utilidad para plataformas de aprendizaje musical.
2. **Data Understanding:** exploración inicial del dataset, variables disponibles, usuarios únicos, actividad por estudiante, evolución temporal, duplicados, valores nulos y métricas de desempeño.
3. **Data Preparation:** transformación del dataset desde nivel interacción hacia nivel estudiante, creación de variables agregadas iniciales y construcción de la variable objetivo.
4. **Modeling:** entrenamiento de modelos de clasificación para predecir `Performance_Improvement`.
5. **Evaluation:** evaluación del desempeño predictivo del modelo.
6. **Deployment:** propuesta de uso del modelo para identificar estudiantes que podrían necesitar apoyo adicional.

## Variable objetivo

La variable objetivo es:

```text
Performance_Improvement
```

Se construye comparando el desempeño inicial del estudiante contra su desempeño posterior:

- **Periodo inicial:** primeros 7 días de actividad.
- **Periodo posterior:** días 15 a 30.
- `Performance_Improvement = 1` si la precisión promedio posterior es mayor que la precisión promedio inicial.
- `Performance_Improvement = 0` si el estudiante no mejora.

Esta separación temporal evita fuga de información, ya que las variables predictoras se crean únicamente con información de las primeras sesiones.

## Variables generadas

El dataset procesado se encuentra a nivel estudiante e incluye:

- `user_id`
- `total_days_active_initial`
- `number_of_sessions_initial`
- `total_time_playing_initial`
- `average_time_per_session_initial`
- `total_exercises_initial`
- `average_note_accuracy_initial`
- `average_chord_accuracy_initial`
- `average_difficulty_initial`
- `completion_rate_initial`
- `abandonment_rate_initial`
- `average_session_index_initial`
- `songs_practiced_initial`
- `exercises_practiced_initial`
- `play_mode_frequency_initial`
- `Performance_Improvement`

Las variables identificadoras `song_id` y `exercise_id` no se utilizan directamente como predictores. Solo se conserva `user_id` como identificador.

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/
│   │   └── yousician_ukulele.json
│   └── processed/
│       └── yousician_processed.csv
├── notebooks/
│   └── 01_data_understanding_yousician.ipynb
├── scripts/
│   └── process_yousician.py
├── requirements.txt
└── README.md
```

## Reproducibilidad

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Generar el dataset procesado:

```bash
python scripts/process_yousician.py
```

El archivo resultante se exporta a:

```text
data/processed/yousician_processed.csv
```

## Estado actual

- Se creó el notebook de **Data Understanding**.
- Se identificó que el dataset original está a nivel interacción, no a nivel estudiante.
- Se detectaron duplicados aparentes y métricas indefinidas cuando no existen notas o acordes evaluados.
- Se creó un script para transformar los datos a nivel estudiante.
- Se generó el dataset procesado `yousician_processed.csv`.

## Limitaciones identificadas

- El dataset original contiene múltiples registros por usuario.
- Algunas filas pueden aparecer duplicadas o representar reintentos dentro de una sesión.
- Algunas métricas de precisión no pueden calcularse cuando el denominador es cero.
- Las variables de identificación de canciones y ejercicios no deben usarse directamente como predictores.
- La variable objetivo debe construirse con ventanas temporales separadas para evitar fuga de información.
