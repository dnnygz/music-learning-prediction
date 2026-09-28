## Problema

Este proyecto investiga si las señales conductuales registradas durante los primeros días de práctica en una plataforma de aprendizaje musical permiten anticipar resultados futuros del estudiante.

La pregunta principal es:

Can behavioral signals recorded during a student's first seven elapsed days predict future observability and, among observable students, the subsequent trajectory of platform-recorded performance?

Debido a que el desempeño observado depende de la dificultad del contenido y del contexto de práctica, primero se construye una medida ajustada de trayectoria mediante modelos jerárquicos probabilísticos antes de evaluar modelos predictivos.

## Metodología

El proyecto se divide en dos etapas:

### 1. Estimación de desempeño y trayectoria

Los eventos originales se transforman en observaciones estudiante-ejercicio-contexto.

Se utiliza un modelo jerárquico beta-binomial:

Y ~ BetaBinomial(N,p)

considerando:

- dificultad del ejercicio;
- efecto del ejercicio;
- diferencias individuales;
- evolución temporal.

Esto permite estimar una trayectoria latente individual:

λ_i

que representa el cambio ajustado del desempeño durante el periodo futuro.

### 2. Evaluación predictiva

Se evalúa si las variables disponibles durante los primeros siete días permiten predecir:

1. Future observability

X_0-7 → FutureObservable


2. Future trajectory

X_0-7 → λ_i

## Experimentos

### Experimento 1 — Future observability

Objetivo:
Determinar si las señales tempranas permiten anticipar si un estudiante tendrá suficiente actividad futura para estimar una trayectoria.

Modelo:

X_0-7 → FutureObservable

Resultado:

- 945 estudiantes inicialmente elegibles: 880 observables y 65 no observables.
- Regresión logística balanceada: ROC-AUC 0.841 e average precision 0.305.
- Random Forest balanceado: ROC-AUC 0.831 e average precision 0.302.
- Las señales tempranas permiten ordenar el riesgo de no observabilidad, pero
  las probabilidades todavía requieren calibración y un umbral operacional.


### Experimento 2 — Sensibilidad temporal

Objetivo:
Evaluar si extender la ventana inicial mejora la capacidad predictiva.

Comparaciones:

- primeros 7 días;
- primeros 14 días;
- primeros 21 días.

La comparación principal usará un resultado común en `[21, 31)` para evitar
solapamiento entre predictores y resultado.


### Experimento 3 — Predicción de desempeño inmediato

Objetivo:
Evaluar un problema alternativo donde la etiqueta es más cercana a la observación.

Modelo:

(student, exercise, context) → P(success)

Este experimento utiliza directamente el modelo beta-binomial de desempeño.

## Resultado de la pregunta principal

La evidencia actual produce una conclusión asimétrica:

> Las señales de los primeros siete días predicen la observabilidad futura mejor
> que un baseline de prevalencia, pero no predicen de forma confiable la
> trayectoria individual posterior de desempeño ajustado.

La definición completa, métricas, limitaciones y próximos experimentos están en
[`docs/problem_definition.md`](docs/problem_definition.md). El reporte específico
de observabilidad está en
[`reports/prediction/observability_v1.md`](reports/prediction/observability_v1.md).

## Estructura del proyecto

```text
src/
├── data/                  # Ingesta, limpieza, tablas analíticas y auditorías
├── analysis/              # Análisis exploratorios y de estabilidad
├── models/
│   ├── measurement/       # Medición jerárquica de desempeño y trayectoria
│   └── prediction/        # Modelos predictivos a nivel estudiante
└── evaluation/            # Reservado para métricas y validaciones compartidas

reports/
├── data_quality/          # Calidad, semántica, conectividad y tabla de modelado
├── measurement/           # Resultados de modelos jerárquicos y de inferencia
├── prediction/            # Resultados de los experimentos predictivos
└── summary/               # Síntesis globales del proyecto

data/
├── raw/                   # Dataset original inmutable
├── interim/               # Artefactos intermedios
├── processed/             # Datasets finales para modelado
└── model_outputs/         # Parámetros, predicciones y diagnósticos generados
```

## Ejecución

Los módulos deben ejecutarse desde la raíz del repositorio. Ejemplos:

```bash
python -m src.data.clean_events
python -m src.data.build_student_tables
python -m src.data.audit_semantics
python -m src.data.audit_connectivity
python -m src.data.build_modeling_table

python -m src.models.measurement.hierarchical_measurement
python -m src.models.measurement.dispersion_comparison
python -m src.models.measurement.inference_validation
python -m src.models.measurement.nonlinear_bridge
python -m src.models.measurement.random_slope_measurement
python -m src.models.measurement.random_slope_inference_validation

python -m src.models.prediction.two_stage_prediction
python -m src.models.prediction.joint_slope_prediction
python -m src.models.prediction.predict_observability
```

Los artefactos numéricos continúan almacenándose en `data/model_outputs/`; la
reorganización solo cambia la ubicación del código y de los reportes Markdown.
