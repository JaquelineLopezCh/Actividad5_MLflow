# Actividad 5 - Design Thinking, definiendo e ideando

## Proyecto
Entrenamiento, ajuste y registro de experimentos para un problema de clasificacion binaria usando el dataset publico **Breast Cancer Wisconsin Diagnostic** disponible en `sklearn.datasets.load_breast_cancer`.

## Fuente del dataset
- Fuente: scikit-learn, `load_breast_cancer`.
- Dataset original: Breast Cancer Wisconsin Diagnostic de UCI.
- Observaciones: 569.
- Variables predictoras: 30 variables numericas derivadas de imagenes de diagnostico.
- Variable objetivo: `target`, donde 0 = malignant y 1 = benign.
- URL de referencia: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Estructura del repositorio
```text
Actividad5/
|-- datos/
|   |-- datos_ini/
|   |   |-- breast_cancer_wisconsin_original.csv
|   |-- datos_limp/
|       |-- breast_cancer_wisconsin_limpio.csv
|   |-- diccionario_datos.csv
|-- fuentes/
|   |-- entrena.ipynb
|   |-- datos_prep.py
|   |-- train.py
|-- figuras/
|-- reportes/
|-- evidencias/
|-- README.md
|-- CHANGELOG.md
|-- requirements.txt
```

## Preparacion de datos
El proceso de limpieza se implementa en `fuentes/datos_prep.py`:
1. Carga del dataset desde scikit-learn.
2. Guardado de datos originales en `datos/datos_ini`.
3. Eliminacion de duplicados.
4. Validacion de tipos numericos.
5. Revision y eliminacion de nulos en campos criticos.
6. Generacion de datos limpios en `datos/datos_limp`.

## Modelos entrenados
Se comparan dos algoritmos:
- Regresion Logistica con escalado estandar.
- Random Forest con balanceo de clases.

## Validacion y busqueda de hiperparametros
Se utiliza `GridSearchCV` con validacion cruzada estratificada de 5 folds y metrica principal `f1`.

## Metricas evaluadas
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Media y desviacion estandar del F1 en validacion cruzada

## Resultados base obtenidos
| modelo              |   accuracy |   precision |   recall |       f1 |   roc_auc |   cv_f1_mean |   cv_f1_std | mejores_parametros                                                              | matriz_confusion   |
|:--------------------|-----------:|------------:|---------:|---------:|----------:|-------------:|------------:|:--------------------------------------------------------------------------------|:-------------------|
| Regresion Logistica |   0.973684 |    0.972603 | 0.986111 | 0.97931  |  0.995701 |     0.986116 |  0.00423737 | {"clf__C": 0.1, "clf__penalty": "l2", "clf__solver": "lbfgs"}                   | [[40, 2], [1, 71]] |
| Random Forest       |   0.947368 |    0.958333 | 0.958333 | 0.958333 |  0.994213 |     0.973728 |  0.0147001  | {"clf__max_depth": null, "clf__min_samples_split": 2, "clf__n_estimators": 100} | [[39, 3], [3, 69]] |

## Uso de MLflow
El script `fuentes/train.py` registra:
- Parametros del mejor modelo.
- Metricas de entrenamiento y prueba.
- Artefactos: matrices de confusion, reportes de clasificacion, comparacion de metricas y modelo serializado.

### Comandos de reproduccion
```bash
pip install -r requirements.txt
python fuentes/datos_prep.py
python fuentes/train.py
mlflow ui --backend-store-uri ./mlruns
```

## Evidencias sugeridas
Guardar en la carpeta `evidencias/`:
1. Captura del experimento en MLflow.
2. Captura de parametros registrados.
3. Captura de metricas comparativas.
4. Captura de artefactos generados.

## Decision tecnica
Se selecciona Random Forest como candidato final si mantiene mejor F1 y ROC-AUC, debido a su capacidad para capturar relaciones no lineales. Regresion Logistica se conserva como baseline interpretable y reproducible.


## Declaratoria de originalidad
El reporte tecnico se redacto de forma original para esta actividad. Las fuentes externas se usan solo como apoyo bibliografico y se citan en el documento Word.

## Fuentes bibliograficas
- scikit-learn developers. load_breast_cancer documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- UCI Machine Learning Repository. Breast Cancer Wisconsin (Diagnostic). https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- scikit-learn developers. GridSearchCV documentation. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- MLflow. MLflow Scikit-learn Integration. https://mlflow.org/docs/latest/ml/traditional-ml/sklearn
