# Evidencias visuales de MLflow

Despues de ejecutar `python fuentes/train.py`, abre MLflow con:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Captura y guarda en esta carpeta:

1. Pantalla principal del experimento `Actividad5_BreastCancer_Clasificacion`.
2. Runs de `Regresion_Logistica` y `Random_Forest`.
3. Parametros registrados para cada modelo.
4. Metricas: accuracy, precision, recall, f1 y roc_auc.
5. Artefactos: matriz de confusion, reporte de clasificacion y modelo.
