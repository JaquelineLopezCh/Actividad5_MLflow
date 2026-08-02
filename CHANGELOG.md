
## [1.1.0] - 2026-08-01
### Mejorado
- Reporte tecnico reconstruido en formato Word con matriz de cumplimiento de rubrica.
- Agregada declaratoria de originalidad y fuentes bibliograficas.
- Agregado checklist final para entrega y evidencias MLflow.

# CHANGELOG

## [1.0.0] - 2026-08-01
### Agregado
- Estructura inicial del repositorio para Actividad 5.
- Dataset original y dataset limpio.
- Diccionario de datos.
- Script `datos_prep.py` para preparacion y versionamiento de datos.
- Script `train.py` para entrenamiento, Grid Search, validacion cruzada y registro en MLflow.
- Notebook `entrena.ipynb` para ejecucion en Google Colab.
- Visualizaciones de calidad de datos y comparacion de modelos.
- Reporte tecnico en PDF.

### Cambiado
- Estandarizacion del flujo para reproducibilidad.
- Separacion entre datos originales y datos limpios.

### Notas de MLOps
- MLflow se usa como primer componente de trazabilidad experimental.
- La deuda tecnica futura incluye automatizar pruebas, versionar modelos y ejecutar pipeline desde CI/CD.
