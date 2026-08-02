"""
Entrenamiento, ajuste de hiperparametros y registro en MLflow.
Ejecutar desde la raiz del repositorio:
python fuentes/train.py
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

try:
    import mlflow
    import mlflow.sklearn
except ImportError as exc:
    raise SystemExit("MLflow no esta instalado. En Colab ejecuta primero: !pip install -q mlflow") from exc

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "datos" / "datos_limp" / "breast_cancer_wisconsin_limpio.csv"
FIG_DIR = BASE_DIR / "figuras"
REPORT_DIR = BASE_DIR / "reportes"
FIG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Configuracion de MLflow local
mlflow.set_tracking_uri(f"file:{BASE_DIR / 'mlruns'}")
mlflow.set_experiment("Actividad5_BreastCancer_Clasificacion")

data = load_breast_cancer()
df = pd.read_csv(DATA_PATH)
X = df[list(data.feature_names)]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
modelos = {
    "Regresion_Logistica": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
    ),
    "Random_Forest": (
        Pipeline([("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10], "clf__min_samples_split": [2, 5]},
    ),
}

resultados = []

for nombre_modelo, (pipeline, grid) in modelos.items():
    with mlflow.start_run(run_name=nombre_modelo):
        busqueda = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        busqueda.fit(X_train, y_train)
        pred = busqueda.predict(X_test)
        proba = busqueda.predict_proba(X_test)[:, 1]

        metricas = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "cv_f1_mean": busqueda.best_score_,
            "cv_f1_std": busqueda.cv_results_["std_test_score"][busqueda.best_index_],
        }

        mlflow.log_params(busqueda.best_params_)
        mlflow.log_metric("train_rows", X_train.shape[0])
        mlflow.log_metric("test_rows", X_test.shape[0])
        for k, v in metricas.items():
            mlflow.log_metric(k, float(v))

        # Artefactos
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de confusion - {nombre_modelo}")
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["malignant", "benign"]); ax.set_yticklabels(["malignant", "benign"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        cm_path = FIG_DIR / f"matriz_confusion_{nombre_modelo}.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        report_path = REPORT_DIR / f"classification_report_{nombre_modelo}.txt"
        report_path.write_text(classification_report(y_test, pred, target_names=["malignant", "benign"]), encoding="utf-8")

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(busqueda.best_estimator_, artifact_path=f"modelo_{nombre_modelo}")

        fila = {"modelo": nombre_modelo, **metricas, "mejores_parametros": json.dumps(busqueda.best_params_)}
        resultados.append(fila)

metrics_df = pd.DataFrame(resultados)
metrics_df.to_csv(REPORT_DIR / "metricas_modelos_mlflow.csv", index=False)
mlflow.log_artifact(str(REPORT_DIR / "metricas_modelos_mlflow.csv"))

# Grafica comparativa
ax = metrics_df.set_index("modelo")[["accuracy", "precision", "recall", "f1", "roc_auc"]].T.plot(kind="bar", figsize=(8, 4.5))
ax.set_title("Comparacion de desempeno por modelo")
ax.set_xlabel("Metrica")
ax.set_ylabel("Valor")
ax.set_ylim(0.85, 1.01)
plt.tight_layout()
fig_path = FIG_DIR / "comparacion_metricas_mlflow.png"
plt.savefig(fig_path, dpi=150)
plt.close()

print("Entrenamiento finalizado. Para abrir MLflow ejecuta:")
print("mlflow ui --backend-store-uri ./mlruns")
print("Despues abre la URL local que indique la terminal o Colab.")
