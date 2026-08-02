"""
Funciones de preparacion de datos para Actividad 5.
Dataset: Breast Cancer Wisconsin Diagnostic via sklearn.datasets.load_breast_cancer.
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "datos" / "datos_ini"
CLEAN_DIR = BASE_DIR / "datos" / "datos_limp"


def obtener_dataset():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})
    return df, data


def limpiar_dataset(df, feature_names):
    df_limpio = df.drop_duplicates().copy()
    for col in feature_names:
        df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")
    df_limpio = df_limpio.dropna(subset=list(feature_names) + ["target"])
    df_limpio["target"] = df_limpio["target"].astype(int)
    df_limpio["target_name"] = df_limpio["target"].map({0: "malignant", 1: "benign"})
    return df_limpio


def guardar_datos():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df, data = obtener_dataset()
    df.to_csv(RAW_DIR / "breast_cancer_wisconsin_original.csv", index=False)
    df_limpio = limpiar_dataset(df, data.feature_names)
    df_limpio.to_csv(CLEAN_DIR / "breast_cancer_wisconsin_limpio.csv", index=False)
    return df_limpio


if __name__ == "__main__":
    df_limpio = guardar_datos()
    print(f"Dataset limpio generado con {df_limpio.shape[0]} filas y {df_limpio.shape[1]} columnas.")
