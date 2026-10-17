"""
Pipeline final unificado de Machine Learning para Ciencia de Materiales.
Archivo: Modulo_de_young/main_modulo_de_young.py

Este script automatiza el flujo completo de preprocesamiento para modelado en Colab:
1. Carga de datos base desde 'dataset.csv' (carpeta Modulo_de_young).
2. Limpieza de seguridad con .dropna() para garantizar cero valores nulos.
3. Featurización química avanzada con matminer:
   - Conversión de 'Formula' a objeto Composition (StrToComposition).
   - Extracción de descriptores físicos mediante MultipleFeaturizer:
     * ElementProperty.from_preset("magpie")
     * YangSolidSolution()
     * ValenceOrbital(props=['frac'])
     * ElementFraction()
   - Eliminación del objeto intermedio 'composition'.
4. Procesamiento matemático (One-Hot Encoding con dtype=int):
   - Codificación binaria estricta (0 y 1) de:
     * 'Estructura_Cristalina'
     * 'Metodo_Procesamiento'
     * 'Microestructura'
     * 'Tipo_Ensayo'
   - Eliminación de la columna de texto 'Formula', garantizando una matriz
     100% numérica (sin tipos de datos object ni boolean).
5. Exportación de la matriz final lista para modelado: 'dataset_final_ml.csv'.
"""

import sys
import time
from pathlib import Path
from typing import Optional
import pandas as pd

# Importaciones de matminer para featurización en ciencia de materiales
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    YangSolidSolution,
    ValenceOrbital,
    ElementFraction,
)

# Definición de rutas
CARPETA_ACTUAL = Path(__file__).resolve().parent
RUTA_DATASET_ENTRADA = CARPETA_ACTUAL / "dataset.csv"
RUTA_DATASET_SALIDA = CARPETA_ACTUAL / "dataset_final_ml.csv"

# Columnas categóricas designadas para One-Hot Encoding
COLUMNAS_CATEGORICAS = [
    "Estructura_Cristalina",
    "Metodo_Procesamiento",
    "Microestructura",
    "Tipo_Ensayo",
]


def ejecutar_pipeline_ml(
    ruta_entrada: Optional[Path] = None,
    ruta_salida: Optional[Path] = None,
    eliminar_columna_texto: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Ejecuta el pipeline unificado de carga, featurización y codificación matemática.

    Args:
        ruta_entrada: Ruta al archivo dataset.csv base.
        ruta_salida: Ruta de destino para dataset_final_ml.csv.
        eliminar_columna_texto: Si es True, elimina la columna 'Formula' para asegurar
                                una matriz estrictamente numérica (int/float).
        verbose: Si es True, imprime detalles de cada etapa del pipeline.

    Returns:
        pd.DataFrame: Matriz procesada lista para entrenamiento en Machine Learning.
    """
    path_in = Path(ruta_entrada) if ruta_entrada else RUTA_DATASET_ENTRADA
    path_out = Path(ruta_salida) if ruta_salida else RUTA_DATASET_SALIDA

    if not path_in.is_file():
        raise FileNotFoundError(f"No se encontró el archivo base de entrada en: {path_in}")

    if verbose:
        print("=" * 80)
        print("PIPELINE UNIFICADO DE MACHINE LEARNING: CIENCIA DE MATERIALES")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Carga de Datos Base y Limpieza de Nulos
    # -------------------------------------------------------------------------
    if verbose:
        print(f"1. Cargando datos base desde: {path_in.name}")

    df = pd.read_csv(path_in)
    filas_iniciales = len(df)

    # dropna rápido para asegurar consistencia
    df = df.dropna().reset_index(drop=True)
    filas_limpias = len(df)

    if verbose:
        print(f"   -> Filas cargadas: {filas_iniciales}")
        print(f"   -> Filas tras dropna(): {filas_limpias} (0 valores nulos residuales)")
        print(f"   -> Columnas iniciales: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # 2. Procesamiento Químico con Matminer
    # -------------------------------------------------------------------------
    if verbose:
        print("\n2. Procesamiento Químico con matminer:")

    if "Formula" not in df.columns:
        raise KeyError("La columna 'Formula' es requerida para el cálculo de composiciones.")

    # 2.1 Conversión a objeto composition
    if verbose:
        print("   -> 2.1 Convirtiendo columna 'Formula' en objeto Composition (StrToComposition)...")

    stc = StrToComposition(target_col_id="composition")
    df = stc.featurize_dataframe(df, col_id="Formula", ignore_errors=False)

    # 2.2 Extracción de propiedades físicas y termodinámicas
    if verbose:
        print("   -> 2.2 Aplicando MultipleFeaturizer con 4 clases exactas:")
        print("        [1] ElementProperty.from_preset('magpie')")
        print("        [2] YangSolidSolution()")
        print("        [3] ValenceOrbital(props=['frac'])")
        print("        [4] ElementFraction()")

    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        YangSolidSolution(),
        ValenceOrbital(props=["frac"]),
        ElementFraction(),
    ])

    t0_feat = time.time()
    df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=False)
    t1_feat = time.time()

    if verbose:
        print(f"   -> Featurización completada en {t1_feat - t0_feat:.2f} segundos.")

    # 2.3 Eliminación de la columna temporal 'composition'
    if "composition" in df.columns:
        df = df.drop(columns=["composition"])
        if verbose:
            print("   -> 2.3 Columna temporal 'composition' eliminada con éxito.")

    # -------------------------------------------------------------------------
    # 3. Procesamiento Matemático (One-Hot Encoding con dtype=int)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n3. Procesamiento Matemático (One-Hot Encoding):")

    # Validar presencia de las columnas categóricas
    cats_presentes = [c for c in COLUMNAS_CATEGORICAS if c in df.columns]
    if verbose:
        print(f"   -> Aplicando pd.get_dummies(..., dtype=int) a: {cats_presentes}")

    df = pd.get_dummies(df, columns=cats_presentes, dtype=int)

    # 3.1 Eliminación de la columna de texto 'Formula' para garantizar matriz numérica
    if eliminar_columna_texto and "Formula" in df.columns:
        df = df.drop(columns=["Formula"])
        if verbose:
            print("   -> Columna de texto 'Formula' descartada tras la extracción de descriptores.")

    # -------------------------------------------------------------------------
    # 4. Verificación de Tipos de Datos (Zero Text & Zero Boolean)
    # -------------------------------------------------------------------------
    tipos_no_numericos = df.select_dtypes(include=["object", "string", "bool"]).columns.tolist()
    if tipos_no_numericos:
        if verbose:
            print(f"   -> [Aviso] Descartando variables residuales no numéricas: {tipos_no_numericos}")
        df = df.drop(columns=tipos_no_numericos)

    if verbose:
        total_enteros = sum(df.dtypes == "int32") + sum(df.dtypes == "int64")
        total_flotantes = sum(df.dtypes == "float64") + sum(df.dtypes == "float32")
        print("\n4. Verificación de integridad de la matriz:")
        print(f"   -> Columnas enteras binarias (0/1): {total_enteros}")
        print(f"   -> Columnas flotantes continuas:    {total_flotantes}")
        print(f"   -> Columnas no numéricas residuales: 0")

    # -------------------------------------------------------------------------
    # 5. Exportación del Output Final
    # -------------------------------------------------------------------------
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_out, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"\n5. Matriz final generada exitosamente:")
        print(f"   -> Archivo exportado: {path_out}")
        print(f"   -> Dimensiones finales: {df.shape[0]} filas x {df.shape[1]} columnas")
        print(f"   -> Estado: Listo para entrenamiento y experimentación en Google Colab.")
        print("=" * 80)

    return df


if __name__ == "__main__":
    ejecutar_pipeline_ml()
