"""
Pipeline final unificado de Machine Learning para Ciencia de Materiales: Módulo de Young.
Archivo: Modulo_de_young/main_modulo_de_young.py

Este script automatiza el flujo completo de preprocesamiento para modelado en Machine Learning:
1. Extracción y generación de 'dataset.csv' con las 13 columnas seleccionadas de MPEA_dataset.csv:
   - ID
   - FORMULA
   - BCC/FCC/other
   - Calculated Density (g/cm$^3$)
   - REFERENCE: doi
   - REFERENCE: year
   - REFERENCE: title
   - Processing method
   - Microstructure
   - Test temperature ($^\\circ$C)
   - Type of test
   - YS (MPa)
   - Calculated Young modulus (GPa)
2. Limpieza de seguridad con .dropna() para garantizar cero valores nulos (550 filas retenidas).
3. Featurización química avanzada con matminer:
   - Conversión de 'FORMULA' a objeto Composition (StrToComposition).
   - Extracción de descriptores físicos mediante MultipleFeaturizer:
     * ElementProperty.from_preset("magpie")
     * YangSolidSolution()
     * ValenceOrbital(props=['frac'])
     * ElementFraction()
   - Eliminación del objeto intermedio 'composition'.
4. Procesamiento matemático (One-Hot Encoding con dtype=int):
   - Codificación binaria estricta (0 y 1) de variables categóricas:
     * 'BCC/FCC/other'
     * 'Processing method'
     * 'Microstructure'
     * 'Type of test'
   - Eliminación de columnas de metadatos/referencias no deseadas en ML:
     * 'ID'
     * 'REFERENCE: doi'
     * 'REFERENCE: year'
     * 'REFERENCE: title'
     * 'FORMULA'
   - Garantía de una matriz 100% numérica (sin tipos de datos object ni boolean).
5. Exportación de la matriz final lista para modelado: 'dataset_final_ml.csv'.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional
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
DIRECTORIO_RAIZ = CARPETA_ACTUAL.parent
RUTA_DATASET_ENTRADA = CARPETA_ACTUAL / "dataset.csv"
RUTA_DATASET_SALIDA = CARPETA_ACTUAL / "dataset_final_ml.csv"

# Rutas candidatas del dataset crudo MPEA
RUTAS_CRUDO_CANDIDATAS = [
    DIRECTORIO_RAIZ / "MPEA_dataset - MPEA_dataset.csv",
    DIRECTORIO_RAIZ / "Manejo de los datos" / "MPEA_dataset.csv",
]

# Las 13 columnas exactas especificadas
COLUMNAS_DATASET_BASE: List[str] = [
    "ID",
    "FORMULA",
    "BCC/FCC/other",
    "Calculated Density (g/cm$^3$)",
    "REFERENCE: doi",
    "REFERENCE: year",
    "REFERENCE: title",
    "Processing method",
    "Microstructure",
    r"Test temperature ($^\circ$C)",
    "Type of test",
    "YS (MPa)",
    "Calculated Young modulus (GPa)",
]

# Columnas categóricas designadas para One-Hot Encoding
COLUMNAS_CATEGORICAS: List[str] = [
    "BCC/FCC/other",
    "Processing method",
    "Microstructure",
    "Type of test",
]

# Columnas de metadatos o texto a excluir de la matriz final de ML
COLUMNAS_METADATOS_EXCLUIR: List[str] = [
    "ID",
    "REFERENCE: doi",
    "REFERENCE: year",
    "REFERENCE: title",
    "FORMULA",
]


def encontrar_dataset_crudo() -> Path:
    """Localiza el archivo base MPEA_dataset.csv en las rutas conocidas."""
    for ruta in RUTAS_CRUDO_CANDIDATAS:
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        f"No se encontró el dataset MPEA original en ninguna de las rutas: {RUTAS_CRUDO_CANDIDATAS}"
    )


def generar_dataset_base(
    ruta_crudo: Optional[Path] = None,
    ruta_salida: Optional[Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extrae las 13 columnas requeridas desde el dataset original de MPEA,
    aplica dropna() y exporta el archivo dataset.csv.
    """
    path_raw = Path(ruta_crudo) if ruta_crudo else encontrar_dataset_crudo()
    path_out = Path(ruta_salida) if ruta_salida else RUTA_DATASET_ENTRADA

    if verbose:
        print(f"Extrayendo 13 columnas base desde: {path_raw.name}")

    df_raw = pd.read_csv(path_raw)
    total_filas_raw = len(df_raw)

    # Validar existencia de las columnas
    faltantes = [c for c in COLUMNAS_DATASET_BASE if c not in df_raw.columns]
    if faltantes:
        raise KeyError(f"Columnas faltantes en el dataset crudo: {faltantes}")

    df_filtrado = df_raw[COLUMNAS_DATASET_BASE].dropna().reset_index(drop=True)
    filas_retenidas = len(df_filtrado)

    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_filtrado.to_csv(path_out, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"   -> Filas originales: {total_filas_raw}")
        print(f"   -> Filas retenidas (dropna): {filas_retenidas} ({filas_retenidas / total_filas_raw * 100:.2f}%)")
        print(f"   -> Archivo exportado: {path_out}")

    return df_filtrado


def ejecutar_pipeline_ml(
    ruta_entrada: Optional[Path] = None,
    ruta_salida: Optional[Path] = None,
    eliminar_metadatos: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Ejecuta el pipeline unificado de carga, featurización con matminer y
    codificación matemática (One-Hot Encoding) para el Módulo de Young.

    Args:
        ruta_entrada: Ruta al archivo dataset.csv base.
        ruta_salida: Ruta de destino para dataset_final_ml.csv.
        eliminar_metadatos: Si es True, elimina ID, referencias y Formula para asegurar
                            una matriz estrictamente física/termodinámica y numérica.
        verbose: Si es True, imprime detalles de cada etapa del pipeline.

    Returns:
        pd.DataFrame: Matriz procesada lista para entrenamiento en Machine Learning.
    """
    path_in = Path(ruta_entrada) if ruta_entrada else RUTA_DATASET_ENTRADA
    path_out = Path(ruta_salida) if ruta_salida else RUTA_DATASET_SALIDA

    # Si dataset.csv no existe, generarlo automáticamente a partir del dataset original
    if not path_in.is_file():
        if verbose:
            print(f"No se encontró {path_in.name}. Generándolo automáticamente...")
        generar_dataset_base(ruta_salida=path_in, verbose=verbose)

    if verbose:
        print("=" * 80)
        print("PIPELINE UNIFICADO DE MACHINE LEARNING: MÓDULO DE YOUNG & CIENCIA DE MATERIALES")
        print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Carga de Datos Base y Limpieza de Nulos
    # -------------------------------------------------------------------------
    if verbose:
        print(f"1. Cargando datos base desde: {path_in.name}")

    df = pd.read_csv(path_in)
    filas_iniciales = len(df)

    # dropna de seguridad para garantizar consistencia
    df = df.dropna().reset_index(drop=True)
    filas_limpias = len(df)

    if verbose:
        print(f"   -> Filas cargadas: {filas_iniciales}")
        print(f"   -> Filas tras dropna(): {filas_limpias} (0 valores nulos)")
        print(f"   -> Columnas iniciales ({len(df.columns)}): {list(df.columns)}")

    # -------------------------------------------------------------------------
    # 2. Procesamiento Químico con Matminer
    # -------------------------------------------------------------------------
    if verbose:
        print("\n2. Procesamiento Químico con matminer:")

    columna_formula = "FORMULA" if "FORMULA" in df.columns else "Formula"
    if columna_formula not in df.columns:
        raise KeyError(f"No se encontró columna de fórmula química ('FORMULA' o 'Formula').")

    # 2.1 Conversión a objeto composition
    if verbose:
        print(f"   -> 2.1 Convirtiendo columna '{columna_formula}' en objeto Composition (StrToComposition)...")

    stc = StrToComposition(target_col_id="composition")
    df = stc.featurize_dataframe(df, col_id=columna_formula, ignore_errors=False)

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

    cats_presentes = [c for c in COLUMNAS_CATEGORICAS if c in df.columns]
    if verbose:
        print(f"   -> Aplicando pd.get_dummies(..., dtype=int) a: {cats_presentes}")

    df = pd.get_dummies(df, columns=cats_presentes, dtype=int)

    # 3.1 Eliminación de metadatos y referencias documentales
    if eliminar_metadatos:
        cols_descartar = [c for c in COLUMNAS_METADATOS_EXCLUIR if c in df.columns]
        if verbose and cols_descartar:
            print(f"   -> Descartando columnas de metadatos/documentación no numéricas para ML: {cols_descartar}")
        df = df.drop(columns=cols_descartar)

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
        print(f"   -> Estado: Listo para modelado y experimentación de Machine Learning.")
        print("=" * 80)

    return df


if __name__ == "__main__":
    # Generar dataset base si se llama directamente
    generar_dataset_base()
    # Ejecutar featurización y procesamiento final
    ejecutar_pipeline_ml()
