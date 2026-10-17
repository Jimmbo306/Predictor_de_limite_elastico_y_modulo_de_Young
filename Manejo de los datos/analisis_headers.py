"""
Módulo para el análisis y extracción de headers de MPEA_dataset.csv.

Este script contiene las funciones para:
1. Cargar el dataset con pandas.
2. Extraer la lista de encabezados (headers).
3. Evaluar la cantidad de filas que tienen datos no nulos simultáneamente
   en un conjunto específico de encabezados.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Set, Union
import pandas as pd


# Ruta base por defecto al archivo MPEA_dataset.csv dentro de 'Manejo de los datos'
CARPETA_MANEJO_DATOS = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = CARPETA_MANEJO_DATOS / "MPEA_dataset.csv"

# Variable para almacenar en caché el DataFrame cargado por defecto
_DATASET_CACHE: Optional[pd.DataFrame] = None


def cargar_dataset(ruta_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Carga el archivo CSV utilizando pandas.
    
    Args:
        ruta_csv: Ruta opcional al archivo CSV. Si no se especifica,
                  se usa la ruta por defecto (MPEA_dataset.csv).
                  
    Returns:
        pd.DataFrame: DataFrame cargado con la información del CSV.
        
    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta dada.
    """
    global _DATASET_CACHE
    path = Path(ruta_csv) if ruta_csv else DEFAULT_DATASET_PATH

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo del dataset en: {path}")

    # Si es la ruta por defecto y ya está en caché, retornamos copia para evitar recargas
    if ruta_csv is None and _DATASET_CACHE is not None:
        return _DATASET_CACHE

    df = pd.read_csv(path)
    if ruta_csv is None:
        _DATASET_CACHE = df
    return df


def extraer_headers(df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Extrae la lista de nombres de columnas (headers) del dataset.
    
    Args:
        df: DataFrame del cual extraer los headers. Si es None, carga
            el dataset por defecto.
            
    Returns:
        List[str]: Lista ordenada de headers presentes en el dataset.
    """
    if df is None:
        df = cargar_dataset()
    return list(df.columns)


def contar_filas_no_nulas(
    headers: Union[Iterable[str], str],
    df: Optional[pd.DataFrame] = None
) -> int:
    """
    Calcula la cantidad de filas que tienen información no nula (not NaN)
    simultáneamente en todos los headers indicados.
    
    Args:
        headers: Conjunto, lista, tupla o iterable de nombres de columnas.
        df: DataFrame a evaluar. Si no se proporciona, se utiliza el dataset cargado.
        
    Returns:
        int: Número de registros (filas) completos en el subconjunto de columnas.
        
    Raises:
        ValueError: Si algún header indicado no existe en las columnas del dataset.
    """
    if df is None:
        df = cargar_dataset()

    # Si se pasa un string por error, convertir a lista de un solo elemento
    if isinstance(headers, str):
        headers_lista = [headers]
    else:
        headers_lista = list(headers)

    # Caso base: si la lista está vacía, no hay restricciones
    if not headers_lista:
        return 0

    # Validación de existencia de columnas
    columnas_dataset = set(df.columns)
    columnas_faltantes = [h for h in headers_lista if h not in columnas_dataset]
    if columnas_faltantes:
        raise ValueError(
            f"Los siguientes headers no existen en el dataset: {columnas_faltantes}"
        )

    # Contar filas donde todas las columnas especificadas son no nulas
    # notna().all(axis=1) genera una máscara booleana por fila
    filas_validas = df[headers_lista].notna().all(axis=1).sum()
    return int(filas_validas)


def obtener_resumen_no_nulos(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Genera una tabla resumen con el conteo y porcentaje de valores no nulos
    para cada columna individual del dataset.
    
    Args:
        df: DataFrame a evaluar. Si es None, usa el dataset por defecto.
        
    Returns:
        pd.DataFrame: Resumen ordenado de mayor a menor cantidad de datos válidos.
    """
    if df is None:
        df = cargar_dataset()

    total_filas = len(df)
    conteos = df.notna().sum()
    porcentajes = (conteos / total_filas) * 100

    resumen = pd.DataFrame({
        "Header": conteos.index,
        "Valores_No_Nulos": conteos.values,
        "Valores_Nulos": total_filas - conteos.values,
        "Porcentaje_Completitud (%)": porcentajes.round(2).values
    })

    return resumen.sort_values(by="Valores_No_Nulos", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 70)
    print("ANÁLISIS DE HEADERS Y COMPLETITUD DE DATOS")
    print("=" * 70)

    # 1. Cargar datos
    df = cargar_dataset()
    print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.\n")

    # 2. Extraer headers
    headers = extraer_headers(df)
    print(f"Total de headers extraídos ({len(headers)}):")
    for i, h in enumerate(headers, 1):
        print(f"  {i:2d}. {h}")
    print()

    # 3. Mostrar resumen individual
    print("Resumen de valores no nulos por columna (Top 10 más completos):")
    resumen = obtener_resumen_no_nulos(df)
    print(resumen.head(10).to_string(index=False))
    print()

    # 4. Pruebas de la función contar_filas_no_nulas
    print("-" * 70)
    print("PRUEBAS DE LA FUNCIÓN contar_filas_no_nulas:")
    print("-" * 70)

    ejemplo_1 = ["ID", "FORMULA"]
    print(f"Headers: {ejemplo_1}")
    print(f"Filas no nulas simultáneas: {contar_filas_no_nulas(ejemplo_1, df)}\n")

    ejemplo_2 = ["FORMULA", "YS (MPa)", "UTS (MPa)", "Elongation (%)"]
    print(f"Headers: {ejemplo_2}")
    print(f"Filas no nulas simultáneas: {contar_filas_no_nulas(ejemplo_2, df)}\n")

    ejemplo_3 = headers  # Todos los headers juntos
    print("Todos los 23 headers juntos:")
    print(f"Filas con información completa en todos los campos: {contar_filas_no_nulas(ejemplo_3, df)}")
    print("=" * 70)
