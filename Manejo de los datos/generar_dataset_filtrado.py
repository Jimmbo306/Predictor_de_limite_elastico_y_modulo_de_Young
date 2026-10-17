"""
Módulo para filtrar y exportar datasets curados a partir de MPEA_dataset.csv.

Este script:
1. Toma como entrada MPEA_dataset.csv.
2. Filtra por los headers de interés técnico seleccionados (excluyendo
   variables documentales/administrativas de acuerdo con el READMI).
3. Renombra las columnas a nombres estandarizados y altamente legibles.
4. Exporta el archivo dataset.csv en la carpeta Modulo_de_young (o la indicada).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

# Rutas del proyecto
DIRECTORIO_MANEJO_DATOS = Path(__file__).resolve().parent
DIRECTORIO_RAIZ = DIRECTORIO_MANEJO_DATOS.parent
DEFAULT_RUTA_ENTRADA = DIRECTORIO_MANEJO_DATOS / "MPEA_dataset.csv"

# Localizar la carpeta de destino para el Módulo de Young
RUTA_CARPETA_YOUNG = DIRECTORIO_RAIZ / "Modulo_de_young"
DEFAULT_RUTA_SALIDA = RUTA_CARPETA_YOUNG / "dataset.csv"

# Headers de interés técnico predeterminados (Nivel 12 sin variables documentales)
HEADERS_INTERES_DEFAULT: List[str] = [
    "FORMULA",
    "Calculated Density (g/cm$^3$)",
    "BCC/FCC/other",
    "Processing method",
    "Microstructure",
    r"Test temperature ($^\circ$C)",
    "Type of test",
    "YS (MPa)",
]

# Diccionario completo de mapeo a nombres limpios y legibles (sin sintaxis LaTeX)
MAPEO_HEADERS_LEGIBLES: Dict[str, str] = {
    "FORMULA": "Formula",
    "Calculated Density (g/cm$^3$)": "Densidad_Calculada_g_cm3",
    "Exp. Density (g/cm$^3$)": "Densidad_Experimental_g_cm3",
    "BCC/FCC/other": "Estructura_Cristalina",
    "Processing method": "Metodo_Procesamiento",
    "Microstructure": "Microestructura",
    r"Test temperature ($^\circ$C)": "Temperatura_Prueba_C",
    "Type of test": "Tipo_Ensayo",
    "YS (MPa)": "Limite_Elastico_YS_MPa",
    "UTS (MPa)": "Resistencia_Tension_UTS_MPa",
    "Elongation (%)": "Elongacion_Porc",
    "Elongation plastic (%)": "Elongacion_Plastica_Porc",
    "Calculated Young modulus (GPa)": "Modulo_Young_Calculado_GPa",
    "Exp. Young modulus (GPa)": "Modulo_Young_Experimental_GPa",
    "HV": "Dureza_Vickers_HV",
    r"grain size ($\mu$m)": "Tamano_Grano_um",
    "O content (wppm)": "Contenido_Oxigeno_wppm",
    "N content (wppm)": "Contenido_Nitrogeno_wppm",
    "C content (wppm)": "Contenido_Carbono_wppm",
    "ID": "ID_Referencia",
    "REFERENCE: doi": "Referencia_DOI",
    "REFERENCE: year": "Referencia_Ano",
    "REFERENCE: title": "Referencia_Titulo",
}


def generar_dataset_filtrado(
    ruta_entrada: Optional[Union[str, Path]] = None,
    ruta_salida: Optional[Union[str, Path]] = None,
    headers_interes: Optional[List[str]] = None,
    incluir_modulo_young: bool = False,
    renombrar_columnas: bool = True,
    mapeo_nombres: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Carga el dataset de aleaciones, extrae el subconjunto de columnas técnicas
    de interés, elimina filas con valores nulos en dichos campos, renombra
    los encabezados a un formato legible y exporta el archivo CSV resultante.

    Args:
        ruta_entrada: Ruta al archivo CSV fuente (por defecto MPEA_dataset.csv).
        ruta_salida: Ruta al archivo CSV destino (por defecto Modulo_de_young/dataset.csv).
        headers_interes: Lista de nombres de columnas originales a incluir.
        incluir_modulo_young: Si es True, añade 'Calculated Young modulus (GPa)' a los headers.
        renombrar_columnas: Si es True, renombra las columnas a formato estandarizado.
        mapeo_nombres: Diccionario opcional para sustitución personalizada de nombres.

    Returns:
        pd.DataFrame: DataFrame filtrado y con encabezados legibles.
    """
    path_in = Path(ruta_entrada) if ruta_entrada else DEFAULT_RUTA_ENTRADA
    path_out = Path(ruta_salida) if ruta_salida else DEFAULT_RUTA_SALIDA

    if not path_in.is_file():
        raise FileNotFoundError(f"No se encontró el dataset de entrada en: {path_in}")

    # 1. Definir lista de columnas a filtrar
    if headers_interes is None:
        columnas_a_usar = list(HEADERS_INTERES_DEFAULT)
    else:
        columnas_a_usar = list(headers_interes)

    if incluir_modulo_young and "Calculated Young modulus (GPa)" not in columnas_a_usar:
        columnas_a_usar.append("Calculated Young modulus (GPa)")

    # 2. Cargar datos
    df = pd.read_csv(path_in)
    total_filas_inicial = len(df)

    # 3. Validar columnas
    columnas_faltantes = [c for c in columnas_a_usar if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Las siguientes columnas no existen en el dataset: {columnas_faltantes}")

    # 4. Seleccionar subconjunto y descartar nulos
    df_filtrado = df[columnas_a_usar].dropna().reset_index(drop=True)
    total_filas_final = len(df_filtrado)

    # 5. Renombrar encabezados para mayor legibilidad
    if renombrar_columnas:
        diccionario_nombres = mapeo_nombres or MAPEO_HEADERS_LEGIBLES
        nuevos_nombres = {
            col: diccionario_nombres.get(col, col)
            for col in columnas_a_usar
        }
        df_filtrado = df_filtrado.rename(columns=nuevos_nombres)

    # 6. Crear directorio de destino si no existe y exportar
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_filtrado.to_csv(path_out, index=False, encoding="utf-8-sig")

    print("=" * 75)
    print("GENERACIÓN DE DATASET FILTRADO")
    print("=" * 75)
    print(f"Archivo de entrada:  {path_in.name} ({total_filas_inicial} filas)")
    print(f"Archivo de salida:   {path_out} ({total_filas_final} filas retenidas)")
    print(f"Retención de datos:  {total_filas_final / total_filas_inicial * 100:.2f}%")
    print(f"Número de columnas:  {df_filtrado.shape[1]}")
    print("\nEncabezados transformados:")
    for orig in columnas_a_usar:
        nuevo = MAPEO_HEADERS_LEGIBLES.get(orig, orig) if renombrar_columnas else orig
        print(f"  * '{orig}' -> '{nuevo}'")
    print("=" * 75)

    return df_filtrado


if __name__ == "__main__":
    # Ejecución por defecto: genera dataset.csv en Modulo_de_young con los 8 headers técnicos
    df_resultado = generar_dataset_filtrado()
    print("\nPrimeras 5 filas del dataset generado:")
    print(df_resultado.head(5).to_string(index=False))
