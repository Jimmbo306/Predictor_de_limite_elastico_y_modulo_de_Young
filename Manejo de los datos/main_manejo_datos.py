"""
Módulo orquestador para la Etapa 1: Manejo y Exploración de Datos.

Permite ejecutar de manera secuencial:
1. Carga de MPEA_dataset.csv y extracción de encabezados (analisis_headers.py).
2. Verificación del conteo de filas no nulas simultáneas en subconjuntos de headers.
3. Búsqueda combinatoria y optimización de información (exploracion_combinaciones.py).
4. Generación del reporte resumen en formato CSV.
5. Filtrado y exportación de dataset curado para Modulo_de_young (generar_dataset_filtrado.py).
"""

import sys
from pathlib import Path

DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
if str(DIRECTORIO_ACTUAL) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_ACTUAL))

from analisis_headers import (
    cargar_dataset,
    contar_filas_no_nulas,
    extraer_headers,
    obtener_resumen_no_nulos,
)
from exploracion_combinaciones import (
    encontrar_optimos_globales,
    explorar_con_headers_obligatorios,
    generar_reporte_completo,
)
from generar_dataset_filtrado import generar_dataset_filtrado


def ejecutar_etapa_exploracion():
    print("=" * 75)
    print("ETAPA 1: MANEJO Y EXPLORACIÓN COMBINATORIA DE DATOS")
    print("=" * 75)

    # 1. Carga y extracción básica
    df = cargar_dataset()
    headers = extraer_headers(df)
    print(f"\n[1] Dataset cargado: {len(df)} filas y {len(headers)} headers.")

    # 2. Demostración de la función contar_filas_no_nulas
    headers_demo = ["FORMULA", "YS (MPa)", r"Test temperature ($^\circ$C)"]
    filas_no_nulas = contar_filas_no_nulas(headers_demo, df)
    print(f"\n[2] Conteo de no nulos simultáneos para {headers_demo}:")
    print(f"    -> Filas válidas completas: {filas_no_nulas} de {len(df)}")

    # 3. Exploración de combinaciones óptimas
    print("\n[3] Ejecutando optimización combinatoria...")
    best_k, optimo_global = encontrar_optimos_globales(df)
    print(f"    -> Óptimo global de información (k * filas): {optimo_global['info_total']} datos")
    print(f"    -> Tamaño (k): {optimo_global['k']} columnas")
    print(f"    -> Filas no nulas simultáneas: {optimo_global['filas']}")
    print(f"    -> Columnas: {optimo_global['headers']}")

    # 4. Generación del reporte CSV
    generar_reporte_completo(guardar_csv=True)

    # 5. Generación del dataset filtrado para Modulo_de_young
    print("\n[4] Generando dataset curado para el Módulo de Young...")
    generar_dataset_filtrado()

    print("\n" + "=" * 75)
    print("Ejecución de la Etapa 1 finalizada con éxito.")
    print("=" * 75)


if __name__ == "__main__":
    ejecutar_etapa_exploracion()
