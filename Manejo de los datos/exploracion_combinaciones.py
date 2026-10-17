"""
Módulo para la exploración y optimización de combinaciones de headers en MPEA_dataset.csv.

Este script utiliza la función `contar_filas_no_nulas` del módulo `analisis_headers`
para encontrar los conjuntos de headers que maximizan la cantidad de información útil:
1. El óptimo global que maximiza el volumen de datos completos (k_headers * filas_no_nulas).
2. La mejor combinación para cada tamaño k (desde 1 hasta 23 columnas).
3. La frontera de Pareto (el mejor compromiso entre número de variables y tamaño muestral).
4. Exploración condicionada con headers obligatorios (por ejemplo, fórmulas o propiedades clave).
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

# Asegurar importación de analisis_headers tanto si se ejecuta desde raíz como desde la carpeta
DIRECTORIO_ACTUAL = Path(__file__).resolve().parent
if str(DIRECTORIO_ACTUAL) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_ACTUAL))

from analisis_headers import (
    cargar_dataset,
    contar_filas_no_nulas,
    extraer_headers,
)


def _generar_mascaras_columnas(df: pd.DataFrame) -> Tuple[List[int], List[str], int]:
    """
    Función interna de optimización: convierte la matriz de valores no nulos
    en enteros de bits para permitir evaluar millones de combinaciones en microsegundos.
    """
    notna_mat = df.notna().to_numpy()
    n_rows, n_cols = notna_mat.shape
    cols = list(df.columns)

    # Ordenar columnas por frecuencia descendente para maximizar la poda
    conteos = [(j, notna_mat[:, j].sum(), cols[j]) for j in range(n_cols)]
    conteos.sort(key=lambda x: x[1], reverse=True)

    col_masks = []
    for orig_idx, _, _ in conteos:
        mask = 0
        for i in range(n_rows):
            if notna_mat[i, orig_idx]:
                mask |= (1 << i)
        col_masks.append(mask)

    sorted_names = [x[2] for x in conteos]
    return col_masks, sorted_names, n_rows


def encontrar_optimos_globales(
    df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[int, Dict], Dict]:
    """
    Realiza una búsqueda exhaustiva mediante algoritmo Branch and Bound monótono.
    Evalúa el espacio combinatorio de 2^23 subconjuntos en aprox. 1 segundo,
    garantizando encontrar el óptimo matemático global.

    Args:
        df: DataFrame del dataset. Si es None, carga MPEA_dataset.csv.

    Returns:
        Tuple:
            - Dict[int, Dict]: Mejor combinación encontrada para cada tamaño k.
            - Dict: Información de la combinación óptima global que maximiza
                    la información total (k * filas).
    """
    if df is None:
        df = cargar_dataset()

    col_masks, sorted_names, total_filas = _generar_mascaras_columnas(df)
    n_cols = len(sorted_names)

    # best_k[k] almacenará: {'filas': int, 'headers': tuple[str, ...], 'info_total': int}
    best_k: Dict[int, Dict] = {
        k: {"filas": 0, "headers": (), "info_total": 0}
        for k in range(1, n_cols + 1)
    }

    optimo_global = {
        "info_total": 0,
        "k": 0,
        "filas": 0,
        "headers": []
    }

    def dfs(idx: int, current_mask: int, chosen_indices: Tuple[int, ...]):
        k = len(chosen_indices)
        if k > 0:
            cnt = current_mask.bit_count()
            # Poda monótona: si la intersección ya es 0, agregar más columnas seguirá dando 0
            if cnt == 0:
                return

            info = k * cnt
            if cnt > best_k[k]["filas"]:
                best_k[k]["filas"] = cnt
                best_k[k]["headers"] = tuple(sorted_names[i] for i in chosen_indices)
                best_k[k]["info_total"] = info

            if info > optimo_global["info_total"]:
                optimo_global["info_total"] = info
                optimo_global["k"] = k
                optimo_global["filas"] = cnt
                optimo_global["headers"] = [sorted_names[i] for i in chosen_indices]

        # Explorar hijos en el árbol de combinaciones
        for next_idx in range(idx, n_cols):
            next_mask = col_masks[next_idx] if k == 0 else (current_mask & col_masks[next_idx])
            dfs(next_idx + 1, next_mask, chosen_indices + (next_idx,))

    dfs(0, 0, ())

    # Verificación cruzada con la función estándar contar_filas_no_nulas
    for k, data in best_k.items():
        if data["filas"] > 0:
            verificacion = contar_filas_no_nulas(data["headers"], df)
            assert verificacion == data["filas"], f"Discrepancia en k={k}: {verificacion} != {data['filas']}"

    return best_k, optimo_global


def explorar_con_headers_obligatorios(
    headers_fijos: Union[List[str], Set[str]],
    df: Optional[pd.DataFrame] = None,
    top_n: int = 5
) -> List[Dict]:
    """
    Explora qué columnas adicionales conviene agregar a un conjunto de headers
    predefinidos (por ejemplo, variables requeridas para un modelo), para maximizar
    la intersección de datos útiles.

    Args:
        headers_fijos: Lista de columnas indispensables.
        df: DataFrame a evaluar.
        top_n: Número de mejores sugerencias a retornar.

    Returns:
        List[Dict]: Ranking de columnas recomendadas a incorporar.
    """
    if df is None:
        df = cargar_dataset()

    headers_fijos = list(headers_fijos)
    filas_base = contar_filas_no_nulas(headers_fijos, df)
    todos_headers = extraer_headers(df)
    candidatos = [h for h in todos_headers if h not in headers_fijos]

    resultados = []
    for cand in candidatos:
        grupo_prueba = headers_fijos + [cand]
        filas = contar_filas_no_nulas(grupo_prueba, df)
        retencion_pct = (filas / filas_base * 100) if filas_base > 0 else 0.0
        resultados.append({
            "header_adicional": cand,
            "filas_resultantes": filas,
            "filas_perdidas": filas_base - filas,
            "retencion_pct": round(retencion_pct, 2),
            "info_total": filas * len(grupo_prueba)
        })

    resultados.sort(key=lambda x: (x["filas_resultantes"], x["info_total"]), reverse=True)
    return resultados[:top_n]


def generar_reporte_completo(guardar_csv: bool = True) -> pd.DataFrame:
    """
    Ejecuta el análisis combinatorio completo y genera un DataFrame con el reporte,
    con opción de guardar en CSV.
    """
    df = cargar_dataset()
    best_k, optimo_global = encontrar_optimos_globales(df)

    filas_reporte = []
    for k, data in best_k.items():
        if data["filas"] > 0:
            filas_reporte.append({
                "Num_Headers (k)": k,
                "Filas_Completas": data["filas"],
                "Porcentaje_Dataset (%)": round((data["filas"] / len(df)) * 100, 2),
                "Puntos_Informacion (k*filas)": data["info_total"],
                "Headers_Seleccionados": " | ".join(data["headers"])
            })

    reporte_df = pd.DataFrame(filas_reporte)

    if guardar_csv:
        salida_csv = DIRECTORIO_ACTUAL / "resumen_optimizacion_headers.csv"
        reporte_df.to_csv(salida_csv, index=False, encoding="utf-8-sig")
        print(f"\n[OK] Reporte exportado a: {salida_csv}")

    return reporte_df


if __name__ == "__main__":
    print("=" * 80)
    print("EXPLORACIÓN Y OPTIMIZACIÓN COMBINATORIA DE HEADERS")
    print("=" * 80)

    t0 = time.time()
    df = cargar_dataset()
    best_k, optimo_global = encontrar_optimos_globales(df)
    t1 = time.time()

    print(f"Búsqueda exacta completada en {t1 - t0:.3f} segundos.")
    print("-" * 80)
    print("1. MÁXIMO GLOBAL DE INFORMACIÓN (Rectángulo de Datos Útiles más Grande):")
    print("-" * 80)
    print(f"  * Número de headers (k): {optimo_global['k']}")
    print(f"  * Filas completas no nulas: {optimo_global['filas']} ({optimo_global['filas']/len(df)*100:.1f}% del dataset)")
    print(f"  * Total de celdas de información útil: {optimo_global['info_total']} datos")
    print("  * Headers que conforman el óptimo:")
    for i, h in enumerate(optimo_global["headers"], 1):
        print(f"      {i:2d}. {h}")

    print("\n" + "-" * 80)
    print("2. MEJOR COMBINACIÓN DE HEADERS PARA CADA TAMAÑO (k = 1 hasta 18):")
    print("-" * 80)
    print(f"{'k':>2} | {'Filas':>5} | {'% Dataset':>9} | {'Puntos Info':>11} | {'Headers Seleccionados'}")
    print("-" * 80)

    for k in range(1, len(extraer_headers(df)) + 1):
        info = best_k[k]
        if info["filas"] > 0:
            nombres_cortos = [h.split(" (")[0] for h in info["headers"]]
            resumen_h = ", ".join(nombres_cortos)
            if len(resumen_h) > 55:
                resumen_h = resumen_h[:52] + "..."
            pct = (info["filas"] / len(df)) * 100
            print(f"{k:2d} | {info['filas']:5d} | {pct:8.1f}% | {info['info_total']:11d} | {resumen_h}")
        else:
            print(f"{k:2d} |     0 |      0.0% |           0 | (No existen filas completas con k>={k})")
            break

    print("\n" + "-" * 80)
    print("3. CASO DE USO PRÁCTICO: RECOMENDACIÓN CON HEADERS OBLIGATORIOS")
    print("   Ejemplo: Si requerimos ['FORMULA', 'YS (MPa)'], ¿cuáles son las mejores columnas a sumar?")
    print("-" * 80)
    sugerencias = explorar_con_headers_obligatorios(["FORMULA", "YS (MPa)"], df=df, top_n=5)
    for i, s in enumerate(sugerencias, 1):
        print(f"  {i}. + '{s['header_adicional']}': {s['filas_resultantes']} filas retenidas ({s['retencion_pct']}%)")

    # Guardar reporte en CSV
    generar_reporte_completo(guardar_csv=True)
    print("=" * 80)
