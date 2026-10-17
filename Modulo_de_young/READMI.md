# Módulo de Young: Pipeline Unificado y Modelado Predictivo

Este directorio contiene los scripts, conjuntos de datos y cuadernos de experimentación correspondientes al pipeline del **Módulo de Young Calculado (`Calculated Young modulus (GPa)`)** en Aleaciones de Múltiples Elementos Principales (MPEAs), replicando la metodología de ingeniería de características aplicada en la etapa de Límite Elástico.

---

## 1. Bitácora de Procesamiento y Curación

### 1.1 Extracción del Dataset Base (`dataset.csv`)
A partir del conjunto de datos original `MPEA_dataset - MPEA_dataset.csv` (1,545 registros x 23 columnas), se extrajeron exactamente las **13 columnas óptimas** correspondientes al nivel $k=13$ identificado en la etapa de optimización combinatoria:

1. `ID` (Identificador de referencia documental)
2. `FORMULA` (Fórmula química estequiométrica)
3. `BCC/FCC/other` (Estructura cristalográfica principal)
4. `Calculated Density (g/cm$^3$)` (Densidad teórica calculada)
5. `REFERENCE: doi` (Identificador DOI de la publicación)
6. `REFERENCE: year` (Año de publicación del estudio)
7. `REFERENCE: title` (Título del artículo científico)
8. `Processing method` (Método de procesamiento metalúrgico)
9. `Microstructure` (Fases y microestructura observada)
10. `Test temperature ($^\circ$C)` (Temperatura de ensayo mecánico)
11. `Type of test` (Tipo de ensayo: Tracción 'T' o Compresión 'C')
12. `YS (MPa)` (Límite elástico / Yield Strength)
13. `Calculated Young modulus (GPa)` (Módulo de Young calculado)

#### Criterio de Limpieza (Dropna)
Al exigir simultaneidad de datos válidos (sin nulos) en estas 13 variables:
- **Filas originales**: 1,545
- **Filas retenidas**: **550 registros** (35.60% del dataset original)
- **Valores nulos residuales**: 0

---

## 2. Featurización Automatizada con Matminer

El script `main_modulo_de_young.py` procesa la estequiometría química (`FORMULA`) mediante la biblioteca `matminer`:

1. **Conversión a Objeto Composicional**:
   - `StrToComposition`: Transforma las fórmulas químicas en objetos `Composition` de `pymatgen`.

2. **Extracción Multivariable con `MultipleFeaturizer`**:
   Se aplican exactamente cuatro clases de descriptores físicos y termodinámicos:
   - **`ElementProperty.from_preset("magpie")`**: Estadísticas elementales ponderadas (media, desviación estándar, mínimo, máximo, rango y moda) de propiedades atómicas fundamentales (electronegatividad, radio atómico, volumen molar, etc.).
   - **`YangSolidSolution()`**: Parámetros de solución sólida para aleaciones de alta entropía (desajuste atómico $\delta$, parámetro termodinámico $\Omega$, entalpía $\Delta H_{mix}$ y entropía $\Delta S_{mix}$ de mezcla).
   - **`ValenceOrbital(props=['frac'])`**: Fracción de electrones de valencia distribuidos en capas de orbitales s, p, d y f.
   - **`ElementFraction()`**: Vector de proporciones molares para cada elemento presente en el espacio composicional.

3. **Depuración**:
   - Eliminación del objeto intermedio `composition`.

---

## 3. Procesamiento Matemático y Matriz de Machine Learning (`dataset_final_ml.csv`)

1. **One-Hot Encoding**:
   - Codificación binaria estricta (`dtype=int`, valores 0 y 1) de las 4 variables categóricas:
     - `BCC/FCC/other`
     - `Processing method`
     - `Microstructure`
     - `Type of test`

2. **Exclusión de Metadatos y Documentación**:
   - Siguiendo los principios de modelado físico, se descartan los campos de metadatos que no aportan valor termodinámico ni estructural:
     - `ID`
     - `REFERENCE: doi`
     - `REFERENCE: year`
     - `REFERENCE: title`
     - `FORMULA` (tras extraer sus descriptores químicos)

3. **Garantía Numérica y Dimensiones Finales**:
   - **Dimensiones**: **550 filas x 271 columnas**.
   - **Tipos de datos**: 110 columnas enteras binarias (`int32`/`int64`), 161 columnas continuas de punto flotante (`float64`).
   - 0 variables de texto (`object`), 0 valores booleanos (`bool`), 0 valores nulos (`NaN`).

---

## 4. Estructura de la Carpeta

```text
Modulo_de_young/
├── dataset.csv                   # Dataset base con las 13 columnas originales (550 filas x 13 columnas)
├── dataset_final_ml.csv          # Matriz final 100% numérica para ML (550 filas x 271 columnas)
├── main_modulo_de_young.py       # Pipeline unificado de extracción, featurización y One-Hot Encoding
├── model.ipynb                   # Cuaderno de modelado predictivo (Random Forest & XGBoost)
└── READMI.md                     # Documentación técnica y bitácora
```

---

## 5. Modos de Ejecución

Para reproducir el pipeline completo desde la raíz del proyecto:

```bash
python Modulo_de_young/main_modulo_de_young.py
```
