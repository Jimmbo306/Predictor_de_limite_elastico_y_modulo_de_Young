# Módulo de Young: Modelado Predictivo y Featurización

Este directorio contiene los scripts y conjuntos de datos correspondientes a la etapa de modelado continuo para la predicción del Módulo de Young Calculado en Aleaciones de Múltiples Elementos Principales (MPEAs).

---

## Bitácora de Procesamiento

### Entrada: 14 de octubre de 2026 - Limpieza de Datos y Featurización Composicional

En esta fecha se llevó a cabo el procesamiento del archivo base `MPEA_dataset - MPEA_dataset.csv` ubicado en la raíz del proyecto, ejecutado a través del script `main_modulo_de_young.py`.

#### 1. Limpieza y Filtrado de Datos
* **Dataset original**: 1,545 registros y 23 columnas.
* **Criterio de exclusión por valores nulos**: Se eliminaron exclusivamente las filas que presentaban valores faltantes (NaN) en las siguientes cuatro variables críticas:
  - `PROPERTY: Calculated Young modulus (GPa)` (o `Calculated Young modulus (GPa)`)
  - `FORMULA`
  - `PROPERTY: Processing method` (o `Processing method`)
  - `PROPERTY: Microstructure` (o `Microstructure`)
* **Registros resultantes**:
  - Filas conservadas: **711 filas** (46.02% del total original).
  - Filas descartadas por ausencia del Módulo de Young o descriptores de procesamiento/microestructura: 834 filas.
* **Depuración de columnas**: Se eliminaron por completo las variables de carácter documental y administrativo que no aportan valor físico o termodinámico al modelo:
  - `IDENTIFIER: Reference ID` (o `ID`)
  - `REFERENCE: doi`
  - `REFERENCE: year`
  - `REFERENCE: title`

#### 2. Featurización Automatizada con Matminer
A partir de la estequiometría química (`FORMULA`), se generó una representación de características físicas y composicionales de alta dimensionalidad utilizando la biblioteca `matminer`:

1. **Conversión de Formato**:
   - `StrToComposition`: Conversión de las fórmulas químicas en objetos de composición de `pymatgen` para su procesamiento por los featurizadores.

2. **Extracción Multivariable (`MultipleFeaturizer`)**:
   Se aplicaron de forma automatizada exactamente cuatro clases de featurizadores composicionales:
   - **`ElementProperty.from_preset("magpie")`**: Genera estadísticas ponderadas (media, media armónica, desviación estándar, valor mínimo, máximo y rango) de propiedades atómicas fundamentales basadas en el conjunto de datos de Magpie (electronegatividad, radio covalente, volumen atómico, energía de ionización, etc.).
   - **`YangSolidSolution()`**: Calcula descriptores termodinámicos y geométricos clásicos propuestos por Yang et al. para aleaciones de alta entropía (diferencia de radios atómicos $\delta$, parámetro termodinámico $\Omega$, entropía y entalpía de mezcla).
   - **`ValenceOrbital(props=['frac'])`**: Determina la fracción de electrones de valencia distribuidos en las capas de orbitales s, p, d y f.
   - **`ElementFraction()`**: Genera el vector disperso con la fracción estequiométrica de cada elemento presente en el espacio de la aleación.

3. **Depuración del Objeto Intermedio**:
   - Se removió la columna temporal `composition` para permitir la serialización limpia en formato CSV.

#### 3. Archivo Resultante
* **Nombre de archivo**: `extended_dataset.csv`
* **Dimensiones finales**: **711 filas y 275 columnas**.
* **Propósito**: Base de entrenamiento tabular completa lista para modelos de regresión supervisada (Random Forest, Gradient Boosting, etc.) orientados a la predicción del Módulo de Young.

---

## Estructura de la Carpeta

```text
Modulo_de_young/
├── dataset.csv                   # Dataset filtrado básico (1,008 filas x 8 columnas)
├── extended_dataset.csv          # Dataset featurizado con matminer (711 filas x 275 columnas)
├── main_modulo_de_young.py       # Script principal de limpieza y featurización
└── README.md                     # Documentación y bitácora técnica de la etapa
```

---

## Ejecución del Pipeline

Para reproducir la limpieza y la generación de `extended_dataset.csv`, ejecute desde la raíz del proyecto:

```bash
python Modulo_de_young/main_modulo_de_young.py
```
