# Proyecto de Ciencia de Materiales: Modelado y Análisis de Datos

Documento de seguimiento del proyecto. Este archivo tiene como objetivo registrar el proceso técnico, las metodologías implementadas, los resultados obtenidos en cada fase y las decisiones adoptadas previo al avance entre etapas.

---

## Estructura del Repositorio

```text
Proyecto_1/
├── MPEA_dataset - MPEA_dataset.csv   # Archivo base de aleaciones en la raíz (1,545 filas x 23 columnas)
├── main.py                           # Orquestador general del pipeline por etapas
├── READMI.md                         # Bitácora general y documentación del proceso
├── Modulo_de_young/                  # Pipeline: Módulo de Young (k=13 headers, 550 filas)
│   ├── dataset.csv                   # Dataset filtrado base (550 filas x 13 columnas exactas)
│   ├── dataset_final_ml.csv          # Matriz final 100% numérica para ML (550 filas x 271 columnas)
│   ├── main_modulo_de_young.py       # Pipeline unificado de featurización con matminer y One-Hot Encoding
│   ├── model.ipynb                   # Cuaderno de modelado predictivo (Random Forest & XGBoost)
│   └── READMI.md                     # Bitácora y documentación técnica de la etapa
├── Limite_elastico/                  # Pipeline: Límite Elástico YS (k=12 headers, 1,008 filas)
│   ├── dataset.csv                   # Dataset filtrado base (1,008 filas x 8 columnas técnicas)
│   ├── dataset_final_ml.csv          # Matriz final 100% numérica para ML (1,008 filas x 306 columnas)
│   ├── main_limite_elastico.py       # Pipeline unificado de featurización y One-Hot Encoding
│   ├── model.ipynb                   # Cuaderno de modelado para YS
│   └── READMI.md                     # Bitácora de la etapa de Límite Elástico
├── Front/                            # Interfaz Web interactiva para inferencia (Flask + HTML/CSS/JS)
│   ├── app.py                        # Servidor backend y API REST de predicción en tiempo real
│   ├── templates/index.html          # Vista web con dos pestañas (Módulo de Young y Límite Elástico)
│   ├── static/                       # Estilos Obsidian Dark (sin emojis) y lógica frontend
│   └── README.md                     # Documentación de la interfaz
├── Models/                           # Modelos entrenados XGBoost y metadatos de variables
│   ├── xgboost_Calculated Young modulus (GPa).pkl # Modelo serializado para Módulo de Young
│   ├── variables_Calculated Young modulus (GPa).json # Descriptores de entrada (27 variables)
│   ├── xgboost_Limite_Elastico_YS_MPa.pkl         # Modelo serializado para Límite Elástico
│   └── variables_Limite_Elastico_YS_MPa.json     # Descriptores de entrada (51 variables)
└── Manejo de los datos/              # Etapa 1: Procesamiento y exploración de datos
    ├── MPEA_dataset.csv              # Base de datos original de MPEAs (1,545 filas x 23 columnas)
    ├── analisis_headers.py           # Carga, extracción de columnas y conteo no nulo
    ├── exploracion_combinaciones.py  # Algoritmo de optimización combinatoria (Branch & Bound)
    ├── generar_dataset_filtrado.py   # Filtrado, depuración y exportación de variables técnicas
    ├── main_manejo_datos.py          # Script orquestador de la Etapa 1
    └── resumen_optimizacion_headers.csv # Reporte tabular con los subconjuntos óptimos
```

---

## Etapa 1: Exploración y Manejo de Datos

El objetivo central de esta primera etapa consiste en inspeccionar la completitud del archivo `MPEA_dataset.csv` y determinar qué combinaciones de variables (headers) maximizan la información simultánea utilizable, evitando descartar datos innecesariamente o lidiar con valores nulos excesivos.

### 1.1 Metodología de Procesamiento

1. **Carga y Extracción de Encabezados (`analisis_headers.py`)**:
   - Acceso al archivo CSV mediante la biblioteca `pandas`.
   - Extracción de la lista completa de las 23 columnas que componen el dataset.
   - Diagnóstico individual de valores no nulos por columna.

2. **Función de Intersección Simultánea (`contar_filas_no_nulas`)**:
   - Se diseñó una función que toma por entrada un subconjunto arbitrario de headers y calcula el número exacto de filas que presentan datos válidos al mismo tiempo en todas esas columnas:
     $$\text{Filas Válidas}(S) = \sum_{i=1}^{N} \prod_{c \in S} \mathbb{I}(\text{valor}_{i,c} \neq \text{NaN})$$
   - Permite evaluar de manera rápida cualquier hipótesis o selección de variables de interés.

3. **Optimización Combinatoria (`exploracion_combinaciones.py`)**:
   - Espacio total de búsqueda: $2^{23} = 8,388,608$ combinaciones posibles.
   - Algoritmo aplicado: Búsqueda exhaustiva mediante poda de árbol (*Branch and Bound* monótono) con representación binaria de bits. Dado que la intersección de no nulos es monótonamente decreciente, cualquier rama con 0 filas se descarta de forma inmediata.
   - Tiempo de ejecución: Evaluación de más de 2.7 millones de subconjuntos válidos en aproximadamente 1 segundo.

---

### 1.2 Resultados de la Exploración de Combinaciones

A continuación se resume el mejor subconjunto de variables para cada número de columnas ($k$):

| k (Headers) | Filas No Nulas | Porcentaje del Dataset | Puntos de Información (k x Filas) | Columnas Seleccionadas |
| :---: | :---: | :---: | :---: | :--- |
| **1 a 6** | 1,545 | 100.0% | Hasta 9,270 | `ID`, `FORMULA`, `BCC/FCC/other`, `Calculated Density (g/cm^3)`, `REFERENCE: doi`, `REFERENCE: year` |
| **7** | 1,535 | 99.4% | 10,745 | Agrega `REFERENCE: title` |
| **8** | 1,416 | 91.7% | 11,328 | Agrega `Processing method` |
| **9** | 1,347 | 87.2% | 12,123 | Agrega `Microstructure` |
| **10** | **1,258** | **81.4%** | **12,580** (Máximo global) | Agrega `Test temperature ($^\circ$C)` |
| **11** | 1,082 | 70.0% | 11,902 | Agrega `Type of test` |
| **12** | 1,001 | 64.8% | 12,012 | Agrega `YS (MPa)` |
| **13** | 550 | 35.6% | 7,150 | Agrega `Calculated Young modulus (GPa)` |
| **14** | 305 | 19.7% | 4,270 | Agrega `Elongation (%)`, `UTS (MPa)` |
| **15** | 172 | 11.1% | 2,580 | Agrega `Calculated Young modulus (GPa)` |
| **16** | 86 | 5.6% | 1,376 | Agrega `grain size ($\mu$m)` |
| **17** | 28 | 1.8% | 476 | Agrega `Exp. Young modulus (GPa)` |
| **18** | 5 | 0.3% | 90 | Agrega `HV` |
| **>= 19** | 0 | 0.0% | 0 | No existen registros completos con 19 o más columnas simultáneas |

---

### 1.3 Curación y Exportación del Dataset Técnico (`generar_dataset_filtrado.py`)

A partir de los resultados combinatorios y las decisiones metodológicas, se desarrolló un módulo de depuración y exportación que:
1. Toma como entrada `MPEA_dataset.csv`.
2. Filtra las 8 variables técnicas de interés seleccionadas (descartando las 4 administrativas/documentales).
3. Elimina filas con valores faltantes en dichas columnas, reteniendo 1,008 registros limpios (65.24% de los datos).
4. Estandariza los nombres de las columnas para eliminar caracteres LaTeX y maximizar la legibilidad en entornos de machine learning:
   - `FORMULA` -> `Formula`
   - `Calculated Density (g/cm$^3$)` -> `Densidad_Calculada_g_cm3`
   - `BCC/FCC/other` -> `Estructura_Cristalina`
   - `Processing method` -> `Metodo_Procesamiento`
   - `Microstructure` -> `Microestructura`
   - `Test temperature ($^\circ$C)` -> `Temperatura_Prueba_C`
   - `Type of test` -> `Tipo_Ensayo`
   - `YS (MPa)` -> `Limite_Elastico_YS_MPa`
5. Genera el archivo limpio `dataset.csv` en el directorio `Modulo_de_young/`.

---

### 1.4 Modos de Ejecución

Desde la terminal en el directorio raíz del proyecto:

* Ejecución de la etapa completa de manejo de datos (incluye análisis y exportación):
  ```bash
  python "Manejo de los datos/main_manejo_datos.py"
  ```
* Ejecución del pipeline general:
  ```bash
  python main.py
  ```
* Ejecución de la prueba de conteo no nulo de headers:
  ```bash
  python "Manejo de los datos/analisis_headers.py"
  ```
* Ejecución de la optimización combinatoria y reporte CSV:
  ```bash
  python "Manejo de los datos/exploracion_combinaciones.py"
  ```
* Ejecución directa del filtrado y exportación para el Módulo de Young:
  ```bash
  python "Manejo de los datos/generar_dataset_filtrado.py"
  ```

---

## Conclusiones

Tras el análisis de la base de datos de Aleaciones de Múltiples Elementos Principales (MPEAs), se identificó que el principal desafío es la alta densidad de valores nulos. Mediante una evaluación en cascada, se determinó que el subconjunto de datos más viable es aquel que logra preservar el 64.79% de los registros originales (1,001 filas).

Este corte representa el punto de equilibrio óptimo, ya que permite conservar un conjunto amplio de variables físicas y mecánicas fundamentales para la ingeniería de materiales —tales como la composición química (FORMULA), la densidad teórica (Calculated Density), la clasificación cristalográfica (BCC/FCC/other), el método de procesamiento (Processing method), la microestructura (Microstructure), la temperatura de prueba (Test temperature), el tipo de ensayo (Type of test) y el límite elástico (YS (MPa))— manteniendo un volumen de datos lo suficientemente robusto y representativo para futuros análisis.

Finalmente, para enfocar el conjunto de datos exclusivamente en características técnicas, se determinó la eliminación de las variables de naturaleza documental o administrativa (como IDENTIFIER: Reference ID, REFERENCE: doi, REFERENCE: year y REFERENCE: title), dado que no aportan información de utilidad termodinámica o estructural.

## Decisiones
Tras la evaluación de la densidad de información y el debate sobre la retención de registros frente a la utilidad analítica, se ha determinado que el proyecto debe transicionar de la fase de exploración a la de modelado computacional aplicado. La base de datos, una vez purgada de variables documentales (referencias, DOI, años) y filtrada estratégicamente, ofrece suficiente robustez para implementar algoritmos de machine learning que extraigan valor termodinámico y mecánico.

La hoja de ruta del proyecto se estructurará abordando los siguientes enfoques metodológicos, priorizando la viabilidad técnica y el impacto en la ingeniería de materiales:

1. Objetivo Principal Inmediato: Predicción del Módulo de Young
El primer paso del proyecto consistirá en desarrollar un modelo predictivo continuo enfocado en el Módulo de Young Calculado (Calculated Young modulus (GPa)).

Justificación: Representa el punto de partida óptimo para estructurar el pipeline de datos en Python. Al contar con un volumen manejable de registros teóricos, permite validar la arquitectura de extracción de características químicas (featurización mediante herramientas como matminer para procesar la columna FORMULA) antes de avanzar a propiedades mecánicas más complejas o escasas.

Metodología: Se priorizarán algoritmos de regresión basados en árboles (como Random Forest o XGBoost) para evitar el sobreajuste y permitir la extracción de la importancia de variables (Feature Importance), asegurando que las predicciones tengan un respaldo físico interpretable.

2. Expansión a Propiedades Mecánicas Críticas (El enfoque del 65%)
Una vez consolidado el pipeline con el Módulo de Young, el modelo se adaptará para predecir el Límite Elástico (YS (MPa)).

Utilizando el Nivel 12 de filtrado (retención de ~1,001 registros), se buscará correlacionar la microestructura, el método de procesamiento y la temperatura de prueba con el comportamiento de fluencia del material, lo cual representa el mayor valor agregado para reducir costos en la experimentación física de nuevas aleaciones.

3. Alternativas de Exploración Analítica Adicional
Para maximizar el aprovechamiento de los datos, el proyecto contempla la integración de enfoques complementarios que no dependan exclusivamente de la predicción de variables continuas:

Aprendizaje No Supervisado (Clustering) sobre el 80% del dataset: Al relajar la restricción de poseer propiedades mecánicas explícitas, se retendrán más de 1,200 filas para aplicar algoritmos de agrupamiento estadístico. Esto permitirá identificar familias o clústeres de aleaciones con comportamientos microestructurales similares basándose únicamente en su estequiometría y procesamiento.

Sistemas de Inferencia de Procesamiento: Desarrollar una arquitectura inversa que, dada una composición química específica y una estructura cristalográfica deseada (BCC/FCC), recomiende el método de fabricación (ej. fundición, forja, pulvimetalurgia) estadísticamente más viable para garantizar la estabilidad del material.

# Módulo de Young: Pipeline Unificado de Machine Learning

Se refactorizó y unificó la arquitectura de procesamiento dentro del script `Modulo_de_young/main_modulo_de_young.py`, estableciendo un flujo reproducible y automatizado orientado a la preparación de matrices de datos para algoritmos de regresión y aprendizaje automático en Google Colab.

### Ejecución del Pipeline Unificado

1. **Carga y Verificación de Integridad**:
   - Lectura directa de `Modulo_de_young/dataset.csv` (1,008 registros curados).
   - Aplicación de `.dropna()` para asegurar la ausencia total de valores nulos antes de la extracción de descriptores.

2. **Featurización Composicional con Matminer**:
   - `StrToComposition`: Conversión de la columna `Formula` en representaciones de composición de `pymatgen`.
   - `MultipleFeaturizer`: Extracción automatizada de descriptores físicos y termodinámicos utilizando exactamente cuatro clases:
     - `ElementProperty.from_preset("magpie")`: Propiedades elementales estadísticas (electronegatividad, radio covalente, volumen atómico, etc.).
     - `YangSolidSolution()`: Parámetros característicos de soluciones sólidas (desajuste atómico $\delta$, parámetro termodinámico $\Omega$, entalpía y entropía de mezcla).
     - `ValenceOrbital(props=['frac'])`: Fracción de electrones de valencia en orbitales s, p, d y f.
     - `ElementFraction()`: Vector de fracción molar estequiométrica elemental.
   - Depuración del objeto intermedio `composition` para evitar anomalías durante la serialización tabular.

3. **Procesamiento Matemático (One-Hot Encoding Binario)**:
   - Codificación One-Hot mediante `pd.get_dummies()` sobre las cuatro variables categóricas del dataset:
     - `Estructura_Cristalina`
     - `Metodo_Procesamiento`
     - `Microestructura`
     - `Tipo_Ensayo`
   - Parámetro estricto `dtype=int`: Garantiza que todas las columnas generadas contengan valores enteros binarios (0 y 1), evitando tipos de datos booleanos (`True`/`False`).
   - Eliminación de la columna de texto original `Formula`, garantizando que la matriz resultante no contenga variables residuales de tipo string (`object`).

4. **Resultado y Disponibilidad para Google Colab**:
   - Archivo generado: `Modulo_de_young/dataset_final_ml.csv`.
   - Dimensiones: **1,008 filas y 306 columnas**.
   - Integridad: 0 valores nulos, 140 variables enteras binarias y 166 variables continuas de punto flotante.
   - La matriz se encuentra 100% numéricamente estructurada y lista para su consumo directo por modelos de machine learning (Random Forest, XGBoost, LightGBM, redes neuronales, etc.).