# MPEA Predictor: Interfaz de Usuario para Modelos de Machine Learning

Esta carpeta contiene la aplicación web interactiva diseñada para la inferencia en tiempo real de los modelos entrenados de **Módulo de Young (`Calculated Young modulus (GPa)`)** y **Límite Elástico (`YS (MPa)`)** para Aleaciones de Múltiples Elementos Principales (MPEAs).

---

## 1. Arquitectura y Análisis de Variables de Entrada

Los modelos fueron entrenados utilizando algoritmos de regresión basados en **XGBoost**. La inferencia se alimenta tanto de características químicas extraídas automáticamente por la librería `matminer` como de variables físicas y de procesamiento provistas por el usuario:

### Pestaña 1: Módulo de Young (`Calculated Young modulus (GPa)`)
* **Métricas del modelo**: $R^2 = 0.9969$, $RMSE = 2.53$ GPa.
* **Total de variables**: 27.
* **Variables provistas por la librería (`matminer`) a partir de la fórmula química (26 variables)**:
  - Estadísticas Magpie de electronegatividad, volumen atómico, punto de fusión, número atómico, radios covalentes, etc.
  - Fracción de electrones de valencia en capas de orbitales s, p, d (`frac s/p/d valence electrons`).
  - Fracción molar estequiométrica elemental (`Cr`, `Mo`, etc.).
* **Campos adicionales de petición (no provistos por la librería)**:
  - `Calculated Density (g/cm$^3$)`: Densidad calculada teórica de la aleación.

---

### Pestaña 2: Límite Elástico (`Limite_Elastico_YS_MPa`)
* **Métricas del modelo**: $R^2 = 0.7693$, $RMSE = 268.45$ MPa.
* **Total de variables**: 51.
* **Variables provistas por la librería (`matminer`) a partir de la fórmula química (42 variables)**:
  - Descriptores atómicos de Magpie (`GSvolume_pa`, `CovalentRadius`, `Electronegativity`, `AtomicWeight`, `MeltingT`, etc.).
  - Parámetro de solución sólida de Yang: $\delta$ (`Yang delta`).
  - Distribución de orbitales s (`frac s valence electrons`).
  - Fracción molar elemental (`Al`, `Ni`, `Mo`, `Mn`, `Fe`, `V`).
* **Campos adicionales de petición (no provistos por la librería)**:
  - `Temperatura_Prueba_C`: Temperatura del ensayo mecánico (°C).
  - `Tipo_Ensayo`: Selector entre Compresión (`C` $\to$ `Tipo_Ensayo_C = 1`) y Tracción (`T` $\to$ `Tipo_Ensayo_C = 0`).
  - `Metodo_Procesamiento`: Selector entre Fundición (`CAST`), Forjado/Deformado (`WROUGHT`), Pulvimetalurgia (`POWDER`) u Otro.
  - `Estructura_Cristalina`: Selector entre `FCC`, `BCC` u Otra.
  - `Microestructura`: Selector entre `FCC simple`, `BCC simple`, `BCC+FCC`, `FCC+BCC`, `BCC+HCP`, `FCC+B2+Sec.` u Otra.

---

## 2. Estructura de la Carpeta

```text
Front/
├── app.py                # Servidor backend Flask y API REST de inferencia
├── templates/
│   └── index.html        # Estructura semántica con dos pestañas e iconos SVG (sin emojis)
├── static/
│   ├── css/
│   │   └── style.css     # Sistema de diseño Obsidian Dark, glassmorphism y tipografía Inter
│   └── js/
│       └── app.js        # Lógica de pestañas, consumo de API y renderizado reactivo
└── README.md             # Documentación técnica
```

---

## 3. Ejecución del Servidor Local

Para iniciar la aplicación, ejecuta desde la raíz del proyecto:

```bash
.\.venv\Scripts\python.exe Front/app.py
```

Luego abre en tu navegador preferido:
```
http://127.0.0.1:5000
```
