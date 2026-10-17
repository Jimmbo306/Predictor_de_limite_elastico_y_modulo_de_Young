"""
Servidor Backend para la Interfaz de Modelos Predictivos de Ciencia de Materiales.
Archivo: Front/app.py

Proporciona una API REST y sirve la interfaz web para:
1. Predicción del Módulo de Young Calculado (XGBoost, 27 descriptores).
2. Predicción del Límite Elástico YS (XGBoost, 51 descriptores).
Calcula los descriptores composicionales en tiempo real mediante matminer y pymatgen.
"""

import os
import sys
import json
import pickle
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Rutas del proyecto
CARPETA_FRONT = Path(__file__).resolve().parent
DIRECTORIO_RAIZ = CARPETA_FRONT.parent
CARPETA_MODELS = DIRECTORIO_RAIZ / "Models"

# Inicializar Flask
app = Flask(
    __name__,
    template_folder=str(CARPETA_FRONT / "templates"),
    static_folder=str(CARPETA_FRONT / "static"),
)

# Inicializar featurizador de matminer de forma perezosa/global
featurizer = None
m_young = None
m_ys = None
info_young = None
info_ys = None


def cargar_recursos():
    """Carga featurizadores de matminer y modelos de XGBoost."""
    global featurizer, m_young, m_ys, info_young, info_ys

    if featurizer is None:
        print("[Iniciando] Cargando featurizadores de matminer...")
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.composition import (
            ElementProperty,
            YangSolidSolution,
            ValenceOrbital,
            ElementFraction,
        )

        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset("magpie"),
            YangSolidSolution(),
            ValenceOrbital(props=["frac"]),
            ElementFraction(),
        ])
        print("[OK] Featurizadores listos.")

    if m_young is None:
        path_young_pkl = CARPETA_MODELS / "xgboost_Calculated Young modulus (GPa).pkl"
        path_young_json = CARPETA_MODELS / "variables_Calculated Young modulus (GPa).json"
        with open(path_young_pkl, "rb") as f:
            m_young = pickle.load(f)
        with open(path_young_json, "r", encoding="utf-8") as f:
            info_young = json.load(f)
        print("[OK] Modelo de Módulo de Young cargado.")

    if m_ys is None:
        path_ys_pkl = CARPETA_MODELS / "xgboost_Limite_Elastico_YS_MPa.pkl"
        path_ys_json = CARPETA_MODELS / "variables_Limite_Elastico_YS_MPa.json"
        with open(path_ys_pkl, "rb") as f:
            m_ys = pickle.load(f)
        with open(path_ys_json, "r", encoding="utf-8") as f:
            info_ys = json.load(f)
        print("[OK] Modelo de Límite Elástico cargado.")


# Offsets de centrado del conjunto de entrenamiento
# Los modelos XGBoost fueron entrenados en Colab sobre los residuos con respecto a la media de entrenamiento (y - y_mean).
# Para recuperar el valor físico absoluto en la inferencia, se suma el valor medio del conjunto de entrenamiento:
OFFSET_YOUNG = 166.04  # GPa (media histórica del target en Modulo_de_young)
OFFSET_YS = 879.38     # MPa (media histórica del target en Limite_elastico)


@app.route("/")
def index():
    """Página principal de la interfaz."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Favicon para la pestaña del navegador."""
    return app.send_static_file("favicon.svg")


@app.route("/api/info", methods=["GET"])
def get_info():
    """Retorna metadatos y variables de los modelos."""
    try:
        cargar_recursos()
        return jsonify({
            "success": True,
            "modulo_young": info_young,
            "limite_elastico": info_ys,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict/young", methods=["POST"])
def predict_young_endpoint():
    """Endpoint para predecir el Módulo de Young (GPa)."""
    try:
        cargar_recursos()
        import pandas as pd
        from pymatgen.core import Composition

        data = request.get_json() or {}
        formula_raw = data.get("formula", "").strip()
        densidad_raw = data.get("density", None)

        if not formula_raw:
            return jsonify({"success": False, "error": "Debe proporcionar una fórmula química válida."}), 400

        try:
            densidad = float(densidad_raw)
            if densidad <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "La densidad calculada debe ser un número positivo (g/cm³)."}), 400

        # Validar y parsear fórmula con pymatgen
        try:
            comp = Composition(formula_raw)
        except Exception as e:
            return jsonify({"success": False, "error": f"Fórmula química no válida: {str(e)}"}), 400

        # Featurizar con matminer (directo sin multiprocessing)
        feat_vals = featurizer.featurize(comp)
        feat_dict = dict(zip(featurizer.feature_labels(), feat_vals))

        # Agregar variable externa que NO viene de la librería
        feat_dict["Calculated Density (g/cm$^3$)"] = densidad

        # Construir fila en el orden exacto de variables del modelo
        row = {col: feat_dict.get(col, 0.0) for col in m_young.feature_names_in_}
        df_in = pd.DataFrame([row])

        # Predecir con XGBoost y sumar el offset de centrado del dataset de entrenamiento
        prediccion_residual = float(m_young.predict(df_in)[0])
        prediccion = prediccion_residual + OFFSET_YOUNG

        # Obtener desglose elemental exacto (comp.fractional_composition ya está normalizada en base 1.0)
        fracciones = {el.symbol: round(float(amt) * 100, 2) for el, amt in comp.fractional_composition.items()}

        # Descriptores clave calculados para mostrar en frontend
        descriptores_muestra = {
            "Densidad calculada": f"{densidad:.2f} g/cm³",
            "Electronegatividad media": f"{feat_dict.get('MagpieData mean Electronegativity', 0.0):.3f}",
            "Punto de fusión máx": f"{feat_dict.get('MagpieData maximum MeltingT', 0.0):.1f} K",
            "Volumen atómico modal": f"{feat_dict.get('MagpieData mode GSvolume_pa', 0.0):.2f} Å³/átomo",
            "Fracción e⁻ orbital s": f"{feat_dict.get('frac s valence electrons', 0.0)*100:.1f}%",
            "Fracción e⁻ orbital d": f"{feat_dict.get('frac d valence electrons', 0.0)*100:.1f}%",
        }

        return jsonify({
            "success": True,
            "prediccion": round(prediccion, 2),
            "unidad": "GPa",
            "formula_normalizada": comp.reduced_formula,
            "peso_molecular": round(comp.weight, 2),
            "fracciones_elementales": fracciones,
            "descriptores_clave": descriptores_muestra,
            "metricas_modelo": info_young.get("rendimiento", {}),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error en la predicción: {str(e)}"}), 500


@app.route("/api/predict/limite_elastico", methods=["POST"])
def predict_ys_endpoint():
    """Endpoint para predecir el Límite Elástico YS (MPa)."""
    try:
        cargar_recursos()
        import pandas as pd
        from pymatgen.core import Composition

        data = request.get_json() or {}
        formula_raw = data.get("formula", "").strip()
        temp_raw = data.get("temp", 25.0)
        tipo_ensayo = data.get("tipo_ensayo", "C").upper()
        metodo_proc = data.get("metodo_proc", "CAST").upper()
        est_cristalina = data.get("est_cristalina", "FCC").upper()
        microestructura = data.get("microestructura", "FCC").upper()

        if not formula_raw:
            return jsonify({"success": False, "error": "Debe proporcionar una fórmula química válida."}), 400

        try:
            temp = float(temp_raw)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "La temperatura de prueba debe ser un número (°C)."}), 400

        try:
            comp = Composition(formula_raw)
        except Exception as e:
            return jsonify({"success": False, "error": f"Fórmula química no válida: {str(e)}"}), 400

        # Featurizar con matminer
        feat_vals = featurizer.featurize(comp)
        feat_dict = dict(zip(featurizer.feature_labels(), feat_vals))

        # Variables externas que NO da la librería
        feat_dict["Temperatura_Prueba_C"] = temp
        feat_dict["Tipo_Ensayo_C"] = 1.0 if tipo_ensayo == "C" else 0.0
        feat_dict["Metodo_Procesamiento_POWDER"] = 1.0 if metodo_proc == "POWDER" else 0.0
        feat_dict["Metodo_Procesamiento_WROUGHT"] = 1.0 if metodo_proc == "WROUGHT" else 0.0
        feat_dict["Estructura_Cristalina_FCC"] = 1.0 if est_cristalina == "FCC" else 0.0

        feat_dict["Microestructura_BCC+FCC"] = 1.0 if microestructura in ["BCC+FCC", "BCC_FCC"] else 0.0
        feat_dict["Microestructura_FCC+BCC"] = 1.0 if microestructura in ["FCC+BCC", "FCC_BCC"] else 0.0
        feat_dict["Microestructura_BCC+HCP"] = 1.0 if microestructura in ["BCC+HCP", "BCC_HCP"] else 0.0
        feat_dict["Microestructura_FCC+B2+Sec."] = 1.0 if microestructura in ["FCC+B2+SEC.", "FCC+B2+SEC", "FCC_B2_SEC"] else 0.0

        # Construir fila en orden exacto
        row = {col: feat_dict.get(col, 0.0) for col in m_ys.feature_names_in_}
        df_in = pd.DataFrame([row])

        # Predecir con XGBoost y sumar el offset de centrado del dataset de entrenamiento
        prediccion_residual = float(m_ys.predict(df_in)[0])
        prediccion = prediccion_residual + OFFSET_YS

        # Obtener desglose elemental exacto (comp.fractional_composition normalizada a 1.0)
        fracciones = {el.symbol: round(float(amt) * 100, 2) for el, amt in comp.fractional_composition.items()}

        descriptores_muestra = {
            "Temperatura de prueba": f"{temp:.1f} °C",
            "Parámetro Yang (delta)": f"{feat_dict.get('Yang delta', 0.0)*100:.2f}%",
            "Electronegatividad media": f"{feat_dict.get('MagpieData mean Electronegativity', 0.0):.3f}",
            "Peso atómico medio": f"{feat_dict.get('MagpieData mean AtomicWeight', 0.0):.2f} u",
            "Radio covalente medio": f"{feat_dict.get('MagpieData mean CovalentRadius', 0.0):.2f} pm",
            "Volumen atómico medio": f"{feat_dict.get('MagpieData mean GSvolume_pa', 0.0):.2f} Å³/átomo",
        }

        return jsonify({
            "success": True,
            "prediccion": round(prediccion, 2),
            "unidad": "MPa",
            "formula_normalizada": comp.reduced_formula,
            "peso_molecular": round(comp.weight, 2),
            "fracciones_elementales": fracciones,
            "descriptores_clave": descriptores_muestra,
            "metricas_modelo": info_ys.get("rendimiento", {}),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error en la predicción: {str(e)}"}), 500


def iniciar_servidor(abrir_navegador: bool = True, puerto: int = 5000):
    """Inicializa los recursos y levanta el servidor Flask abriendo el navegador."""
    cargar_recursos()
    url = f"http://127.0.0.1:{puerto}"
    print("\n" + "=" * 70)
    print("MPEA PREDICTOR: INTERFAZ WEB ACTIVA")
    print(f"Servidor disponible en: {url}")
    print("=" * 70)

    if abrir_navegador:
        import webbrowser
        import threading
        print("[Auto-Launch] Abriendo navegador web automáticamente...")
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="127.0.0.1", port=puerto, debug=False)


if __name__ == "__main__":
    iniciar_servidor(abrir_navegador=True, puerto=5000)

