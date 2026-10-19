"""
Servidor Backend para la Interfaz de Modelos Predictivos de Ciencia de Materiales.
Archivo: Front/app.py
"""
import os
import sys
import json
import pickle
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

DIRECTORIO_RAIZ = Path(__file__).resolve().parent
CARPETA_FRONT = DIRECTORIO_RAIZ / "Front"
CARPETA_MODELS = DIRECTORIO_RAIZ / "Models"

app = Flask(
    __name__,
    template_folder=str(CARPETA_FRONT / "templates"),
    static_folder=str(CARPETA_FRONT / "static"),
)

featurizer = None
m_young = None
m_ys = None
info_young = None
info_ys = None

def cargar_recursos():
    global featurizer, m_young, m_ys, info_young, info_ys
    if featurizer is None:
        print("[Iniciando] Cargando featurizadores de matminer...")
        from matminer.featurizers.base import MultipleFeaturizer
        from matminer.featurizers.composition import ElementProperty, YangSolidSolution, ValenceOrbital, ElementFraction
        featurizer = MultipleFeaturizer([
            ElementProperty.from_preset("magpie"),
            YangSolidSolution(),
            ValenceOrbital(props=["frac"]),
            ElementFraction(),
        ])
    if m_young is None:
        with open(CARPETA_MODELS / "xgboost_Calculated Young modulus (GPa).pkl", "rb") as f:
            m_young = pickle.load(f)
        with open(CARPETA_MODELS / "variables_Calculated Young modulus (GPa).json", "r", encoding="utf-8") as f:
            info_young = json.load(f)
    if m_ys is None:
        with open(CARPETA_MODELS / "xgboost_Limite_Elastico_YS_MPa.pkl", "rb") as f:
            m_ys = pickle.load(f)
        with open(CARPETA_MODELS / "variables_Limite_Elastico_YS_MPa.json", "r", encoding="utf-8") as f:
            info_ys = json.load(f)

OFFSET_YOUNG = 166.04 
OFFSET_YS = 879.38 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.svg")

@app.route("/api/info", methods=["GET"])
def get_info():
    try:
        cargar_recursos()
        return jsonify({"success": True, "modulo_young": info_young, "limite_elastico": info_ys})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict/young", methods=["POST"])
def predict_young_endpoint():
    try:
        cargar_recursos()
        import pandas as pd
        from pymatgen.core import Composition
        data = request.get_json() or {}
        formula_raw = data.get("formula", "").strip()
        densidad = float(data.get("density", 0.0))
        comp = Composition(formula_raw)
        
        feat_vals = featurizer.featurize(comp)
        feat_dict = dict(zip(featurizer.feature_labels(), feat_vals))
        feat_dict["Calculated Density (g/cm$^3$)"] = densidad
        
        row = {col: feat_dict.get(col, 0.0) for col in m_young.feature_names_in_}
        df_in = pd.DataFrame([row])
        prediccion = float(m_young.predict(df_in)[0]) + OFFSET_YOUNG
        
        return jsonify({"success": True, "prediccion": round(prediccion, 2), "unidad": "GPa"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/predict/limite_elastico", methods=["POST"])
def predict_ys_endpoint():
    try:
        cargar_recursos()
        import pandas as pd
        from pymatgen.core import Composition
        data = request.get_json() or {}
        comp = Composition(data.get("formula", "").strip())
        
        feat_vals = featurizer.featurize(comp)
        feat_dict = dict(zip(featurizer.feature_labels(), feat_vals))
        feat_dict["Temperatura_Prueba_C"] = float(data.get("temp", 25.0))
        feat_dict["Tipo_Ensayo_C"] = 1.0 if data.get("tipo_ensayo") == "C" else 0.0
        feat_dict["Metodo_Procesamiento_POWDER"] = 1.0 if data.get("metodo_proc") == "POWDER" else 0.0
        feat_dict["Metodo_Procesamiento_WROUGHT"] = 1.0 if data.get("metodo_proc") == "WROUGHT" else 0.0
        feat_dict["Estructura_Cristalina_FCC"] = 1.0 if data.get("est_cristalina") == "FCC" else 0.0
        
        row = {col: feat_dict.get(col, 0.0) for col in m_ys.feature_names_in_}
        df_in = pd.DataFrame([row])
        prediccion = float(m_ys.predict(df_in)[0]) + OFFSET_YS
        
        return jsonify({"success": True, "prediccion": round(prediccion, 2), "unidad": "MPa"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def iniciar_servidor(abrir_navegador=True, puerto=5000):
    cargar_recursos()
    if abrir_navegador:
        import webbrowser, threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{puerto}")).start()
    app.run(host="127.0.0.1", port=puerto, debug=False)

if __name__ == "__main__":
    iniciar_servidor()