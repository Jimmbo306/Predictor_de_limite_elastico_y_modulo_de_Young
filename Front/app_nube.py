import streamlit as st
import pandas as pd
import json
import joblib
import traceback
from pathlib import Path

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty, ValenceOrbital, ElementFraction

# 1. Configuración inicial estricta (Sin emojis)
st.set_page_config(
    page_title="Predictor MPEAs", 
    page_icon=":material/science:", 
    layout="centered"
)

# 2. Inyección de CSS para diseño formal (Tipografía Inter y ajustes de UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Diseño riguroso para botones */
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    
    /* Ajuste de los títulos de las pestañas */
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Rutas de directorios
CARPETA_FRONT = Path(__file__).resolve().parent
CARPETA_MODELS = CARPETA_FRONT.parent / "Models"

# ---------------------------------------------------------
# 3. Carga de Recursos
# ---------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    m_young = joblib.load(CARPETA_MODELS / "xgboost_Calculated Young modulus (GPa).pkl")
    with open(CARPETA_MODELS / "variables_Calculated Young modulus (GPa).json", "r", encoding="utf-8") as f:
        v_young = json.load(f)["variables_entrada"]

    m_ys = joblib.load(CARPETA_MODELS / "xgboost_Limite_Elastico_YS_MPa.pkl")
    with open(CARPETA_MODELS / "variables_Limite_Elastico_YS_MPa.json", "r", encoding="utf-8") as f:
        v_ys = json.load(f)["variables_entrada"]
        
    ep = ElementProperty.from_preset("magpie")
    vo = ValenceOrbital(props=['frac'])
    ef = ElementFraction()
    
    return m_young, v_young, m_ys, v_ys, ep, vo, ef

model_young, vars_young, model_ys, vars_ys, ep_feat, vo_feat, ef_feat = cargar_recursos()

OFFSET_YOUNG = 166.04
OFFSET_YS = 879.38

def extraer_features(formula):
    comp = Composition(formula)
    df_temp = pd.DataFrame({"composition": [comp]})
    magpie = ep_feat.featurize_dataframe(df_temp, "composition", ignore_errors=True)
    vo = vo_feat.featurize_dataframe(df_temp, "composition", ignore_errors=True)
    ef = ef_feat.featurize_dataframe(df_temp, "composition", ignore_errors=True)
    return pd.concat([magpie, vo, ef], axis=1).drop(columns=["composition"])

# ---------------------------------------------------------
# 4. Interfaz de Usuario
# ---------------------------------------------------------
st.title("Plataforma Predictiva MPEA")
st.markdown("Inferencia de propiedades mecánicas en aleaciones de alta entropía (MPEAs) a partir de descriptores estequiométricos.")

tab1, tab2 = st.tabs(["Módulo de Young", "Límite Elástico"])

# --- PESTAÑA 1: MÓDULO DE YOUNG ---
with tab1:
    st.subheader("Cálculo de Módulo de Young (E)")
    col1, col2 = st.columns(2)
    
    with col1:
        y_formula = st.text_input("Fórmula estequiométrica", "Al0.5CoCrFeNi", key="y_form")
        y_densidad = st.number_input("Densidad teórica (g/cm³)", value=8.0, key="y_den")
        # Botón sobrio con icono material de cálculo
        btn_young = st.button("Calcular Módulo de Young", type="primary", icon=":material/calculate:")
        
    with col2:
        if btn_young:
            with st.spinner('Evaluando descriptores...'):
                try:
                    df_in = extraer_features(y_formula)
                    df_in["Calculated Density (g/cm^3)"] = float(y_densidad)
                    
                    df_final = pd.DataFrame(columns=vars_young)
                    for col in vars_young:
                        df_final[col] = df_in[col] if col in df_in.columns else 0.0
                            
                    pred = model_young.predict(df_final)[0] + OFFSET_YOUNG
                    st.success(f"**Resultado:** {pred:.2f} GPa", icon=":material/check_circle:")
                except Exception as e:
                    st.error(f"Error de procesamiento: {e}", icon=":material/error:")

# --- PESTAÑA 2: LÍMITE ELÁSTICO ---
with tab2:
    st.subheader("Cálculo de Límite Elástico (YS)")
    col3, col4 = st.columns(2)
    
    with col3:
        ys_formula = st.text_input("Fórmula estequiométrica", "CoCrFeNiMn", key="ys_form")
        ys_temp = st.number_input("Temperatura de ensayo (°C)", value=25.0, key="ys_tmp")
        ys_ensayo = st.selectbox("Tipo de esfuerzo", ["T", "C"])
        ys_proc = st.selectbox("Método de procesamiento", ["CAST", "WROUGHT", "POWDER", "Otro"])
        ys_estructura = st.selectbox("Estructura cristalina", ["FCC", "BCC", "Otra"])
        ys_micro = st.selectbox("Microestructura", ["FCC simple", "BCC simple", "BCC+FCC", "FCC+BCC", "BCC+HCP", "FCC+B2+Sec.", "Otra"])
        # Botón sobrio con icono material de cálculo
        btn_ys = st.button("Calcular Límite Elástico", type="primary", icon=":material/calculate:")
        
    with col4:
        if btn_ys:
            with st.spinner('Evaluando descriptores...'):
                try:
                    df_in = extraer_features(ys_formula)
                    df_in["Temperatura_Prueba_C"] = float(ys_temp)
                    
                    df_in[f"Tipo_Ensayo_{ys_ensayo}"] = 1.0
                    df_in[f"Metodo_Procesamiento_{ys_proc}"] = 1.0
                    df_in[f"Estructura_Cristalina_{ys_estructura}"] = 1.0
                    df_in[f"Microestructura_{ys_micro}"] = 1.0
                    
                    df_final = pd.DataFrame(columns=vars_ys)
                    for col in vars_ys:
                        df_final[col] = df_in[col] if col in df_in.columns else 0.0
                            
                    pred = model_ys.predict(df_final)[0] + OFFSET_YS
                    st.success(f"**Resultado:** {pred:.2f} MPa", icon=":material/check_circle:")
                except Exception as e:
                    st.error(f"Error de procesamiento: {traceback.format_exc()}", icon=":material/error:")