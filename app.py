import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Reset logic
if 'reset' in st.session_state and st.session_state['reset']:
    st.session_state.clear()
    st.rerun()

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    .stButton>button {background-color: #28a745; color: white; border: none; border-radius: 5px;}
    .stNumberInput input, .stSelectbox select {border-color: #28a745 !important;}
    .stNumberInput input:focus, .stSelectbox select:focus {border-color: #28a745 !important;}
    body {
        background-image: url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None


# Header
st.markdown('<h1 style="color: black; text-align: center; font-size: 2.5em;">🩺💙 Diabetes Prediction System</h1>', unsafe_allow_html=True)
st.markdown("### Outil d’évaluation des risques")

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please run the following command first:
    ```
    python diabetes_prediction.py
    ```
    This will train and save the model files.
    """)
    st.stop()

# Custom CSS for green accent color
st.markdown("""
    <style>
    .stButton>button {background-color: #28a745; color: white; border: none; border-radius: 5px;}
    .stNumberInput input, .stSelectbox select {border-color: #28a745 !important;}
    .stNumberInput input:focus, .stSelectbox select:focus {border-color: #28a745 !important;}
    </style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.title("Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.selectbox('Age', list(range(21, 101)), index=9)
pregnancies = st.sidebar.selectbox('Pregnancies', list(range(21)))

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.selectbox('Glucose (mg/dL)', list(range(0, 201, 10)), index=12)
bp = st.sidebar.selectbox('Blood Pressure (mm Hg)', list(range(0, 131, 10)), index=7)
skin = st.sidebar.selectbox('Skin Thickness (mm)', list(range(0, 101, 5)), index=4)
insulin = st.sidebar.selectbox('Insulin (mu U/ml)', list(range(0, 901, 50)), index=1)
bmi = st.sidebar.selectbox('BMI', [round(x * 0.1, 1) for x in range(100, 701, 10)], index=15)
dpf = st.sidebar.selectbox('Diabetes Pedigree Function', [round(x * 0.01, 2) for x in range(0, 251)], index=50)

# Predict button
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

# Main content
if predict_btn:
    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Standardize
    input_std = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_std)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(input_std)[0]
        prob_negative = probability[0] * 100
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive
    
    # Display results
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="color: #28a745; text-align: center;">Prédiction : {'Négative' if prediction == 0 else 'Positive'}</h2>
            <p style="text-align: center; font-size: 18px; color: #6c757d;">{'Aucun risque de diabète détecté' if prediction == 0 else 'Risque de diabète détecté'}</p>
            <hr style="border: 1px solid #dee2e6;">
            <h4 style="color: #495057;">Probabilité de diabète :</h4>
            <div style="display: flex; align-items: center;">
                <div style="flex: 1;">
                    <div style="background-color: #e9ecef; border-radius: 5px; height: 20px; width: 100%;">
                        <div style="background-color: #28a745; height: 100%; width: {prob_positive}%; border-radius: 5px;"></div>
                    </div>
                </div>
                <div style="margin-left: 10px; font-size: 16px; color: #495057;">{prob_positive:.1f}%</div>
            </div>
            <hr style="border: 1px solid #dee2e6;">
            <h4 style="color: #495057;">Évaluation du risque :</h4>
            <div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                {'Risque faible' if prediction == 0 else 'Risque élevé'}
            </div>
            <p style="color: #6c757d; font-size: 16px; margin-top: 10px;">
                {'Le modèle prédit une faible probabilité de diabète. Continuez à maintenir un mode de vie sain avec une activité physique régulière et une alimentation équilibrée.' if prediction == 0 else 'Le modèle prédit un risque élevé de diabète. Consultez un professionnel de santé pour un suivi approprié.'}
            </p>
            <hr style="border: 1px solid #dee2e6;">
            <h4 style="color: #495057;">Vos valeurs d'entrée :</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #e9ecef;">
                        <th style="padding: 10px; text-align: left; border: 1px solid #dee2e6;">Métrique</th>
                        <th style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">Valeur</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Grossesses</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{pregnancies}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Glucose</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{glucose}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Pression artérielle</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{bp}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Épaisseur de la peau</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{skin}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Insuline</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{insulin}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">IMC</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{bmi}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Fonction de Pédigree du Diabète</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{dpf}</td></tr>
                    <tr><td style="padding: 10px; border: 1px solid #dee2e6;">Âge</td><td style="padding: 10px; text-align: right; border: 1px solid #dee2e6;">{age}</td></tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add buttons below the table
    summary = f"""Prediction Results

Prediction: {'Negative' if prediction == 0 else 'Positive'}
Probability of Diabetes: {prob_positive:.1f}%

Input Values:
Pregnancies: {pregnancies}
Glucose: {glucose}
Blood Pressure: {bp}
Skin Thickness: {skin}
Insulin: {insulin}
BMI: {bmi}
Diabetes Pedigree Function: {dpf}
Age: {age}
"""
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("🖨️ Print Results", data=summary, file_name="prediction_results.txt", mime="text/plain", use_container_width=True)
    with col2:
        st.button("🔄 New Prediction", on_click=lambda: st.session_state.update({'reset': True}), use_container_width=True)
    
    # Risk factors
    st.markdown("---")
    st.subheader("⚠️ Analyse des Facteurs de Risque")

    risk_factors = []
    positive_factors = []

    if glucose > 125:
        risk_factors.append("🔴 Niveau de Glucose Élevé (>125 mg/dL)")
    elif glucose < 100:
        positive_factors.append("🟢 Niveau de Glucose Normal")

    if bmi > 30:
        risk_factors.append("🔴 IMC Élevé - Obésité (>30)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 IMC Sain (18.5-24.9)")

    if age > 45:
        risk_factors.append("🟡 Facteur d'Âge (>45)")

    if bp > 80:
        risk_factors.append("🔴 Pression Artérielle Élevée (>80 mm Hg)")
    elif 60 <= bp <= 80:
        positive_factors.append("🟢 Pression Artérielle Normale")

    if dpf > 0.5:
        risk_factors.append("🟡 Prédisposition Génétique Élevée")

    if risk_factors:
        st.warning("**Facteurs de Risque Identifiés :**")
        st.markdown(
            "<ul style='list-style-type: none; padding: 0;'>" +
            "".join([f"<li style='color: #dc3545; font-size: 16px;'>• {factor}</li>" for factor in risk_factors]) +
            "</ul>", unsafe_allow_html=True
        )

    if positive_factors:
        st.success("**Indicateurs de Santé Positifs :**")
        st.markdown(
            "<ul style='list-style-type: none; padding: 0;'>" +
            "".join([f"<li style='color: #28a745; font-size: 16px;'>• {factor}</li>" for factor in positive_factors]) +
            "</ul>", unsafe_allow_html=True
        )

    # Recommandations
    st.markdown("---")
    st.subheader("💡 Recommandations")

    if prediction == 1:
        st.error("""
        **Actions Importantes :**
        - Consultez un professionnel de la santé immédiatement
        - Obtenez un dépistage complet du diabète
        - Surveillez régulièrement votre glycémie
        - Envisagez des modifications de votre mode de vie
        """)
    else:
        st.success("""
        **Maintenez des Pratiques Saines :**
        - Effectuez des bilans de santé réguliers
        - Adoptez une alimentation équilibrée
        - Faites de l'exercice régulièrement (30+ minutes par jour)
        - Surveillez votre poids et votre IMC
        """)

    # Avertissement
    st.markdown("---")
    st.warning("""
    **⚠️ AVERTISSEMENT MÉDICAL**

    Cette prédiction est à des fins éducatives uniquement. Elle NE doit PAS remplacer 
    les conseils médicaux professionnels. Consultez toujours des professionnels de santé qualifiés 
    pour toute préoccupation médicale.
    """)

else:
    # Page initiale
    st.markdown("---")
    st.info("Bonjour 👋, remplissez les informations du patient dans la barre latérale à gauche." \
    "Merci et prenez soin de votre santé 🩺💙 ")

    col1, col2, col3 = st.columns(3)
    col1.metric("Type de Modèle", "SVM")
    col2.metric("Précision", "~78%")
    col3.metric("Jeu de Données", "768 échantillons")