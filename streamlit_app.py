import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Charger les modèles et le scaler
clf_tree = load('decision_tree.joblib')
clf_rf = load('random_forest.joblib')
clf_lr = load('logistic_regression.joblib')
scaler = load('scaler.joblib')

# Lire le dataset pour récupérer les options de catégories
@st.cache_data
def load_options():
    df = pd.read_csv('agriculture_burundi.csv')
    provinces = sorted(df['province'].dropna().unique())
    cultures = sorted(df['culture'].dropna().unique())
    return provinces, cultures

provinces, cultures = load_options()

st.title('Prédiction de bonne récolte - Burundi')
st.markdown(
    'Entrez les caractéristiques de la parcelle pour obtenir la prédiction de bonne ou mauvaise récolte ' 
    'avec trois modèles de machine learning.'
)

with st.sidebar:
    st.header('Paramètres d’entrée')
    annee = st.number_input('Année', min_value=2015, max_value=2030, value=2022, step=1)
    saison = st.selectbox('Saison', ['A', 'B'])
    province = st.selectbox('Province', provinces)
    culture = st.selectbox('Culture', cultures)
    altitude_m = st.number_input('Altitude (m)', min_value=0, max_value=3000, value=1500)
    pluviometrie_mm = st.number_input('Pluviométrie (mm)', min_value=0.0, max_value=2000.0, value=800.0)
    temperature_moy_C = st.number_input('Température moyenne (°C)', min_value=-10.0, max_value=40.0, value=20.0)
    superficie_ha = st.number_input('Superficie (ha)', min_value=0.01, max_value=100.0, value=1.0, format='%.2f')
    utilisation_engrais = st.selectbox('Utilisation d’engrais', [0, 1])
    acces_irrigation = st.selectbox('Accès à l’irrigation', [0, 1])
    nb_menages = st.number_input('Nombre de ménages', min_value=1, max_value=1000, value=5)

st.markdown('---')

input_data = pd.DataFrame([
    {
        'annee': annee,
        'saison': saison,
        'province': province,
        'culture': culture,
        'altitude_m': altitude_m,
        'pluviometrie_mm': pluviometrie_mm,
        'temperature_moy_C': temperature_moy_C,
        'superficie_ha': superficie_ha,
        'utilisation_engrais': utilisation_engrais,
        'acces_irrigation': acces_irrigation,
        'nb_menages': nb_menages,
    }
])

st.subheader('Données saisies')
st.dataframe(input_data)

@st.cache_data
def prepare_input(df, reference_columns):
    proc = df.copy()
    proc['saison'] = proc['saison'].map({'A': 0, 'B': 1})
    proc = pd.get_dummies(proc, columns=['province', 'culture'], drop_first=True)
    for col in reference_columns:
        if col not in proc.columns:
            proc[col] = 0
    proc = proc[reference_columns]
    numeric_cols = ['annee', 'altitude_m', 'pluviometrie_mm', 'temperature_moy_C', 'superficie_ha', 'utilisation_engrais', 'acces_irrigation', 'nb_menages']
    proc[numeric_cols] = scaler.transform(proc[numeric_cols])
    return proc

# Charger les colonnes de référence depuis le jeu d'entraînement
@st.cache_data
def load_reference_columns():
    df = pd.read_csv('agriculture_burundi.csv')
    clean = df.dropna(subset=['bonne_recolte']).copy()
    clean = clean.drop(columns=['bonne_recolte', 'rendement_t_ha', 'production_totale_t'])
    clean['saison'] = clean['saison'].map({'A': 0, 'B': 1})
    clean = pd.get_dummies(clean, columns=['province', 'culture'], drop_first=True)
    return clean.columns.tolist()

reference_columns = load_reference_columns()
processed_input = prepare_input(input_data, reference_columns)

if st.button('Prédire'):
    models = [
        (clf_tree, 'Arbre de décision'),
        (clf_rf, 'Forêt aléatoire'),
        (clf_lr, 'Régression logistique'),
    ]
    results = []
    for model, name in models:
        pred = model.predict(processed_input)[0]
        prob = model.predict_proba(processed_input)[0, 1]
        results.append({
            'Modèle': name,
            'Prédiction': 'Bonne récolte' if pred == 1 else 'Mauvaise récolte',
            'Probabilité (%)': round(prob * 100, 1),
        })

    st.subheader('Résultats')
    st.table(pd.DataFrame(results))

    st.markdown('### Interprétation rapide')
    st.write('Les résultats sont donnés pour chaque modèle. Comparez les prédictions et les probabilités pour juger de la confiance.')

    st.markdown('---')
    st.write('Pour un déploiement réel, exécutez cette app avec :')
    st.code('streamlit run streamlit_app.py')
