# import streamlit as st
# import requests
# import pandas as pd

# st.set_page_config(page_title="ML Factory - Iris Predictor", layout="centered")
# st.title("ML Factory - Iris Predictor")

# # Chargement du dataset de test
# df = pd.read_csv("data/iris_test.csv")
# row = st.selectbox("Choisir une ligne", df.index)
# data = df.iloc[row].tolist()

# if st.button("Predict"):
#     try:
#         res = requests.post("http://api:8000/predict", json=data).json()
#         st.success(f"Prediction: {res.get('prediction', 'N/A')}")
#         st.info(f"Model version: {res.get('model_version', 'N/A')}")
#         st.write("Probabilities:")
#         st.write(res.get("probabilities", []))
#     except requests.exceptions.ConnectionError:
#         st.error("Impossible de se connecter à l'API. Vérifiez que le service 'api' est en ligne.")
#     except KeyError:
#         st.error("La réponse de l'API ne contient pas de prédiction. Le modèle n'est peut-être pas encore publié.")

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ML Factory - Iris Predictor", layout="centered")
st.title("🌸 ML Factory - Iris Predictor")

# --- Chargement du dataset de test ---
df = pd.read_csv("data/iris_test.csv")
row = st.selectbox("Choisir une ligne à prédire", df.index)

# Affichage des valeurs de la ligne sélectionnée
st.subheader("📄 Valeurs de l'échantillon sélectionné")
st.table(df.iloc[row])

data = df.iloc[row].tolist()

# --- Bouton de prédiction ---
if st.button("Predict"):
    try:
        res = requests.post("http://api:8000/predict", json=data).json()

        # Affichage de la prédiction
        prediction = res.get("prediction", "N/A")
        st.success(f"Prediction: **{prediction}**")

        # Badge pour version du modèle
        model_version = res.get("model_version", "N/A")
        st.markdown(f"**Model version:** `{model_version}`")

        # Probabilités sous forme de tableau
        probs = res.get("probabilities", [])
        if probs:
            probs_df = pd.DataFrame(probs, columns=[f"Class {i}" for i in range(len(probs[0]))])
            st.subheader("Probabilities")
            st.dataframe(probs_df)
        else:
            st.warning("Aucune probabilité disponible. Le modèle peut ne pas être chargé.")

    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API. Vérifiez que le service 'api' est en ligne.")
    except Exception as e:
        st.error(f"⚠️ Erreur inattendue : {e}")