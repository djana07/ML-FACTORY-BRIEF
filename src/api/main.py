from fastapi import FastAPI, HTTPException, Body
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import os

app = FastAPI(title="ML Factory API")

# Configuration
MODEL_NAME = "iris-model"
MODEL_ALIAS = "Production"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model = None
current_version = None

def load_model():
    """
    Charge le modèle depuis MLflow si l'alias 'Production' change de version.
    """
    global model, current_version
    try:
        latest = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    except Exception:
        latest = None

    if not latest:
        model = None
        current_version = None
        return

    if current_version != latest.version:
        print(f"[MLFactory] Hot-reload du modèle version {latest.version}")
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        current_version = latest.version


@app.get("/")
def home():
    load_model()
    return {
        "status": "ok",
        "model_version": current_version
    }


@app.post("/predict")
def predict(data: list = Body(...)):
    load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle en Production disponible")

    df = pd.DataFrame([data])

    try:
        preds = model.predict(df)
        probs = model.predict_proba(df).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

    return {
        "prediction": int(preds[0]),
        "probabilities": probs,
        "model_version": current_version
    }