import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Configuration MLflow ---
import os
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

MODEL_NAME = "iris-model"
FORCE_PRODUCTION = True  # Phase 1 = True, Phase 2 = False

# --- Chargement des données ---
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Choix du modèle ---
# Phase 1 : LogisticRegression
model = LogisticRegression(max_iter=200)

# Phase 2 : RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# --- Entraînement ---
model.fit(X_train, y_train)

# --- Enregistrement dans MLflow ---
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_metric("train_score", model.score(X_train, y_train))

    run_id = run.info.run_id
    print(f"Modèle enregistré avec run_id={run_id}")

# --- Optionnel : forcer l'alias 'Production' ---
if FORCE_PRODUCTION:
    # Récupérer la dernière version du modèle enregistré
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    latest_version = versions[-1].version  # dernière version
    client.set_registered_model_alias(name=MODEL_NAME, version=latest_version, alias="Production")
    print(f"Alias 'Production' forcé sur la version {latest_version}")

print("✅ Modèle prêt et enregistré dans MLflow")