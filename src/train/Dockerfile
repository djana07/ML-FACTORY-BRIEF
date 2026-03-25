# src/train/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copier le code d'entraînement
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir mlflow scikit-learn pandas
RUN apt-get update && apt-get install -y git

# Par défaut, rien n’est exécuté ici
CMD ["python", "train.py"]