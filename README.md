# 🚀 ML Factory Project — Pipeline MLOps Zero-Downtime

## 📌 Présentation

Ce projet démontre une **architecture MLOps prête pour la production** où les modèles de machine learning sont **totalement découplés** de l’application qui les consomme.

Le système permet :

* Chargement dynamique des modèles
* Mise à jour sans interruption (zero-downtime)
* Prédictions en temps réel
* Versioning et traçabilité des modèles

---

## 🏗️ Architecture

```
                +-------------------+
                |   Streamlit UI    |
                |   (Frontend)      |
                +---------+---------+
                          |
                          v
                +-------------------+
                |     FastAPI       |
                |      (API)        |
                +---------+---------+
                          |
                          v
                +-------------------+
                |     MLflow        |
                |  Model Registry   |
                +---------+---------+
                          |
                          v
                +-------------------+
                |      MinIO        |
                |   Object Storage  |
                +-------------------+
```

---

## 📁 Structure du projet

```
ml-factory-project/
├── data/
│   └── iris_test.csv
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── front/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── train/
│       ├── train.py
│       └── Dockerfile
├── docs/
├── docker-compose.yml
├── pyproject.toml
├── .env
└── README.md
```

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/djana07/ML-FACTORY-BRIEF.git
cd ml-factory-project
```

### 2. Installer les dépendances

```bash
uv sync
```

### 3. Configurer `.env`

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MINIO_ENDPOINT=minio:9000
```

---

## 🐳 Lancer le projet

```bash
docker compose up --build
```

---

## 🤖 Entraînement du modèle

```bash
docker compose run --rm train python train.py
```

---

## 🔮 API

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '[5.1,3.5,1.4,0.2]'
```

---

## 🖥️ Frontend

[http://localhost:8501](http://localhost:8501)

---

## 📚 Documentation

```bash
cd docs
uv run make html
```

---

## 🧪 Tests

```bash
pytest --cov=src
```

---

## 🧹 Lint

```bash
ruff check .
```

---

## 🔐 Sécurité

```bash
gitleaks detect
```

---

## ⚡ CI/CD

* Lint
* Tests
* Sécurité
* Build Docker

---

## 👨‍💻 Auteur

Djana
