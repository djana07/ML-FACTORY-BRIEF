FROM python:3.10

WORKDIR /app

COPY main.py .

RUN pip install fastapi uvicorn mlflow boto3 pandas scikit-learn python-multipart

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]