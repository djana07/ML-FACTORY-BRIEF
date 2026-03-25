FROM python:3.10

WORKDIR /app

COPY app.py .

RUN pip install streamlit pandas requests

COPY app.py .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]