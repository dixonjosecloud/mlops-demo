FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy

CMD ["python3", "train_with_mlflow.py"]
