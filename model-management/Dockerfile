FROM python:3.12-slim
RUN pip install --no-cache-dir "mlflow==3.17.1"
WORKDIR /app
EXPOSE 8742
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8742", "--workers", "1", "--backend-store-uri", "sqlite:////data/mlflow.db", "--serve-artifacts", "--artifacts-destination", "/data/artifacts", "--allowed-hosts", "192.168.68.88:8742,training:8742,localhost:8742,127.0.0.1:8742"]
