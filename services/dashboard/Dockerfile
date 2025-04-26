FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Variables de entorno predeterminadas
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=trading
ENV IBKR_SERVICE_URL=http://ibkr-connector:8080/api
ENV RISK_SERVICE_URL=http://risk-manager:8080/api
ENV LOG_LEVEL=INFO
ENV CONFIG_DIR=/app/config

# Puerto para la aplicación
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
