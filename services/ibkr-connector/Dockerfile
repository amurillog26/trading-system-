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
ENV IBKR_HOST=127.0.0.1
ENV IBKR_PORT=7496
ENV IBKR_CLIENT_ID=1
ENV LOG_LEVEL=INFO
ENV CONFIG_DIR=/app/config

# Puerto para API
EXPOSE 8080

# Comando para ejecutar el servicio
CMD ["python", "ibkr_connector.py"]
