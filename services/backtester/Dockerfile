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
ENV LOG_LEVEL=INFO
ENV CONFIG_DIR=/app/config

# Comando por defecto
ENTRYPOINT ["python", "backtester.py"]
