FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Modificar requirements.txt para reemplazar ta-lib con pandas-ta
COPY requirements.txt .
RUN sed -i 's/ta-lib==0.4.28/pandas-ta==0.3.14b0/g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Variables de entorno predeterminadas
ENV POSTGRES_HOST=postgres
ENV POSTGRES_PORT=5432
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=trading
ENV LOG_LEVEL=INFO
ENV CONFIG_DIR=/app/config

# Comando para ejecutar el servicio
ENTRYPOINT ["python", "pattern_detector.py"]
