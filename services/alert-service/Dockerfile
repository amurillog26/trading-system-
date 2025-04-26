FROM python:3.9-slim

WORKDIR /app

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
ENV LOG_LEVEL=INFO
ENV CHECK_INTERVAL=60
ENV CONFIG_DIR=/app/config

# Puerto para API (si se implementa en el futuro)
EXPOSE 8080

# Comando para ejecutar el servicio
CMD ["python", "alert_service.py"]
