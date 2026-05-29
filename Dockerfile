FROM python:3.11-slim

WORKDIR /app

# Install dependencies utama langsung agar efisien
RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy joblib fastapi uvicorn prometheus-fastapi-instrumentator

# Copy seluruh file dari repository ke dalam container
COPY . .

# Menerima RUN_ID sebagai argumen saat build dan menjadikannya env variable
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]