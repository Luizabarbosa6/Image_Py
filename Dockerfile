FROM python:3.10-slim

# Reduz tamanho: remove cache, desativa pip cache, etc.
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instala dependências básicas
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
