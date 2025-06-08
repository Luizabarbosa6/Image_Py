FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /root/.cache /var/lib/apt/lists/*

COPY . .

CMD ["python", "app.py"]
