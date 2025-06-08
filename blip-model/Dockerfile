FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /root/.cache /var/lib/apt/lists/*


EXPOSE 7860

CMD ["python", "app.py"]