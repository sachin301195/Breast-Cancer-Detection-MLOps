FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY ./models /app/models

EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT