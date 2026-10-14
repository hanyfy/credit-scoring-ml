FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
ENV PYTHONPATH=/app/src
# Entraîne au build (télécharge le jeu public German Credit depuis OpenML)
RUN python -m scoring.train
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
