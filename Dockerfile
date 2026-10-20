# Serves the already-trained model. Run `python train.py` locally first so that
# artifacts/model.joblib and artifacts/metrics.json exist before building.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features.py app.py ./
COPY static/ static/
COPY artifacts/ artifacts/

# Hugging Face Spaces uses 7860; Render and others set $PORT
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
