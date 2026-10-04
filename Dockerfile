FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Generate the synthetic data and train the TensorFlow intent model at build time so containers start ready.
RUN python scripts/generate_data.py --anchor 2026-10-03 && python scripts/train_intent_model.py \
    && python scripts/build_vector_store.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000 8001 8501
CMD ["uvicorn", "banking_ai.api.assistant_app:app", "--host", "0.0.0.0", "--port", "8000"]
