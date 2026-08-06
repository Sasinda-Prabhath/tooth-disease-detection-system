FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY shared-lib /shared-lib
COPY services/module2-impaction-boneloss/requirements.txt .
RUN pip install --no-cache-dir -e /shared-lib && \
    pip install --no-cache-dir -r requirements.txt

COPY services/module2-impaction-boneloss/app ./app
COPY services/module2-impaction-boneloss/models_store ./models_store

ENV MODELS_DIR=/app/models_store
ENV PIXEL_SPACING_DEFAULT=0.1
ENV PYTHONPATH=/app

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
