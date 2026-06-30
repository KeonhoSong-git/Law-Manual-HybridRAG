FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy
COPY kg_api.py ./
ENV KG_TTL=/data/kg.ttl VECTORS_PATH=/data/vectors.jsonl
EXPOSE 8800
CMD ["uvicorn", "kg_api:app", "--host", "0.0.0.0", "--port", "8800"]
