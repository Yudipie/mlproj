FROM python:3.13-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds that
# only change application code, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the runtime code/artifacts the API actually needs — not notebooks,
# tests, or raw data.
COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
