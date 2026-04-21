FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn gymnasium numpy openai python-dotenv

COPY guardian/ ./guardian/
COPY setup.py .

RUN pip install -e . --no-deps

EXPOSE 8000

CMD ["uvicorn", "guardian.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
