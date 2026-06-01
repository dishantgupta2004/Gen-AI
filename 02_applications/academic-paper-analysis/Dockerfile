FROM python:3.11-slim

WORKDIR /app

# System deps for PDF processing (pypdf) and unstructured URL loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health || exit 1

# Entry point is inside the app/ subdirectory
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8503", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
