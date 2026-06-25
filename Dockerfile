# Use a lightweight python base image
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies, including Tesseract OCR for ocrmypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    icc-profiles-free \
    libpng-dev \
    libjpeg-dev \
    zlib1g-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies (ignoring system clashes and pinning pip limits)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Healthcheck to verify container health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the application
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
