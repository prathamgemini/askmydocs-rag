# Use lightweight Python 3.11 as base image
# slim = no unnecessary OS packages = smaller size
FROM python:3.11-slim

# Create /app folder inside container and work from there
# All next commands will run inside /app
WORKDIR /app

# Copy requirements.txt FIRST (before code)
# Why first? Docker caches this layer
# If only code changes, pip install won't re-run → faster builds
COPY requirements.txt .

# System packages needed at build/run time:
#   build-essential + cmake -> compile llama-cpp-python
#   libgl1 + libglib2.0-0   -> OpenCV (used by Docling / RapidOCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install all Python libraries inside container
# --no-cache-dir = don't save pip cache = smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Now copy all backend code into container
# app/ on your machine → /app inside container
COPY app/ .

# Tell Docker our server runs on port 8000
# This is just documentation, doesn't actually open port
EXPOSE 8000

# Command to start the server when container runs
# --host 0.0.0.0 = accept connections from outside container
# --port 8000 = run on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


