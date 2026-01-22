# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# --- FIX: Install system dependencies for LightGBM ---
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# -----------------------------------------------------

# Copy requirements first (to leverage Docker caching)
COPY requirements-backend.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-backend.txt

# Copy the rest of the application
COPY . .

# Set PYTHONPATH so python can find your 'src' modules
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]