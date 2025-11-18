# Base Python image
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Expose port
EXPOSE 8080

# Gunicorn command (Flask app file = app.py, variable = app)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
