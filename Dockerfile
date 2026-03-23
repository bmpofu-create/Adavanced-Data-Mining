# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional: gcc for some ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipenv files first (for dependency caching)
COPY Pipfile Pipfile.lock /app/

# Install pipenv and dependencies
RUN pip install pipenv && \
    pipenv install --deploy --ignore-pipfile

# Copy application code (including app.py, templates/, static/, model file)
COPY . /app

# Expose Flask port
EXPOSE 9696

# Run with Gunicorn (bind to all interfaces, port 9696)
CMD ["pipenv", "run", "gunicorn", "-b", "0.0.0.0:9696", "app:app"]
