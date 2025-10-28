FROM python:3.10-slim

# Install system dependencies
RUN apt update && apt -y upgrade && \
    apt install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools==60.0.0
RUN python3 -m pip install --no-cache-dir -r /opt/requirements.txt

# Create app directory and copy application
RUN mkdir -p /app/output
EXPOSE 8000
COPY ./app /app/app


# If using docker-compose, environment variables are passed via env_file
# If using docker run directly, the .env file will be available in the image
COPY .env /app/.env

WORKDIR /app

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]