#for local
# FROM python:3.10.12-slim

#for openshift
FROM --platform=linux/amd64 python:3.10-slim-buster as build

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .

# Install ffmpeg and other dependencies, compulsory for openshift
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . .
RUN chmod -R 777 logs

# EXPOSE 8000

# Run the app using Gunicorn (with 4 workers)
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]
CMD ["python", "main.py"]
