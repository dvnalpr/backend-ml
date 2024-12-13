# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies (Flask, TensorFlow, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and all other necessary files
COPY . .

# Expose the port for the Flask app (5000)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
