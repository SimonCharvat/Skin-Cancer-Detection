# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary source code
# This will respect .dockerignore and exclude unnecessary files
COPY src/app.py .

# Set the exposed port for Streamlit (default 8501)
EXPOSE 8501

# Command to execute the Python script
CMD ["streamlit", "run", "app.py", "--server.port=8501"]