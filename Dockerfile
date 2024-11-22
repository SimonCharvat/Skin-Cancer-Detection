# Use an official lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /src

# Copy the required library list from the local machine to the container
COPY requirements.txt .

# Install all the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code to the container
COPY src/ .

# Set the exposed port for Streamlit (default 8501)
EXPOSE 8501

# Command to execute the Python script
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
