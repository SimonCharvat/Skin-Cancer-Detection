# Skin Cancer Classification MLOPS Project

This project uses a machine learning model to classify skin cancer types based on input images. The application is containerized using Docker for easy deployment and scalability.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Downloading the Project](#downloading-the-project)
3. [Installation](#installation)
4. [Running the Application Locally](#running-the-application-locally)
5. [Building the Docker Image](#building-the-docker-image)
6. [Running the Docker Container](#running-the-docker-container)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- Git
- Python 3.8 or higher
- pip (Python package manager)
- Docker

## Downloading the Project

1. Open your terminal or command prompt.
2. Clone the repository from GitLab using the following command:

   ```
   git clone https://gitlab.com/zidi_nonplus_ultra/mlops.git
   ```

 

3. Navigate to the project directory:

   ```
   cd your-project-name
   ```

## Installation

1. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   ```

2. Activate the virtual environment:

   - On Windows:
     ```
     venv\\Scripts\\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Running the Application Locally

To run the application locally without Docker, use the following command:

```
python src/run_app.py
```

The application should now be running on `http://localhost:8500/8501` (or another port if specified in the code).

## Building the Docker Image

1. Ensure you are in the project root directory.
2. Build the Docker image using the following command:

   ```
   docker build -t project .
   ```

   This command builds a Docker image with the tag `project`.

## Running the Docker Container

After building the Docker image, you can run the container using the following command:

```
docker run -p 8501:8501 project
```

This command does the following:
- `-p 8501:8501`: Maps port 8501 of the container to port 8501 on your host machine.
- `project`: Specifies the image to use for creating the container.

The application should now be accessible at `http://localhost:8501`.

## Usage

1. Open a web browser and navigate to `http://localhost:8501`.
2. Upload an image of a skin lesion using the provided interface.
3. Click the "Classify" button to get the prediction result.

## Troubleshooting

If you encounter any issues, try the following steps:

1. Ensure all prerequisites are correctly installed.
2. Check that you're using the correct Python version.
3. Verify that all required ports are available and not in use by other applications.
4. If using Docker, make sure the Docker daemon is running.

For any persistent problems, please open an issue on the GitLab repository.

```

This README provides a comprehensive guide for users to download, install, and run your skin cancer classification MLOPS project, both locally and using Docker. It includes detailed steps for each process and a troubleshooting section to help users resolve common issues. You may want to customize some parts of this README to better fit your specific project structure and requirements.