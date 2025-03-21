# Skin Cancer Detection and Classification

This project uses a machine learning model to classify skin cancer types based on input images. The application is containerized using Docker for easy deployment and scalability, and features a **user-friendly web interface built with Streamlit**.

> **Note:** This project was originally developed on GitLab and later **transferred to GitHub**, including the full CI/CD pipeline which now runs via **GitHub Actions**.

It was done **in collaboration with other team members** — see the [commit history](https://github.com/simoncharvat/skin-cancer-detection/commits/master) for details.

## Table of Contents
- [Skin Cancer Detection and Classification](#skin-cancer-detection-and-classification)
- [Running the Application as a Container Image](#running-the-application-as-a-container-image)
  - [Pull the Docker Image](#pull-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
- [Running the Application Locally](#running-the-application-locally)
  - [Downloading the Project](#downloading-the-project)
  - [Installation](#installation)
- [Troubleshooting](#troubleshooting)

## Running the Application as a Container Image

The container image is now hosted on **GitHub Container Registry (GHCR)**.

### Pull the Docker Image

Use the following command to pull the latest container image (replace the tag with the desired version):

```bash
docker pull ghcr.io/simoncharvat/skin-cancer-detection:<branch>-<commit_hash>
```

### Run the Docker Container

After pulling the image, run the container:  

```bash
docker run -p 8501:8501 skin-cancer-detection
```

You can now access the web application at `http://localhost:8501`.




## Running the Application Locally

### Downloading the Project

1. Open your terminal or command prompt 
2. Clone the repository from GitHub

```bash
git clone https://github.com/simoncharvat/skin-cancer-detection.git
```

### Installation

Install the required dependencies in file `requirements.txt`:  

```bash
pip install -r requirements.txt
```


You can now run the app without Docker using `run_app.py`:

```bash
python src/run_app.py
```

The application will start and can be accessed via `http://localhost:8501`.


## Troubleshooting

If you encounter any issues:  

1. Verify all prerequisites are installed.  
2. Check that you're using Python 3.8 or newer.  
3. Ensure that port 8501 is not being used by other applications.  
4. Confirm that the Docker daemon is running if using Docker.  

For persistent problems, please open an issue on the [GitHub repository](https://github.com/simoncharvat/skin-cancer-detection/issues).

