# Docker Setup Guide for Vision Parse

This guide explains how to set up Vision Parse using Docker on macOS and Linux systems.

## Prerequisites

- Docker and Docker Compose installed on your system
- Nvidia GPU (optional, but recommended for better performance)

### macOS
1. Install Docker Desktop for Mac from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
2. For Apple Silicon (M1/M2) users, ensure you're using ARM64 compatible images

### Linux
1. Install Docker Engine:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```
2. Install Docker Compose:
   ```bash
   sudo apt-get install docker-compose
   ```
3. For GPU support, install NVIDIA Container Toolkit:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update

   sudo apt-get install -y nvidia-container-toolkit
   ```

## Environment Setup

Export the required environment variables in your terminal:
```bash
# Required: Choose one of the following models
export MODEL_NAME=llama3.2-vision:11b  # select the model name from the list of supported models

# Optional: API keys (required only for specific models)
export OPENAI_API_KEY=your_openai_api_key
export GEMINI_API_KEY=your_gemini_api_key
```

## Running Docker Container

1. If you have an Nvidia GPU, uncomment the following lines in your docker-compose.yml:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. Build and start the container:
   ```bash
   # Build the image
   docker compose build

   # Start the container in detached mode
   docker compose up -d
   ```

## Troubleshooting

1. If using Ollama-based models, ensure port 11434 is not being used by another service:
```bash
# macOS
lsof -i :11434

# Linux
sudo netstat -tulpn | grep 11434
```

2. Check container logs for any errors:
```bash
docker compose logs vision-parse
```

## Stopping the Container

To stop the Vision Parse container:
```bash
docker compose down
```