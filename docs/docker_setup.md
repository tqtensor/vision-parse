# Docker Setup Guide for Vision Parse

This guide explains setting up Vision Parse using Docker on macOS and Linux systems.

## Prerequisites

- Docker and Docker Compose installed on your system
- Nvidia GPU (optional, but recommended for better performance)

## Installation Steps

**macOS**
- Download and install [Docker Desktop](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
- Docker Compose is included in Docker Desktop

**Linux**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose

# For GPU Support (Optional)
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

1. If you have Nvidia GPU, uncomment the following lines in docker-compose.yml:
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
3. Verify the container is running:
   ```bash
   docker ps
   ```

## Running Vision Parse

To run the Vision Parse application:

```bash
# Execute the python script inside the container
docker compose exec vision-parse python docs/examples/gradio_app.py
```

## Troubleshooting

1. If you're using Ollama-based models and encounter connection issues, check if port 11434 is already in use:
```bash
sudo lsof -i :11434
```

2. Check container logs for errors:
```bash
docker compose logs vision-parse
```

## Managing the Container

```bash
# Stop the container and preserve data
docker compose stop

# Stop and remove containers, networks
docker compose down
