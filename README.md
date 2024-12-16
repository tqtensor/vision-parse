# Multimodal Parser

Extract content from PDF documents using multimodal LLMs in markdown format.

## Usage

```bash
uv run markdown.py [input-file]
```

## Installation

- ***Install uv:***
    - *Mac OS and Linux:*
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    - *Windows:*
        ```bash
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

- ***Install dependencies:***
    ```bash
    uv sync --all-extras && source .venv/bin/activate
    ```

## Host Ollama locally

- ***Install Ollama:***
    - *Linux:*
        ```bash
        curl -fsSL https://ollama.com/install.sh | sh
        ```
    - *Mac OS:*
        ```bash
        brew install ollama
        ```
    - *Windows:*
        Download from [here](https://ollama.com/download/OllamaSetup.exe) and install.

- ***Run Ollama:***
    - Pull models using: `ollama pull [model-name]`
    - Start Ollama server: `ollama serve`
    - Verify if the server is running: `curl http://localhost:11434/api/version`
