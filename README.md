# Vision Parse

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)
[![PyPI version](https://img.shields.io/pypi/v/vision-parse.svg)](https://pypi.org/project/vision-parse/)

> 🚀 Parse PDF documents into beautifully formatted markdown content using state-of-the-art Vision Language Models - all with just a few lines of code!

## 🎯 Introduction

Vision Parse harnesses the power of Vision Language Models to revolutionize document processing:

- 📝 **Smart Content Extraction**: Intelligently identifies and extracts text, tables, and LaTeX equations with high precision
- 🎨 **Advanced Content Formatting**: Preserves LaTeX equations, hyperlinks, images, and footnotes for markdown-formatted content
- 🤖 **Multi-LLM Support**: Seamlessly integrates with multiple Vision LLM providers such as OpenAI, Gemini, and Llama for optimal accuracy and speed
- 🔄 **Scanned PDF Document Processing**: Extracts text, tables, images, and LaTeX equations from scanned PDF documents into well-structured markdown content
- 📁 **Local Model Hosting**: Supports local model hosting with Ollama for secure, no-cost, private, and offline document processing


## 🚀 Getting Started

### Prerequisites

- 🐍 Python >= 3.9
- 🖥️ Ollama (if you want to use local models)
- 🤖 API Key for OpenAI or Google Gemini (if you want to use OpenAI or Google Gemini)

### Installation

**Install the core package using pip (Recommended):**

```bash
pip install vision-parse
```

**Install the additional dependencies for OpenAI or Gemini:**

```bash
# For OpenAI support
pip install 'vision-parse[openai]'
```

```bash
# For Gemini support
pip install 'vision-parse[gemini]'
```

```bash
# To install all the additional dependencies
pip install 'vision-parse[all]'
```

**Install the package from source:**

```bash
pip install 'git+https://github.com/iamarunbrahma/vision-parse.git#egg=vision-parse[all]'
```

### Setting up Ollama (Optional)
See [examples/ollama_setup.md](examples/ollama_setup.md) on how to setup Ollama locally.

## ⌛️ Usage

### Basic Example Usage

```python
from vision_parse import VisionParser

# Initialize parser
parser = VisionParser(
    model_name="llama3.2-vision:11b", # For local models, you don't need to provide the api key
    temperature=0.4,
    top_p=0.5,
    image_mode="url", # Image mode can be "url", "base64" or None
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=False, # Set to True for parallel processing
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf" # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
```

### Customize Ollama configuration for better performance

```python
from vision_parse import VisionParser

custom_prompt = """
Strictly preserve markdown formatting during text extraction from scanned document.
"""

# Initialize parser with Ollama configuration
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.6,
    num_ctx=4096,
    image_mode="base64",
    custom_prompt=custom_prompt,
    detailed_extraction=True,
    ollama_config={
        "OLLAMA_NUM_PARALLEL": "8",
        "OLLAMA_REQUEST_TIMEOUT": "240.0",
    },
    enable_concurrency=True,
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)
```

### OpenAI or Gemini Model Usage

```python
from vision_parse import VisionParser

# Initialize parser with OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-openai-api-key", # Get the OpenAI API key from https://platform.openai.com/api-keys
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=True, # Set to True for more detailed extraction
    enable_concurrency=True,
)

# Initialize parser with Google Gemini model
parser = VisionParser(
    model_name="gemini-1.5-flash",
    api_key="your-gemini-api-key", # Get the Gemini API key from https://aistudio.google.com/app/apikey
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=True, # Set to True for more detailed extraction
    enable_concurrency=True,
)
```

## ✅ Supported Models

This package supports the following Vision LLM models:

- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Google Gemini: `gemini-1.5-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`
- Meta Llama and LLava from Ollama: `llava:13b`, `llava:34b`, `llama3.2-vision:11b`, `llama3.2-vision:70b`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
