# Vision Parse

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)
[![PyPI version](https://img.shields.io/pypi/v/vision-parse.svg)](https://pypi.org/project/vision-parse/)

> 🚀 Parse PDF documents into beautifully formatted markdown content using state-of-the-art Vision Language Models - all with just a few lines of code!

## 🎯 Introduction

Vision Parse harnesses the power of Vision Language Models to revolutionize document processing:

- 📝 **Smart Content Extraction**: Intelligently identifies and extracts text and tables with high precision
- 🎨 **Content Formatting**: Preserves document hierarchy, styling, and indentation for markdown formatted content
- 🤖 **Multi-LLM Support**: Supports multiple Vision LLM providers i.e. OpenAI, LLama, Gemini etc. for accuracy and speed
- 🔄 **PDF Document Support**: Handle multi-page PDF documents effortlessly by converting each page into byte64 encoded images
- 📁 **Local Model Hosting**: Supports local model hosting using Ollama for secure document processing and for offline use


## 🚀 Getting Started

### Prerequisites

- 🐍 Python >= 3.9
- 🖥️ Ollama (if you want to use local models)
- 🤖 API Key for OpenAI or Google Gemini (if you want to use OpenAI or Google Gemini)

### Installation

Install the package using pip:

```bash
pip install vision-parse
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
    temperature=0.9,
    top_p=0.4,
    extraction_complexity=False # Set to True for more detailed extraction
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
```

### PDF Page Configuration

```python
from vision_parse import VisionParser, PDFPageConfig

# Configure PDF processing settings
page_config = PDFPageConfig(
    dpi=400,
    color_space="RGB",
    include_annotations=True,
    preserve_transparency=False
)

# Initialize parser with custom page config
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.9,
    top_p=0.4,
    extraction_complexity=True,
    page_config=page_config
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
    temperature=0.9,
    top_p=0.4,
    extraction_complexity=False # Set to True for more detailed extraction
)

# Initialize parser with Google Gemini model
parser = VisionParser(
    model_name="gemini-1.5-flash",
    api_key="your-gemini-api-key", # Get the Gemini API key from https://aistudio.google.com/app/apikey
    temperature=0.9,
    top_p=0.4,
    extraction_complexity=False # Set to True for more detailed extraction
)
```

## Supported Models

This package supports the following Vision LLM models:

- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Google Gemini: `gemini-1.5-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`
- Meta Llama and LLava from Ollama: `llava:13b`, `llava:34b`, `llama3.2-vision:11b`, `llama3.2-vision:70b`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.