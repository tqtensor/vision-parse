# Vision Parse

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/vision-parse)](https://pypi.org/project/vision-parse)

> 🚀 Transform PDF documents into beautifully formatted markdown using state-of-the-art Vision Language Models - all with just a few lines of code!

## 🎯 Introduction

Vision Parse harnesses the power of Vision Language Models to revolutionize document processing:

- 📝 **Smart Content Extraction**: Intelligently identifies and extracts text, tables, and images with high precision
- ✨ **Markdown Magic**: Converts complex document layouts into clean, well-structured markdown
- 🎨 **Format Preservation**: Maintains document hierarchy, styling, and visual elements
- 🤖 **AI-Powered Analysis**: Leverages cutting-edge vision models for superior accuracy
- 🔄 **Batch Processing**: Handle multi-page documents effortlessly


## 🚀 Getting Started

### Prerequisites

- 🐍 Python >= 3.8
- 🖥️ Ollama (for local model hosting)

### Installation

Install the package using pip:

```bash
pip install vision-parse
```

### Setting up Ollama

1. **Install Ollama** based on your operating system:

   **Linux:**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   **MacOS:**
   ```bash
   brew install ollama
   ```

   **Windows:**
   Download and install from [Ollama Website](https://ollama.com/download/OllamaSetup.exe)

2. **Pull and start** the Ollama server:
   ```bash
   ollama pull llama3.2-vision:11b
   ollama serve
   ```

3. **Verify** server status:
   ```bash
   curl http://localhost:11434/api/version
   ```


## ⌛️ Usage

### Basic Example

```python
from vision_parse import VisionParser

# Initialize parser
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.7
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages, 1):
    print(f"\n--- Page {i} ---\n{page_content}")
```

### Custom Configuration

```python
from vision_parse import VisionParser, PDFPageConfig

# Configure PDF processing settings
page_config = PDFPageConfig(
    dpi=400,
    color_space="RGB",
    include_annotations=True,
    preserve_transparency=False
)

# Initialize parser with custom config
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.7,
    page_config=page_config
)

# Convert PDF to markdown
pdf_path = "path/to/your/document.pdf"
markdown_pages = parser.convert_pdf(pdf_path)
```


## 🛠️ Development

### Setting Up Development Environment

1. **Clone and Setup**:
   ```bash
   # Clone the repository
   git clone https://github.com/iamarunbrahma/vision-parse.git
   cd vision-parse
   ```

2. **Install Dependencies**:
   ```bash
   # Install uv (Mac and Linux)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install uv (Windows)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Install dependencies
   uv sync --all-extras && source .venv/bin/activate
   ```

3. **Quality Checks**:
   ```bash
   # Run test suite
   make test
   
   # Code quality
   make lint    # Run code linting
   make format  # Format code
   ```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
