<div align='center'>

# Vision Parse ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)

> Parse PDF documents into beautifully formatted markdown content using state-of-the-art Vision Language Models - all with just a few lines of code!

[Getting Started](#-getting-started) •
[Usage](#-usage) •
[Tested Models](#-tested-models) •
[Configuration](#-configuration-options)
</div>

## 🎯 Introduction

Vision Parse harnesses the power of Vision Language Models to revolutionize document processing:

- 📝 **Scanned Document Processing**: Intelligently identifies and extracts text, tables, and LaTeX equations from scanned documents into markdown-formatted content with high precision
- 🎨 **Advanced Content Formatting**: Preserves LaTeX equations, hyperlinks, images, and document hierarchy for markdown-formatted content
- 🤖 **Multi-LLM Support**: Seamlessly integrates with multiple Vision LLM providers such as OpenAI, Gemini, and DeepSeek for optimal accuracy and speed

## ⚠️ Important Notice

> [!NOTE]
> This repository is a revised version of the original work by [Arun Brahma](https://github.com/iamarunbrahma/vision-parse).
>
> The key improvements in this fork include:
> - Integration with [LiteLLM](https://github.com/BerriAI/litellm) to support multiple LLM providers with a unified interface
> - Implementation of [instructor](https://github.com/567-labs/instructor) for structured outputs and improved response handling
> - Enhanced reliability and performance with multiple Vision LLM providers

## 🚀 Getting Started

### Prerequisites

- 🐍 Python >= 3.9
- 🤖 API Key for OpenAI, Google Gemini, or DeepSeek

### Installation

**Install the package from source:**

```bash
pip install 'git+https://github.com/tqtensor/vision-parse.git#egg=vision-parse[all]'
```

## 📚 Usage

### Basic Example Usage

```python
from vision_parse import VisionParser

# Initialize parser
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-openai-api-key",
    temperature=0.4,
    top_p=0.5,
    image_mode="url", # image mode can be "url", "base64" or None
    detailed_extraction=False, # set to True for more detailed extraction
    enable_concurrency=False, # set to True for parallel processing
)

# Convert PDF to markdown
pdf_path = "input_document.pdf" # local path to your PDF file
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
```

### API Models Usage (OpenAI, Azure OpenAI, Gemini, DeepSeek)

```python
from vision_parse import VisionParser


# Initialize parser with OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-openai-api-key", # get the OpenAI API key from https://platform.openai.com/api-keys
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False, # set to True for more detailed extraction
    enable_concurrency=True,
)

# Initialize parser with Azure OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-azure-openai-api-key", # replace with your Azure OpenAI API key
    image_mode="url",
    detailed_extraction=False, # set to True for more detailed extraction
    enable_concurrency=True,
    provider_config={
        "base_url": "https://****.openai.azure.com/", # replace with your Azure endpoint URL
        "api_version": "2024-08-01-preview", # replace with latest Azure OpenAI API version
        "azure": True, # specify that this is Azure OpenAI
        "azure_deployment": "*******", # replace with Azure deployment name
    },
)

# Initialize parser with Google Gemini model
parser = VisionParser(
    model_name="gemini-1.5-flash",
    api_key="your-gemini-api-key", # get the Gemini API key from Google AI Studio: https://aistudio.google.com/app/apikey
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False, # set to True for more detailed extraction
    enable_concurrency=True,
)

# Initialize parser with DeepSeek model
parser = VisionParser(
    model_name="deepseek/deepseek-chat",
    api_key="your-deepseek-api-key", # get the DeepSeek API key from https://platform.deepseek.com/api_keys
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False, # set to True for more detailed extraction
    enable_concurrency=True,
)
```

## ✅ Tested Models

The following Vision LLM models have been thoroughly tested with Vision Parse, but thanks to our [LiteLLM](https://github.com/BerriAI/litellm) integration, you can experiment with other vision-capable models as well:

|  **Model Name**  | **Provider Name** |
| :--------------: | :---------------: |
|      gpt-4o      |      OpenAI       |
|   gpt-4o-mini    |      OpenAI       |
|     gpt-4.1      |      OpenAI       |
|  gemini-1.5-pro  | Google AI Studio  |
| gemini-2.0-flash | Google AI Studio  |
|  deepseek-chat   |     DeepSeek      |

> [!TIP]
> To use other vision-capable models, simply pass the appropriate model identifier as supported by LiteLLM. For a complete list of supported providers and models, check the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

## 🔧 Configuration Options

### Core Parameters

- **model_name** `(str)`: Name of the Vision LLM model to use (e.g., "gpt-4o", "gemini-1.5-flash")
- **api_key** `(str)`: API key for the chosen provider
- **temperature** `(float)`: Controls randomness in the generation (0.0-1.0)
- **top_p** `(float)`: Controls diversity via nucleus sampling (0.0-1.0)

### Content Processing Options

- **detailed_extraction** `(bool)`: When `True`, enables advanced extraction of complex elements (LaTeX, tables, etc.)
- **custom_prompt** `(str)`: Custom instructions to guide the model's extraction behavior
- **image_mode** `(str)`: How images are handled in the output ("url", "base64", or `None`)
- **enable_concurrency** `(bool)`: When `True`, processes multiple pages in parallel

### Provider-Specific Configuration

The `provider_config` parameter lets you configure provider-specific settings through a unified interface:

```python
# For OpenAI
provider_config = {
    "base_url": "https://api.openai.com/v1",  # optional
    "max_retries": 3,                         # optional
    "timeout": 240.0,                         # optional
}

# For Azure OpenAI
provider_config = {
    "base_url": "https://your-resource.openai.azure.com/",
    "api_version": "2024-08-01-preview",
    "azure": True,
    "azure_deployment": "your-deployment-name",
}

# For Gemini (Google AI Studio)
provider_config = {
    "max_retries": 3,    # optional
    "timeout": 240.0,    # optional
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
