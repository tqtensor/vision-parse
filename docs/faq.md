# Frequently Asked Questions (FAQ)

1. **Does Vision Parse work with scanned PDF documents?**

    Yes, Vision Parse is specifically designed to handle scanned PDF documents effectively. It uses advanced Vision LLMs to extract text, tables, images, and LaTeX equations from both regular and scanned PDF documents with high precision.

2. **What are the recommended values for model parameters to improve extraction accuracy?**

    Here are the recommended values for model parameters to improve extraction accuracy:
    - Set `temperature` to 0.7 and `top_p` to 0.5

    - For OpenAI models:
        - Increase `max_tokens` to 8192 (depending on the model size)
        - Set `frequency_penalty` to 0.3

    Note: These recommended values are generic and may need to be adjusted based on your document structure and the model's capabilities.
