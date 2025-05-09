from typing import Dict, List

SUPPORTED_PROVIDERS: Dict[str, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "gemini": "Google AI Studio",
    "deepseek": "DeepSeek",
}

# Common model prefixes for provider detection
PROVIDER_PREFIXES: Dict[str, List[str]] = {
    "openai": ["gpt", "litellm"],
    "azure": ["gpt"],
    "gemini": ["gemini"],
    "deepseek": ["deepseek"],
}
