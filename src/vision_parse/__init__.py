from .parser import VisionParser, PDFPageConfig, VisionParserError, UnsupportedFileError
from .llm import LLMError, UnsupportedModelError
from importlib.metadata import version

try:
    __version__ = version("vision-parse")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "VisionParser",
    "PDFPageConfig",
    "VisionParserError",
    "UnsupportedFileError",
    "UnsupportedModelError",
    "LLMError",
]
