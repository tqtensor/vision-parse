from .parser import VisionParser, PDFPageConfig, VisionParserError, UnsupportedFileError
from .llm import LLMError, UnsupportedModelError
from .utils import ImageExtractionError
from importlib.metadata import version, PackageNotFoundError
from .constants import SUPPORTED_MODELS

try:
    __version__ = version("vision-parse")
except PackageNotFoundError:
    # Use a development version when package is not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "VisionParser",
    "PDFPageConfig",
    "ImageExtractionError",
    "VisionParserError",
    "UnsupportedFileError",
    "UnsupportedModelError",
    "LLMError",
    "SUPPORTED_MODELS",
    "__version__",
]
