from importlib.metadata import PackageNotFoundError, version

from .constants import SUPPORTED_MODELS
from .llm import LLMError, UnsupportedModelError
from .parser import PDFPageConfig, UnsupportedFileError, VisionParser, VisionParserError
from .utils import ImageExtractionError

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
