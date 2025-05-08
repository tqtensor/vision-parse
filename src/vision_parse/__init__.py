from importlib.metadata import PackageNotFoundError, version

from .constants import SUPPORTED_PROVIDERS
from .llm import LLMError, UnsupportedProviderError
from .parser import PDFPageConfig, UnsupportedFileError, VisionParser, VisionParserError
from .utils import ImageExtractionError

try:
    __version__ = version("vision-parse")
except PackageNotFoundError:
    # Use a development version when package is not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "ImageExtractionError",
    "LLMError",
    "PDFPageConfig",
    "SUPPORTED_PROVIDERS",
    "UnsupportedFileError",
    "UnsupportedProviderError",
    "VisionParser",
    "VisionParserError",
    "__version__",
]
