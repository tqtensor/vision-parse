class UnsupportedProviderError(BaseException):
    """Raises an error when the specified LLM provider is not supported.

    This exception is raised when attempting to use a model from an unsupported
    LLM provider or when the model name does not match any known provider prefix.
    """

    pass


class LLMError(BaseException):
    """Raises an error when LLM processing encounters a failure.

    This exception is raised when there are issues during LLM initialization,
    API calls, or response processing.
    """

    pass


class UnsupportedFileError(BaseException):
    """Raises an error when an unsupported file type is provided.

    This exception is raised when attempting to process a file that is not a PDF.
    """

    pass


class VisionParserError(BaseException):
    """Raises an error when there is an error in the PDF to markdown conversion process.

    This exception is raised when there are issues during the conversion of PDF pages
    to markdown format, such as conversion failures or processing errors.
    """

    pass
