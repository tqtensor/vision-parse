import logging
import re
from importlib.resources import files
from typing import Any, Dict, Literal, Optional

import fitz
import instructor
from jinja2 import Template
from litellm import acompletion, completion
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .constants import PROVIDER_PREFIXES, SUPPORTED_PROVIDERS
from .exceptions import LLMError, UnsupportedProviderError
from .utils import ImageData

logger = logging.getLogger(__name__)


class ImageDescription(BaseModel):
    """Defines the schema for image description and analysis results.

    Attributes:
        text_detected (Literal["Yes", "No"]): Indicates whether text is present in the image.
        tables_detected (Literal["Yes", "No"]): Indicates whether tables are present in the image.
        images_detected (Literal["Yes", "No"]): Indicates whether embedded images are present.
        latex_equations_detected (Literal["Yes", "No"]): Indicates whether LaTeX equations are present.
        extracted_text (str): Contains the raw text extracted from the image.
        confidence_score_text (float): Provides the confidence score for text extraction accuracy.
    """

    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    latex_equations_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class LLM:
    try:
        _image_analysis_prompt = Template(
            files("vision_parse").joinpath("prompts/image_analysis.j2").read_text()
        )
        _md_prompt_template = Template(
            files("vision_parse").joinpath("prompts/markdown_prompt.j2").read_text()
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt files: {str(e)}")

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        provider_config: Optional[Dict] = None,
        image_mode: Literal["url", "base64", None] = None,
        custom_prompt: Optional[str] = None,
        detailed_extraction: bool = False,
        enable_concurrency: bool = False,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.provider_config = provider_config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.image_mode = image_mode
        self.custom_prompt = custom_prompt
        self.detailed_extraction = detailed_extraction
        self.kwargs = kwargs
        self.enable_concurrency = enable_concurrency

        self.provider = self._get_provider_name(model_name)
        self._init_llm()

    def _init_llm(self) -> None:
        """Initializes the LLM client with LiteLLM integration.

        This method sets up the instructor client with appropriate completion mode
        based on concurrency settings.

        Raises:
            LLMError: If client initialization fails.
        """

        try:
            # Initialize instructor client with litellm
            completion_func = acompletion if self.enable_concurrency else completion
            self.client = instructor.from_litellm(completion_func)
        except Exception as e:
            raise LLMError(f"Unable to initialize LLM client: {str(e)}")

    def _get_provider_name(self, model_name: str) -> str:
        """Determines the provider name based on the model name prefix.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            str: The provider name (e.g., 'openai', 'gemini').

        Raises:
            UnsupportedProviderError: If the model name doesn't match any known provider.
        """

        for provider, prefixes in PROVIDER_PREFIXES.items():
            if any(model_name.startswith(prefix) for prefix in prefixes):
                return provider

        supported_providers = ", ".join(
            f"{name} ({provider})" for provider, name in SUPPORTED_PROVIDERS.items()
        )
        raise UnsupportedProviderError(
            f"Model '{model_name}' is not from a supported provider. "
            f"Supported providers are: {supported_providers}"
        )

    def _get_model_params(self, structured: bool = False) -> Dict[str, Any]:
        """Constructs model parameters based on provider and configuration.

        Args:
            structured (bool, optional): Whether to use structured output parameters.
                Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing model parameters for API calls.
        """

        # Base parameters common across providers
        params = {
            "model": self.model_name,
            "temperature": 0.0 if structured else self.temperature,
            "top_p": 0.4 if structured else self.top_p,
            **self.kwargs,
        }

        # Add API key and provider-specific config
        params.update({"api_key": self.api_key, **self.provider_config})

        return params

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _get_response(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Retrieves response from LLM using LiteLLM and instructor.

        Args:
            base64_encoded (str): Base64 encoded image data.
            prompt (str): The prompt to send to the LLM.
            structured (bool, optional): Whether to use structured output.
                Defaults to False.

        Returns:
            Any: The LLM response, either as structured JSON or raw text.

        Raises:
            LLMError: If LLM processing fails.
        """

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}"
                            },
                        },
                    ],
                }
            ]

            params = self._get_model_params(structured)

            if self.enable_concurrency:
                # Async path
                if structured:
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        response_model=ImageDescription,
                        **params,
                    )
                    return response.model_dump_json()
                else:
                    # For non-structured responses, use str as the response model
                    response = await self.client.chat.completions.create(
                        messages=messages,
                        response_model=str,
                        **params,
                    )
            else:
                # Sync path
                if structured:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        response_model=ImageDescription,
                        **params,
                    )
                    return response.model_dump_json()
                else:
                    # For non-structured responses, use str as the response model
                    response = self.client.chat.completions.create(
                        messages=messages,
                        response_model=str,
                        **params,
                    )

            # Process the response for non-structured output
            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response,
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"LLM processing failed: {str(e)}")

    async def generate_markdown(
        self, base64_encoded: str, pix: fitz.Pixmap, page_number: int
    ) -> Any:
        """Generates markdown formatted text from an image using the configured LLM provider.

        This method handles both detailed and simple extraction modes, processes
        embedded images if configured, and formats the output as markdown.

        Args:
            base64_encoded (str): Base64 encoded image data.
            pix (fitz.Pixmap): The image pixmap for additional processing.
            page_number (int): The page number being processed.

        Returns:
            Any: The generated markdown text, including any embedded images if configured.
        """

        extracted_images = []
        if self.detailed_extraction:
            try:
                response = await self._get_response(
                    base64_encoded,
                    self._image_analysis_prompt.render(),
                    structured=True,
                )

                json_response = ImageDescription.model_validate_json(response)

                if json_response.text_detected.strip() == "No":
                    return ""

                if (
                    float(json_response.confidence_score_text) > 0.6
                    and json_response.tables_detected.strip() == "No"
                    and json_response.latex_equations_detected.strip() == "No"
                    and (
                        json_response.images_detected.strip() == "No"
                        or self.image_mode is None
                    )
                ):
                    return json_response.extracted_text

                if (
                    json_response.images_detected.strip() == "Yes"
                    and self.image_mode is not None
                ):
                    extracted_images = ImageData.extract_images(
                        pix, self.image_mode, page_number
                    )

                prompt = self._md_prompt_template.render(
                    extracted_text=json_response.extracted_text,
                    tables_detected=json_response.tables_detected,
                    latex_equations_detected=json_response.latex_equations_detected,
                    confidence_score_text=float(json_response.confidence_score_text),
                    custom_prompt=self.custom_prompt,
                )

            except Exception:
                logger.warning(
                    "Detailed extraction failed. Falling back to simple extraction."
                )
                self.detailed_extraction = False

        if not self.detailed_extraction:
            prompt = self._md_prompt_template.render(
                extracted_text="",
                tables_detected="Yes",
                latex_equations_detected="No",
                confidence_score_text=0.0,
                custom_prompt=self.custom_prompt,
            )

        markdown_content = await self._get_response(
            base64_encoded, prompt, structured=False
        )

        if extracted_images:
            if self.image_mode == "url":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.image_url})"
                    )
            elif self.image_mode == "base64":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.base64_encoded})"
                    )

        return markdown_content
