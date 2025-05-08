import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import fitz
import nest_asyncio
from pydantic import BaseModel
from tqdm import tqdm

from .exceptions import UnsupportedFileError, VisionParserError
from .llm import LLM
from .utils import get_device_config

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class PDFPageConfig(BaseModel):
    """Configures settings for PDF page conversion.

    Attributes:
        dpi (int): Resolution for PDF to image conversion. Defaults to 400.
        color_space (str): Color mode for image output. Defaults to "RGB".
        include_annotations (bool): Includes PDF annotations in conversion. Defaults to True.
        preserve_transparency (bool): Preserves alpha channel in output. Defaults to False.
    """

    dpi: int = 400
    color_space: str = "RGB"
    include_annotations: bool = True
    preserve_transparency: bool = False


class VisionParser:
    def __init__(
        self,
        page_config: Optional[PDFPageConfig] = None,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        ollama_config: Optional[Dict] = None,
        openai_config: Optional[Dict] = None,
        gemini_config: Optional[Dict] = None,
        image_mode: Literal["url", "base64", None] = None,
        custom_prompt: Optional[str] = None,
        detailed_extraction: bool = False,
        enable_concurrency: bool = False,
        **kwargs: Any,
    ):
        """Initializes the parser with PDF page configuration and LLM settings.

        Args:
            page_config (Optional[PDFPageConfig]): Configuration for PDF page processing.
            model_name (str): Name of the LLM model to use. Defaults to "gpt-4o".
            api_key (Optional[str]): API key for the LLM provider.
            temperature (float): Controls randomness in LLM output. Defaults to 0.7.
            top_p (float): Controls diversity in LLM output. Defaults to 0.7.
            ollama_config (Optional[Dict]): Configuration for Ollama provider.
            openai_config (Optional[Dict]): Configuration for OpenAI provider.
            gemini_config (Optional[Dict]): Configuration for Google AI Studio provider.
            image_mode (Literal["url", "base64", None]): Mode for handling embedded images.
            custom_prompt (Optional[str]): Custom prompt for LLM processing.
            detailed_extraction (bool): Enables detailed text extraction. Defaults to False.
            enable_concurrency (bool): Enables concurrent page processing. Defaults to False.
            **kwargs: Additional keyword arguments for LLM configuration.
        """

        self.page_config = page_config or PDFPageConfig()
        self.device, self.num_workers = get_device_config()
        self.enable_concurrency = enable_concurrency

        self.llm = LLM(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            ollama_config=ollama_config,
            openai_config=openai_config,
            gemini_config=gemini_config,
            image_mode=image_mode,
            detailed_extraction=detailed_extraction,
            custom_prompt=custom_prompt,
            enable_concurrency=enable_concurrency,
            device=self.device,
            num_workers=self.num_workers,
            **kwargs,
        )

    def _calculate_matrix(self, page: fitz.Page) -> fitz.Matrix:
        """Calculates the transformation matrix for page conversion.

        Args:
            page (fitz.Page): The PDF page to process.

        Returns:
            fitz.Matrix: The calculated transformation matrix.
        """

        # Calculate zoom factor based on target DPI
        zoom = self.page_config.dpi / 72
        matrix = fitz.Matrix(zoom * 2, zoom * 2)

        # Handle page rotation if present
        if page.rotation != 0:
            matrix.prerotate(page.rotation)

        return matrix

    async def _convert_page(self, page: fitz.Page, page_number: int) -> str:
        """Converts a single PDF page into markdown formatted text.

        Args:
            page (fitz.Page): The PDF page to convert.
            page_number (int): The page number being processed.

        Returns:
            str: The markdown formatted text extracted from the page.

        Raises:
            VisionParserError: If page conversion fails.
        """

        try:
            matrix = self._calculate_matrix(page)

            # Create high-quality image from PDF page
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=self.page_config.preserve_transparency,
                colorspace=self.page_config.color_space,
                annots=self.page_config.include_annotations,
            )

            # Convert image to base64 for LLM processing
            base64_encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            return await self.llm.generate_markdown(base64_encoded, pix, page_number)

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert page {page_number + 1} to base64-encoded PNG: {str(e)}"
            )
        finally:
            # Clean up pixmap to free memory
            if pix is not None:
                pix = None

    async def _convert_pages_batch(self, pages: List[fitz.Page], start_idx: int):
        """Processes a batch of PDF pages concurrently.

        Args:
            pages (List[fitz.Page]): List of PDF pages to process.
            start_idx (int): Starting index for page numbering.

        Returns:
            List[str]: List of markdown formatted texts from the processed pages.
        """

        try:
            tasks = []
            for i, page in enumerate(pages):
                tasks.append(self._convert_page(page, start_idx + i))
            return await asyncio.gather(*tasks)
        finally:
            await asyncio.sleep(0.5)

    def convert_pdf(self, pdf_path: Union[str, Path]) -> List[str]:
        """Converts all pages in a PDF file to markdown text.

        Args:
            pdf_path (Union[str, Path]): Path to the PDF file to convert.

        Returns:
            List[str]: List of markdown formatted texts for each page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            UnsupportedFileError: If the file is not a PDF.
            VisionParserError: If PDF conversion fails.
        """

        pdf_path = Path(pdf_path)
        converted_pages = []

        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"File is not a PDF: {pdf_path}")

        try:
            with fitz.open(pdf_path) as pdf_document:
                total_pages = pdf_document.page_count

                with tqdm(
                    total=total_pages,
                    desc="Converting pages in PDF file into markdown format",
                ) as pbar:
                    if self.enable_concurrency:
                        # Process pages in batches based on num_workers
                        for i in range(0, total_pages, self.num_workers):
                            batch_size = min(self.num_workers, total_pages - i)
                            # Extract only required pages for the batch
                            batch_pages = [
                                pdf_document[j] for j in range(i, i + batch_size)
                            ]
                            batch_results = asyncio.run(
                                self._convert_pages_batch(batch_pages, i)
                            )
                            converted_pages.extend(batch_results)
                            pbar.update(len(batch_results))
                    else:
                        for page_number in range(total_pages):
                            # For non-concurrent processing, still need to run async code
                            text = asyncio.run(
                                self._convert_page(
                                    pdf_document[page_number], page_number
                                )
                            )
                            converted_pages.append(text)
                            pbar.update(1)

                return converted_pages
        except Exception as e:
            raise VisionParserError(
                f"Failed to convert PDF file into markdown content: {str(e)}"
            )
