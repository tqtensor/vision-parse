import fitz
from pathlib import Path
from typing import Optional, List, Literal
import logging
import argparse
from tqdm import tqdm
import base64
from pydantic import BaseModel
import ollama
import sys
from jinja2 import Template


class PDFPageConfig(BaseModel):
    """Configuration settings for PDF page conversion."""

    dpi: int = 400
    color_space: str = "RGB"
    include_annotations: bool = True
    preserve_transparency: bool = False


class ImageAnalysis(BaseModel):
    """Model Schema for image analysis."""

    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class LoggerConfig:
    """Handles stream logging configuration."""

    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """Sets up a stream logger with specified configurations."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


class VisionParserError(Exception):
    """Custom exception for handling Markdown Parser errors."""

    def __init__(self, message: str, error_code: int = None, source: str = None):
        """Initialize the VisionParserError."""
        self.message = message
        self.error_code = error_code
        self.source = source
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message including source and code if available."""
        exception_lst = [self.message]
        if self.source:
            exception_lst.append(f"Source: {self.source}")
        if self.error_code:
            exception_lst.append(f"Error code: {self.error_code}")
        return " | ".join(exception_lst)


class VisionParser:
    """Convert PDF pages to base64-encoded images and then extract text from the images in markdown format."""

    def __init__(
        self,
        model_name: str = "llama3.2-vision:11b",
        temperature: float = 0.7,
        top_p: float = 0.7,
        page_config: Optional[PDFPageConfig] = None,
    ):
        """Initialize parser with configuration and logger."""
        self.page_config = page_config or PDFPageConfig()
        self.logger = LoggerConfig.setup_logger(__name__)

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        try:
            # For Python 3.8+ compatibility
            try:
                from importlib.resources import files  # Python 3.9+
            except ImportError:
                from importlib_resources import files  # Python 3.8

            prompt_text = files("vision_parse").joinpath("md_prompt.j2").read_text()
            self.md_prompt = Template(prompt_text)
        except Exception as e:
            self.logger.critical("Markdown prompt file for vision LLM not found")
            raise VisionParserError(
                message=f"Markdown prompt file for vision LLM not found: {str(e)}",
                error_code=2,
                source="__init__.py in VisionParser class",
            )

        try:
            self.image_analysis_prompt = (
                files("vision_parse").joinpath("img_analysis.prompt").read_text()
            )
        except Exception as e:
            self.logger.critical("Image Analysis prompt file not found")
            raise VisionParserError(
                message=f"Image Analysis prompt file not found: {str(e)}",
                error_code=2,
                source="__init__.py in VisionParser class",
            )

        try:
            ollama.show(self.model_name)
        except ollama.ResponseError as e:
            self.logger.error(e.error)
            if e.status_code == 404:
                ollama.pull(self.model_name)
        except Exception as e:
            self.logger.error(f"Model {self.model_name}: {str(e)}")
            raise VisionParserError(
                message=f"Model {self.model_name}: {str(e)}",
                source="__init__.py in VisionParser class",
            )

    def _vision_llm(self, base64_encoded: str, prompt: str) -> str:
        """Analyze scanned image through Vision LLM."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64_encoded],
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            )
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Vision LLM processing failed: {str(e)}")
            raise VisionParserError(
                message=f"Vision LLM processing failed: {str(e)}",
                source="_vision_llm in VisionParser class",
            )

    def _structured_llm(self, base64_encoded: str) -> str:
        """Generate structured data from image through Vision LLM."""
        try:
            response = ollama.chat(
                model=self.model_name,
                format=ImageAnalysis.model_json_schema(),
                messages=[
                    {
                        "role": "user",
                        "content": self.image_analysis_prompt,
                        "images": [base64_encoded],
                    }
                ],
                options={
                    "temperature": 0.0,
                    "top_p": 0.4,
                },
            )
            return ImageAnalysis.model_validate_json(response["message"]["content"])
        except Exception as e:
            self.logger.error(f"Vision LLM processing failed: {str(e)}")
            raise VisionParserError(
                message=f"Vision LLM processing failed: {str(e)}",
                source="_structured_llm in VisionParser class",
            )

    def _calculate_matrix(self, page: fitz.Page) -> fitz.Matrix:
        """Calculate transformation matrix for page conversion."""
        zoom = self.page_config.dpi / 72
        matrix = fitz.Matrix(zoom * 2, zoom * 2)

        if page.rotation != 0:
            matrix.prerotate(page.rotation)
            self.logger.info(f"Applied rotation of {page.rotation} degrees")

        return matrix

    def _convert_page(self, page: fitz.Page, page_number: int) -> str:
        """Convert a single PDF page to base64-encoded PNG and extract markdown formatted text."""
        try:
            matrix = self._calculate_matrix(page)
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=self.page_config.preserve_transparency,
                colorspace=self.page_config.color_space,
                annots=self.page_config.include_annotations,
            )

            base64_encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            analysis_text = self._structured_llm(base64_encoded)
            if analysis_text.text_detected.strip() == "No":
                return ""

            prompt = self.md_prompt.render(
                extracted_text=analysis_text.extracted_text,
                tables_detected=analysis_text.tables_detected,
                images_detected=analysis_text.images_detected,
                confidence_score_text=float(analysis_text.confidence_score_text),
            )
            text = self._vision_llm(base64_encoded, prompt)

            return text

        except Exception as e:
            self.logger.error(
                f"Text Extraction failed for page {page_number + 1}: {str(e)}"
            )
            raise VisionParserError(
                message=f"Text Extraction failed for page {page_number + 1}: {str(e)}",
                source="_convert_page in VisionParser class",
            )
        finally:
            pix = None

    def convert_pdf(self, pdf_path: str) -> List[str]:
        """Convert all the pages in the given PDF file to markdown text."""
        pdf_path = Path(pdf_path)
        converted_pages = []

        if not pdf_path.exists():
            self.logger.critical(f"PDF file not found: {pdf_path}")
            raise VisionParserError(
                message=f"PDF file not found: {pdf_path}",
                error_code=2,
                source="convert_pdf in VisionParser class",
            )

        try:
            with fitz.open(pdf_path) as pdf_document:
                total_pages = pdf_document.page_count
                self.logger.info(f"Starting conversion of PDF with {total_pages} pages")

                with tqdm(total=total_pages, desc="Converting pages") as pbar:
                    for page_number in range(total_pages):
                        text = self._convert_page(
                            pdf_document[page_number], page_number
                        )

                        converted_pages.append(text)
                        pbar.update(1)

                self.logger.info(f"Successfully converted {len(converted_pages)} pages")
                return converted_pages

        except Exception as e:
            self.logger.critical(f"Markdown Parser failed: {str(e)}")
            raise VisionParserError(
                message=f"Markdown Parser failed: {str(e)}",
                source="convert_pdf in VisionParser class",
            )


def main():
    parser = argparse.ArgumentParser(description="Convert PDF pages to markdown text")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file")
    args = parser.parse_args()

    try:
        converter = VisionParser()
        converted_pages = converter.convert_pdf(args.pdf_path)
        print(
            f"\nSuccessfully converted {len(converted_pages)} pages into markdown text"
        )

    except VisionParserError as e:
        print(f"Conversion failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
