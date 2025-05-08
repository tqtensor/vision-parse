import os
from pathlib import Path

import fitz
import pytest
from dotenv import load_dotenv

from vision_parse.parser import PDFPageConfig, VisionParser


# Fixture providing the path to the test PDF file
@pytest.fixture
def pdf_path():
    return Path("tests/Texas-Holdem-Rules.pdf")


# Fixture configuring PDF page rendering settings
@pytest.fixture
def page_config():
    return PDFPageConfig(
        dpi=400,
        color_space="RGB",
        include_annotations=True,
        preserve_transparency=False,
    )


# Fixture initializing the vision-based markdown parser with specific model settings
@pytest.fixture
def markdown_parser(page_config):
    return VisionParser(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        top_p=0.7,
        page_config=page_config,
    )


# Fixture handling PDF document lifecycle with proper cleanup
@pytest.fixture
def pdf_document(pdf_path):
    doc = fitz.open(pdf_path)
    yield doc
    doc.close()
