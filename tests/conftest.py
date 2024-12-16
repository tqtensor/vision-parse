import pytest
from pathlib import Path
import fitz
from vision_parse.parser import PDFPageConfig, VisionParser


@pytest.fixture
def pdf_path():
    return Path("tests/Texas-Holdem-Rules.pdf")


@pytest.fixture
def page_config():
    return PDFPageConfig(
        dpi=400,
        color_space="RGB",
        include_annotations=True,
        preserve_transparency=False,
    )


@pytest.fixture
def markdown_parser(page_config):
    return VisionParser(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        page_config=page_config,
    )


@pytest.fixture
def pdf_document(pdf_path):
    doc = fitz.open(pdf_path)
    yield doc
    doc.close()
