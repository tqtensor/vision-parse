from unittest.mock import MagicMock, patch

import pytest

from vision_parse import (
    PDFPageConfig,
    UnsupportedFileError,
    VisionParser,
    VisionParserError,
)


def test_convert_pdf_nonexistent_file(markdown_parser):
    """Test error handling for non-existent PDF files."""

    with pytest.raises(FileNotFoundError) as exc_info:
        markdown_parser.convert_pdf("non-existent.pdf")
    assert "PDF file not found" in str(exc_info.value)


def test_convert_pdf_invalid_file(markdown_parser, tmp_path):
    """Test error handling for invalid file types."""

    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test content")

    with pytest.raises(UnsupportedFileError) as exc_info:
        markdown_parser.convert_pdf(invalid_file)
    assert "File is not a PDF" in str(exc_info.value)


def test_calculate_matrix(markdown_parser, pdf_document):
    """Test the matrix calculation for PDF page transformation."""

    page = pdf_document[0]
    matrix = markdown_parser._calculate_matrix(page)
    expected_zoom = markdown_parser.page_config.dpi / 72 * 2
    assert matrix.a == expected_zoom
    assert matrix.d == expected_zoom


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
async def test_convert_pdf_integration(mock_get_response, markdown_parser, pdf_path):
    """Test PDF conversion with mocked LLM."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    # Mock the PDF document
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
        mock_pixmap.height = 200
        mock_pixmap.width = 200
        mock_pixmap.n = 3
        mock_pixmap.tobytes = MagicMock(return_value=b"test_image_data")
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        converted_pages = markdown_parser.convert_pdf(pdf_path)
        assert len(converted_pages) == 1
        assert "Test content" in converted_pages[0]
        assert mock_get_response.called


def test_convert_pdf_vision_parser_error(markdown_parser, monkeypatch, pdf_path):
    """Test VisionParserError handling in convert_pdf method."""

    def mock_convert_page(*args, **kwargs):
        raise Exception("Failed to process page")

    monkeypatch.setattr(markdown_parser, "_convert_page", mock_convert_page)

    with pytest.raises(VisionParserError) as exc_info:
        markdown_parser.convert_pdf(pdf_path)
    assert "Failed to process page" in str(exc_info.value)


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
@patch("tqdm.tqdm")
async def test_parser_with_base64_image_mode(mock_tqdm, mock_get_response, pdf_path):
    """Test parser with base64 image mode configuration."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = (
        "# Test Header\n\n![Image 1](data:image/png;base64,test_image)"
    )

    parser = VisionParser(
        page_config=PDFPageConfig(dpi=300),
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        image_mode="base64",
        detailed_extraction=True,
        enable_concurrency=True,
    )

    # Mock the PDF document
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
        mock_pixmap.height = 200
        mock_pixmap.width = 200
        mock_pixmap.n = 3
        mock_pixmap.tobytes = MagicMock(return_value=b"test_image_data")
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        # Mock tqdm
        mock_progress = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        converted_pages = parser.convert_pdf(pdf_path)
        assert isinstance(converted_pages, list)
        assert len(converted_pages) > 0
        # Check if any page contains base64 image data
        assert any("data:image/png;base64," in page for page in converted_pages if page)
        assert mock_get_response.called


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
@patch("tqdm.tqdm")
async def test_parser_with_concurrent_processing(
    mock_tqdm, mock_get_response, pdf_path
):
    """Test parser with concurrent processing enabled."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    parser = VisionParser(
        page_config=PDFPageConfig(dpi=300),
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        enable_concurrency=True,
    )

    # Mock the PDF document
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
        mock_pixmap.height = 200
        mock_pixmap.width = 200
        mock_pixmap.n = 3
        mock_pixmap.tobytes = MagicMock(return_value=b"test_image_data")
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.page_count = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        # Mock tqdm
        mock_progress = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        converted_pages = parser.convert_pdf(pdf_path)
        assert isinstance(converted_pages, list)
        assert len(converted_pages) > 0
        assert mock_get_response.called
        assert all("Test content" in page for page in converted_pages)


def test_parser_with_custom_page_config():
    """Test parser initialization with custom page configuration."""

    custom_config = PDFPageConfig(
        dpi=600,
        color_space="GRAY",
        include_annotations=False,
        preserve_transparency=True,
    )
    parser = VisionParser(
        page_config=custom_config,
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
    )
    assert parser.page_config.dpi == 600
    assert parser.page_config.color_space == "GRAY"
    assert not parser.page_config.include_annotations
    assert parser.page_config.preserve_transparency


def test_parser_with_provider_config():
    """Test parser initialization with provider-specific configuration."""

    provider_config = {
        "base_url": "https://api.openai.com/v1",
        "max_retries": 3,
        "timeout": 240.0,
    }
    parser = VisionParser(
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        provider_config=provider_config,
    )
    # Test that provider_config is correctly passed through to the LLM
    assert parser.llm.provider_config == provider_config
