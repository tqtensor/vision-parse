import pytest
import base64
import json
from unittest.mock import patch
from vision_parse.parser import VisionParserError, ImageAnalysis


def test_calculate_matrix(markdown_parser, pdf_document):
    """Test the matrix calculation for PDF page transformation."""
    page = pdf_document[0]
    # Calculate zoom matrix for PDF page rendering
    matrix = markdown_parser._calculate_matrix(page)
    # Expected zoom is based on DPI to points (72) ratio with 2x scaling
    expected_zoom = markdown_parser.page_config.dpi / 72 * 2
    assert matrix.a == expected_zoom
    assert matrix.d == expected_zoom


@patch("ollama.chat")
def test_structured_llm(mock_chat, markdown_parser):
    """Test structured LLM analysis of image content."""
    # Mock the expected JSON response from the LLM
    mock_response = {
        "message": {
            "content": json.dumps(
                {
                    "text_detected": "Yes",
                    "tables_detected": "Yes",
                    "images_detected": "No",
                    "extracted_text": "Sample text content",
                    "confidence_score_text": 0.85,
                }
            )
        }
    }
    mock_chat.return_value = mock_response

    # Create a sample base64 encoded image for testing
    sample_base64 = base64.b64encode(b"test_image").decode("utf-8")
    result = markdown_parser._structured_llm(sample_base64)

    # Verify the parsed ImageAnalysis object matches expected values
    assert isinstance(result, ImageAnalysis)
    assert result.text_detected == "Yes"
    assert result.tables_detected == "Yes"
    assert result.images_detected == "No"
    assert result.extracted_text == "Sample text content"
    assert result.confidence_score_text == 0.85


@patch("ollama.chat")
def test_vision_llm(mock_chat, markdown_parser):
    """Test Vision LLM conversion of image to markdown."""
    # Define expected markdown output
    expected_markdown = "# Test Header\n\nThis is a test markdown content."
    mock_response = {"message": {"content": expected_markdown}}
    mock_chat.return_value = mock_response

    # Create test image data and verify conversion
    sample_base64 = base64.b64encode(b"test_image").decode("utf-8")
    result = markdown_parser._vision_llm(sample_base64, "Test prompt")

    assert result == expected_markdown
    mock_chat.assert_called_once()


def test_convert_pdf_nonexistent_file(markdown_parser):
    """Test error handling for non-existent PDF files."""
    # Verify proper error handling for missing PDF files
    with pytest.raises(VisionParserError) as exc_info:
        markdown_parser.convert_pdf("non-existent.pdf")
    assert "PDF file not found" in str(exc_info.value)


def test_convert_pdf_integration(markdown_parser, pdf_path):
    """Integration test for PDF to markdown conversion."""
    # Test full PDF conversion pipeline
    converted_pages = markdown_parser.convert_pdf(pdf_path)
    assert isinstance(converted_pages, list)
    assert len(converted_pages) > 0
    # Verify that at least one page contains non-empty content
    assert any(
        isinstance(page, str) and len(page.strip()) > 0 for page in converted_pages
    )


@pytest.mark.parametrize(
    "error_input",
    [
        {"message": "Test error", "error_code": 1, "source": "test_source"},
        {"message": "Test error", "error_code": None, "source": None},
    ],
)
def test_markdown_parser_error(error_input):
    """Test error handling and error message formatting."""
    # Create error instance with test parameters
    error = VisionParserError(
        message=error_input["message"],
        error_code=error_input["error_code"],
        source=error_input["source"],
    )
    error_str = str(error)
    # Verify error message formatting for different input combinations
    assert error_input["message"] in error_str
    if error_input["source"]:
        assert f"Source: {error_input['source']}" in error_str
    if error_input["error_code"]:
        assert f"Error code: {error_input['error_code']}" in error_str


@patch("ollama.chat")
def test_convert_page(mock_chat, markdown_parser, pdf_document):
    """Test single page conversion with proper mocking of LLM responses."""
    # Mock the structured analysis response
    structured_response = {
        "message": {
            "content": json.dumps(
                {
                    "text_detected": "Yes",
                    "tables_detected": "No",
                    "images_detected": "No",
                    "extracted_text": "Test page content",
                    "confidence_score_text": 0.9,
                }
            )
        }
    }

    # Mock the markdown conversion response
    markdown_response = {
        "message": {"content": "# Test page content\n\nThis is formatted markdown."}
    }

    # Set up sequential responses for the two LLM calls
    mock_chat.side_effect = [structured_response, markdown_response]

    page = pdf_document[0]
    result = markdown_parser._convert_page(page, 0)

    # Verify successful page conversion
    assert isinstance(result, str)
    assert "Test page content" in result
    assert mock_chat.call_count == 2

    # Test the case when no text is detected in the page
    structured_response["message"]["content"] = json.dumps(
        {
            "text_detected": "No",
            "tables_detected": "No",
            "images_detected": "No",
            "extracted_text": "",
            "confidence_score_text": 0.0,
        }
    )

    mock_chat.side_effect = [structured_response]
    mock_chat.reset_mock()

    # Verify empty result when no text is detected
    result = markdown_parser._convert_page(page, 0)
    assert result == ""
    assert mock_chat.call_count == 1  # Only _structured_llm should be called
