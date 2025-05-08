import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from vision_parse.llm import LLM, UnsupportedProviderError


@pytest.fixture
def sample_base64_image():
    return base64.b64encode(b"test_image").decode("utf-8")


@pytest.fixture
def mock_structured_response():
    return {
        "message": {
            "content": json.dumps(
                {
                    "text_detected": "Yes",
                    "tables_detected": "No",
                    "images_detected": "No",
                    "extracted_text": "Test content",
                    "confidence_score_text": 0.9,
                }
            )
        }
    }


@pytest.fixture
def mock_markdown_response():
    return {"message": {"content": "# Test Header\n\nThis is test content."}}


@pytest.fixture
def mock_pixmap():
    mock = MagicMock()
    mock.samples = b"\x00" * (200 * 200 * 3)
    mock.height = 200
    mock.width = 200
    mock.n = 3
    mock.tobytes.return_value = b"test_image_data"
    return mock


def test_unsupported_model():
    """Test error handling for unsupported models."""

    with pytest.raises(UnsupportedProviderError) as exc_info:
        LLM(
            model_name="unsupported-model",
            temperature=0.7,
            top_p=0.7,
            api_key=None,
            openai_config=None,
            gemini_config=None,
            image_mode=None,
            custom_prompt=None,
            detailed_extraction=False,
            enable_concurrency=False,
            device=None,
            num_workers=1,
        )
    assert "not from a supported provider" in str(exc_info.value)


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
async def test_openai_generate_markdown(
    mock_get_response, sample_base64_image, mock_pixmap
):
    """Test markdown generation using OpenAI."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    llm = LLM(
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        openai_config={"OPENAI_API_KEY": "test-key"},
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_get_response.called


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
async def test_azure_openai_generate_markdown(
    mock_get_response, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Azure OpenAI."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    llm = LLM(
        model_name="gpt-4o",
        api_key=None,
        temperature=0.7,
        top_p=0.7,
        openai_config={
            "AZURE_ENDPOINT_URL": "https://test.openai.azure.com/",
            "AZURE_DEPLOYMENT_NAME": "gpt-4o",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        },
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_get_response.called


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
async def test_gemini_generate_markdown(
    mock_get_response, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Gemini."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    llm = LLM(
        model_name="gemini-2.5-pro",
        api_key=None,
        temperature=0.7,
        top_p=0.7,
        openai_config=None,
        gemini_config={"GOOGLE_API_KEY": "test-key"},
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_get_response.called


@pytest.mark.asyncio
@patch("vision_parse.llm.LLM._get_response")
async def test_deepseek_generate_markdown(
    mock_get_response, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Deepseek."""

    # Mock the LLM's _get_response method
    mock_get_response.return_value = "# Test Header\n\nTest content"

    llm = LLM(
        model_name="deepseek-vision",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        openai_config=None,
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_get_response.called
