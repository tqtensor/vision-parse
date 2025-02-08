from pathlib import Path
from typing import Optional, Union, List
import gc
import os
import pyperclip
import streamlit as st
from dotenv import load_dotenv

from vision_parse import VisionParser

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE_MB: int = 5
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024  # 5MB limit


def init_page_config() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(page_title="Vision Parse Demo", page_icon="🦩", layout="wide")


def apply_custom_styles() -> None:
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white !important;
            transition: all 0.3s ease;
            opacity: 1 !important;
        }
        .stButton>button:hover {
            color: #e0e0e0 !important;
            opacity: 0.9 !important;
        }
        .stButton>button:active {
            color: #e0e0e0 !important;
            transform: scale(0.98);
            opacity: 0.9 !important;
        }
        .stButton>button:focus {
            color: white !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }
        .output-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .uploadedFile {
            font-family: "Source Sans Pro", sans-serif;
        }
        .uploadedFile:before {
            content: "Limit 5MB per file • PDF";
            color: #787878;
        }
        .stMarkdown {
            background: transparent !important;
        }
        .element-container:has(> .stMarkdown) {
            background: transparent !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> iframe) {
            background: transparent !important;
        }
        .copy-button {
            position: absolute;
            right: 10px;
            top: 10px;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "results" not in st.session_state:
        st.session_state.results = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False


def process_pdf(pdf_file) -> Optional[Union[str, List[str]]]:
    """Process the uploaded PDF file using Vision Parse."""
    # Check file size
    file_size = len(pdf_file.getvalue())
    if file_size > MAX_FILE_SIZE_BYTES:
        st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB")
        return None

    temp_path = Path(f"temp_{os.getpid()}.pdf")
    try:
        # Save uploaded file temporarily
        temp_path.write_bytes(pdf_file.getvalue())

        # Process the PDF using Vision Parse
        vision_parser = VisionParser(
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            top_p=0.4,
            image_mode=None,
            detailed_extraction=False,
        )
        result = vision_parser.convert_pdf(temp_path)
        return result
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


def clear_output() -> None:
    """Clear all session state and temporary files."""
    # Reset session state
    st.session_state.results = None
    st.session_state.file_processed = False
    st.session_state.uploaded_file = None

    # Remove file uploader from session state
    if "file_uploader" in st.session_state:
        del st.session_state["file_uploader"]

    # Clean up temporary files
    for temp_file in Path(".").glob("temp_*.pdf"):
        if temp_file.exists():
            temp_file.unlink()

    gc.collect()


def copy_to_clipboard() -> None:
    """Copy content to clipboard and show confirmation toast."""
    if st.session_state.results:
        pyperclip.copy(st.session_state.results)
        st.toast("Copied to clipboard!", icon="✂️")


def format_results(results: Union[str, List[str]]) -> str:
    """Format the results by combining pages and removing page numbers."""
    if isinstance(results, list):
        return "\n\n".join(
            content.replace("Page " + str(i + 1), "").strip()
            for i, content in enumerate(results)
        )
    return results


def main() -> None:
    """Main application function."""
    init_page_config()
    apply_custom_styles()
    init_session_state()

    # Title and description
    st.title("Vision Parse - Demo")
    st.markdown("Upload a PDF file to extract detailed information using Vision Parse.")

    # File uploader
    st.session_state.uploaded_file = st.file_uploader(
        "Choose a PDF file (max 5MB limit)",
        type="pdf",
        disabled=st.session_state.is_processing,
        key="file_uploader",
    )

    # Process uploaded file
    if (
        st.session_state.uploaded_file is not None
        and not st.session_state.is_processing
        and not st.session_state.file_processed
    ):
        st.session_state.is_processing = True
        with st.spinner("Processing PDF..."):
            try:
                st.toast(
                    "Please wait while the current file is being processed...",
                    icon="🔄",
                )
                result = process_pdf(st.session_state.uploaded_file)
                if result:
                    st.session_state.results = result
                    st.session_state.file_processed = True
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                st.session_state.is_processing = False


    # Display results
    if st.session_state.results:
        st.markdown("### Markdown Output")

        # Format results
        if isinstance(st.session_state.results, list):
            st.session_state.results = format_results(st.session_state.results)

        # Create button columns
        col1, col2 = st.columns([6, 1])

        # Display results and buttons
        st.markdown(st.session_state.results)
        with col1:
            st.button(
                "Copy to Clipboard",
                on_click=copy_to_clipboard,
                type="secondary",
                key="copy_btn",
            )
        with col2:
            st.button("Clear", on_click=clear_output, type="primary", key="clear_btn")


if __name__ == "__main__":
    main()
