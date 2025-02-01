from pathlib import Path
import os
import gradio as gr
from dotenv import load_dotenv

from vision_parse import VisionParser

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE_MB: int = 5
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024  # 5MB limit

def process_pdf(pdf_file) -> str:
    """Process the uploaded PDF file using Vision Parse."""
    if pdf_file is None:
        return "Please upload a PDF file."
        
    # Check file size
    file_size = os.path.getsize(pdf_file.name)
    if file_size > MAX_FILE_SIZE_BYTES:
        return f"File size exceeds the limit of {MAX_FILE_SIZE_MB}MB"

    try:
        # Process the PDF using Vision Parse
        vision_parser = VisionParser(
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            top_p=0.4,
            image_mode=None,
            detailed_extraction=False,
        )
        result = vision_parser.convert_pdf(pdf_file.name)
        
        # Format results
        if isinstance(result, list):
            return "\n\n".join(
                content.replace("Page " + str(i + 1), "").strip()
                for i, content in enumerate(result)
            )
        return result
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def create_demo() -> gr.Interface:
    """Create and configure the Gradio interface."""
    
    # Define the interface
    demo = gr.Interface(
        fn=process_pdf,
        inputs=gr.File(
            label="Upload PDF (max 5MB)",
            file_types=[".pdf"],
            height=200,
        ),
        outputs=gr.Textbox(
            label="Extracted Text",
            lines=15,
            show_copy_button=True,
        ),
        title="Vision Parse - Demo",
        description="Upload a PDF file to extract detailed information using Vision Parse.",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=["Source Sans Pro", "ui-sans-serif", "system-ui", "sans-serif"],
        ),
        css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: 0 auto;
                padding: 20px;
            }
            #component-0 {
                max-width: 100%;
                border: 2px dashed #7c3aed;
                border-radius: 8px;
                background-color: #1e293b;
                transition: border-color 0.3s ease;
            }
            #component-0:hover {
                border-color: #4f46e5;
            }
            .upload-box {
                height: 200px !important;
            }
            .gr-box {
                border-radius: 12px !important;
                background-color: #1e293b !important;
            }
            .gr-button {
                border-radius: 8px !important;
            }
            .gr-button.primary {
                background: linear-gradient(to right, #4f46e5, #7c3aed) !important;
            }
            .gr-form {
                flex: 1;
                gap: 20px;
            }
            .gr-padded {
                padding: 20px !important;
            }
            .gr-text-input, .gr-text-output {
                background-color: #1e293b !important;
                border: 1px solid #4b5563 !important;
                border-radius: 8px !important;
            }
            .dark {
                color-scheme: dark;
            }
        """
    )
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_port=7860)
