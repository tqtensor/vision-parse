import base64
import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar, List, Literal, Tuple

import cv2
import fitz
import numpy as np

logger = logging.getLogger(__name__)


class ImageExtractionError(BaseException):
    """Raises an error when image extraction or processing fails.

    This exception is raised when there are issues during image detection,
    processing, or validation steps.
    """

    pass


@dataclass
class ImageData:
    """Represents extracted image data with its associated metadata.

    Attributes:
        image_url (str): URL path for extracted images.
        base64_encoded (str | None): Base64 string if image_mode is base64, None otherwise.
        _lock (ClassVar[Lock]): Lock for thread safety during image processing.
    """

    image_url: str
    base64_encoded: str | None
    _lock: ClassVar[Lock] = Lock()

    @staticmethod
    def _prepare_image_for_detection(image: np.ndarray) -> np.ndarray:
        """Processes image to highlight potential image regions.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Processed binary image highlighting potential image regions.

        Raises:
            ImageExtractionError: If image processing fails.
        """
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(grayscale, (5, 5), 0)
            binary = cv2.adaptiveThreshold(
                smooth,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        except Exception as e:
            raise ImageExtractionError(f"Image detection failed: {str(e)}")

    @staticmethod
    def _check_region_validity(
        region: np.ndarray, contour: np.ndarray, region_dims: tuple
    ) -> bool:
        """Determines if region contains a valid image based on statistical properties.

        Args:
            region (np.ndarray): Image region to validate.
            contour (np.ndarray): Contour of the region.
            region_dims (tuple): Dimensions (width, height) of the region.

        Returns:
            bool: True if region contains a valid image, False otherwise.

        Raises:
            ImageExtractionError: If image validation fails.
        """
        try:
            width, height = region_dims
            region_area = cv2.contourArea(contour) / (width * height)
            region_variance = cv2.Laplacian(
                cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()

            return (
                np.std(region) > 25
                and 20 < np.mean(region) < 235
                and region_area > 0.4
                and region_variance < 500
            )
        except Exception as e:
            raise ImageExtractionError(
                f"Image validity check for region failed: {str(e)}"
            )

    @classmethod
    def extract_images(
        cls,
        pix: fitz.Pixmap,
        image_mode: Literal["url", "base64", None],
        page_number: int,
        min_dimensions: tuple = (100, 100),
    ) -> List["ImageData"]:
        """Extracts images from a PDF page with specified processing mode.

        This method processes the page image to detect and extract embedded images,
        validates them, and saves them according to the specified image mode.

        Args:
            pix (fitz.Pixmap): PDF page pixmap containing the image data.
            image_mode (Literal["url", "base64", None]): Mode for image extraction.
            page_number (int): Page number for image naming.
            min_dimensions (tuple, optional): Minimum dimensions for valid images.
                Defaults to (100, 100).

        Returns:
            List[ImageData]: List of extracted images with their metadata.

        Raises:
            ImageExtractionError: If image extraction or processing fails.
        """
        with cls._lock:
            try:
                min_width, min_height = min_dimensions
                page_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                page_image = cv2.cvtColor(
                    page_array, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR
                )
                processed_image = cls._prepare_image_for_detection(page_image)

                contours, _ = cv2.findContours(
                    processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                idx = 1
                extracted_images = []
                img_height, img_width = page_image.shape[:2]

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    if (
                        w < min_width
                        or h < min_height
                        or w > img_width * 0.95
                        or h > img_height * 0.95
                    ):
                        continue

                    # Get the exact region from the original page_image
                    region = page_image[y : y + h, x : x + w].copy()

                    if cls._check_region_validity(region, contour, (w, h)):
                        # Encode image based on mode
                        if image_mode == "url":
                            image_url = f"image_{page_number+1}_{idx}.png"

                            if not cv2.imwrite(image_url, region):
                                continue

                            idx += 1

                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=None,
                                )
                            )
                        else:  # base64 mode
                            image_url = f"image_{page_number+1}_{idx}.png"

                            img_bytes = cv2.imencode(".png", region)[1].tobytes()
                            if not img_bytes:
                                continue
                            base64_encoded = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8', errors='ignore')}"
                            idx += 1

                            extracted_images.append(
                                ImageData(
                                    image_url=image_url,
                                    base64_encoded=base64_encoded,
                                )
                            )

                return extracted_images
            except Exception as e:
                raise ImageExtractionError(f"Image processing failed: {str(e)}")


def get_device_config() -> Tuple[Literal["cuda", "mps", "cpu"], int]:
    """Determines optimal device configuration for processing.

    This function checks available hardware (GPU, MPS, CPU) and returns
    the appropriate device type and number of worker processes.

    Returns:
        Tuple[Literal["cuda", "mps", "cpu"], int]: Device type and optimal number
            of worker processes.
    """
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if nvidia_smi.returncode == 0:
            return "cuda", min(len(nvidia_smi.stdout.strip().split("\n")) * 2, 8)
    except Exception:
        pass

    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps", 4

    return "cpu", max(2, (os.cpu_count() // 2))
