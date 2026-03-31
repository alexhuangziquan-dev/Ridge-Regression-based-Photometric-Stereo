"""Image and mask I/O utilities with preprocessing (dark/white/gamma/WB).

Provides helpers for loading, saving, and preprocessing GelSight sensor
images. The ``Preprocessor`` class encapsulates the full radiometric
correction pipeline: dark-field subtraction, white-field normalisation,
inverse-gamma linearisation, and white-balance gain.
"""

from typing import List, Tuple, Optional
import os
import cv2
import numpy as np


def list_images(
    root_dir: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
) -> List[str]:
    """Lists image file paths in a directory, sorted alphabetically.

    Args:
        root_dir: Directory to scan.
        exts: Tuple of accepted file extensions (case-insensitive).

    Returns:
        Sorted list of absolute image file paths.
    """
    files = []
    for fn in sorted(os.listdir(root_dir)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(root_dir, fn))
    return files


def imread_color(path: str) -> np.ndarray:
    """Reads a colour image in BGR uint8 format.

    Args:
        path: Path to the image file.

    Returns:
        BGR uint8 ndarray of shape ``[H, W, 3]``.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    """Writes an image to disk, creating parent directories as needed.

    Args:
        path: Destination file path.
        img: Image array (BGR uint8 or single-channel).

    Raises:
        RuntimeError: If OpenCV fails to write the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"Failed to write image: {path}")


def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Converts a BGR image to greyscale if necessary.

    Args:
        img_bgr: Input image, either ``[H, W, 3]`` BGR or ``[H, W]`` grey.

    Returns:
        Single-channel greyscale image of shape ``[H, W]``.
    """
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr


def percentile_threshold(img_gray: np.ndarray, p: float) -> float:
    """Returns the pixel intensity at the given percentile.

    Args:
        img_gray: Single-channel greyscale image.
        p: Percentile in ``[0, 100]``.

    Returns:
        Intensity value at percentile ``p``.
    """
    return float(np.percentile(img_gray.reshape(-1), p))


class Preprocessor:
    """Radiometric preprocessor: dark/white-field, gamma, and white balance.

    Attributes:
        dark: Dark-field reference image (float32 BGR), or ``None``.
        white: White-field reference image (float32 BGR), or ``None``.
        gamma_inv: Inverse gamma exponent for sRGB linearisation.
        wb_gain: Per-channel white-balance gain ``[B, G, R]``.
        eps: Small constant to prevent division by zero.
    """

    def __init__(self, cfg_preproc: dict):
        self.dark = None
        self.white = None
        self.gamma_inv = float(cfg_preproc.get("gamma_inv", 1.0))
        # White-balance gains in BGR channel order.
        self.wb_gain = np.array(
            cfg_preproc.get("wb_gain", [1.0, 1.0, 1.0]), dtype=np.float32
        ).reshape(1, 1, 3)
        self.eps = float(cfg_preproc.get("clip_eps", 1e-6))

        dpath = (cfg_preproc.get("dark_path") or "").strip()
        wpath = (cfg_preproc.get("white_path") or "").strip()
        if dpath:
            d = cv2.imread(dpath, cv2.IMREAD_COLOR)
            if d is None:
                raise FileNotFoundError(f"dark frame not found: {dpath}")
            self.dark = d.astype(np.float32)
        if wpath:
            w = cv2.imread(wpath, cv2.IMREAD_COLOR)
            if w is None:
                raise FileNotFoundError(f"white frame not found: {wpath}")
            self.white = w.astype(np.float32)

    def apply(self, img_bgr: np.ndarray) -> np.ndarray:
        """Applies the full radiometric correction pipeline.

        Steps:
            1. Dark/white-field normalisation (if both are provided),
               otherwise simple ``/255`` scaling.
            2. Inverse-gamma linearisation (sRGB approximate).
            3. Per-channel white-balance gain.

        Args:
            img_bgr: Raw BGR uint8 image of shape ``[H, W, 3]``.

        Returns:
            Linearised BGR float32 image in ``[0, 1]``.
        """
        I = img_bgr.astype(np.float32)
        if self.dark is not None and self.white is not None:
            I = (I - self.dark) / (self.white - self.dark + self.eps)
            I = np.clip(I, 0.0, 1.0)
        else:
            I = I / 255.0
        # Linearise (approximate sRGB -> linear).
        if self.gamma_inv and self.gamma_inv != 1.0:
            I = np.power(I, self.gamma_inv).astype(np.float32)
        # Apply per-channel white-balance gain.
        I = I * self.wb_gain
        I = np.clip(I, 0.0, 1.0)
        return I


def auto_contact_mask(
    img_bgr: np.ndarray, p_lo: float = 60, p_hi: float = 99.5
) -> np.ndarray:
    """Estimates a coarse contact mask using percentile thresholding.

    Args:
        img_bgr: BGR uint8 input image.
        p_lo: Lower percentile for the initial threshold.
        p_hi: Upper percentile for the secondary threshold.

    Returns:
        Binary uint8 mask (1 = contact, 0 = background).
    """
    gray = ensure_gray(img_bgr)
    t1 = percentile_threshold(gray, p_lo)
    t2 = percentile_threshold(gray, p_hi)
    _, m1 = cv2.threshold(gray, t1, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(gray, t2, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        m1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    mask = cv2.bitwise_or(mask, m2)
    return (mask > 0).astype(np.uint8)


def detect_circle_with_prior(
    gray: np.ndarray,
    r_pix: float,
    dp: float = 1.2,
    minDist: int = 30,
    p1: int = 80,
    p2: int = 20,
) -> Optional[Tuple[float, float, float]]:
    """Detects a circle using Hough transform with a radius prior.

    Args:
        gray: Single-channel greyscale image.
        r_pix: Expected circle radius in pixels.
        dp: Inverse accumulator resolution ratio.
        minDist: Minimum distance between detected circle centres.
        p1: Canny high threshold.
        p2: Accumulator threshold for circle detection.

    Returns:
        ``(cx, cy, r)`` on success, or ``None`` if no circle is found.
    """
    gray8 = (
        gray
        if gray.dtype == np.uint8
        else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    )
    circles = cv2.HoughCircles(
        gray8,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=p1,
        param2=p2,
        minRadius=int(0.8 * r_pix),
        maxRadius=int(1.2 * r_pix),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0, 0]
    return float(cx), float(cy), float(r)
