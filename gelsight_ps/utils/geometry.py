# -*- coding: utf-8 -*-
"""Geometry and surface-normal utilities for sphere-based calibration.

Provides functions for computing ground-truth surface normals from known
sphere geometry within the contact region, and for Frankot-Chellappa
frequency-domain gradient integration.
"""

import numpy as np
import cv2
from typing import Tuple


def sphere_normals_for_contact(
    H: int,
    W: int,
    cx: float,
    cy: float,
    r_pix: float,
    pixel_size_mm: float,
    sphere_radius_mm: float,
    rim_shrink_pix: float = 1.5,
    nz_floor: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-pixel sphere surface normals within the contact region.

    Given the detected contact circle (centre and pixel radius) and the known
    physical sphere radius, this function generates ground-truth unit normals
    for every pixel inside the contact disc. The camera is assumed to look
    along the +z axis, so valid contact normals have nz > 0.

    The contact mask uses the *contact radius a* (which varies per frame),
    not the fixed sphere radius R. Pixels outside the mask receive zero
    normals and are excluded from downstream regression.

    Args:
        H: Image height in pixels.
        W: Image width in pixels.
        cx: Circle centre x-coordinate in pixels.
        cy: Circle centre y-coordinate in pixels.
        r_pix: Detected contact radius in pixels.
        pixel_size_mm: Physical size of one pixel (mm/pixel).
        sphere_radius_mm: Known sphere radius in millimetres.
        rim_shrink_pix: Number of pixels to shrink inward from the contact
            boundary, avoiding the nz ~ 0 singularity at the rim.
        nz_floor: Minimum allowed nz value. Normals with a smaller z-component
            are clamped to this floor to prevent p = -nx/nz and q = -ny/nz
            from diverging.

    Returns:
        A tuple ``(normals, mask)`` where ``normals`` has shape ``[H, W, 3]``
        (unit vectors, zero outside the contact) and ``mask`` is a uint8
        binary array (1 = inside contact, 0 = outside).
    """
    # Pixel coordinate grids.
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx_pix = (xx - cx).astype(np.float64)
    dy_pix = (yy - cy).astype(np.float64)

    # Contact radius in pixels and millimetres.
    a_pix = max(float(r_pix) - float(rim_shrink_pix), 0.0)
    a_mm = a_pix * float(pixel_size_mm)

    # Physical coordinates in millimetres.
    dx = dx_pix * float(pixel_size_mm)
    dy = dy_pix * float(pixel_size_mm)
    rr = np.sqrt(dx * dx + dy * dy)

    # Contact mask (uses contact radius a, not sphere radius R).
    mask = (rr <= a_mm).astype(np.uint8)

    # Sphere surface geometry (defined only inside the contact disc).
    R = float(sphere_radius_mm)
    z = np.zeros_like(rr, dtype=np.float64)
    inside = rr <= a_mm
    if inside.any():
        z[inside] = np.sqrt(np.maximum(0.0, R * R - rr[inside] * rr[inside]))

    # Unit surface normals: (x, y, z) / R, assigned only inside the contact.
    nx = np.zeros_like(rr, dtype=np.float64)
    ny = np.zeros_like(rr, dtype=np.float64)
    nz = np.zeros_like(rr, dtype=np.float64)
    if inside.any():
        nx[inside] = dx[inside] / R
        ny[inside] = dy[inside] / R
        nz_raw = z[inside] / R
        # Clamp nz to the floor to prevent gradient singularities.
        nz[inside] = np.maximum(nz_raw, float(nz_floor))

    # Re-normalise to unit length (theoretically redundant; defensive).
    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n / norm

    return n, mask


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Integrates a gradient field (p, q) into a depth map via Frankot-Chellappa.

    Uses the frequency-domain closed-form solution to find the depth surface
    whose gradients best match the input field in the least-squares sense.

    Args:
        p: Gradient dz/dx of shape ``[H, W]``.
        q: Gradient dz/dy of shape ``[H, W]``.

    Returns:
        Depth map of shape ``[H, W]`` (zero-mean).
    """
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2 * np.pi
    wy = np.fft.fftfreq(H) * 2 * np.pi
    wx, wy = np.meshgrid(wx, wy, indexing='xy')
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0  # Avoid division by zero at DC.
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0
    z = np.fft.ifft2(Z).real
    z = z - z.mean()
    return z
