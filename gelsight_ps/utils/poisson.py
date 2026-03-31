# -*- coding: utf-8 -*-
"""Frankot-Chellappa frequency-domain normal integration.

Provides ``integrate_normals_poisson``, the unified interface for recovering
a depth map from surface-normal gradient fields. After integration the depth
baseline is aligned by subtracting the median of the outermost pixel ring
(fixed 2-pixel margin), so that the border sits at approximately zero.
"""

import numpy as np


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Integrates gradient fields (p, q) into a depth map.

    Uses the closed-form frequency-domain solution. The result is determined
    only up to an additive constant (the DC component is set to zero).

    Args:
        p: Gradient dz/dx of shape ``[H, W]``.
        q: Gradient dz/dy of shape ``[H, W]``.

    Returns:
        Raw depth map of shape ``[H, W]`` (float32), undetermined up to a
        constant offset.
    """
    H, W = p.shape

    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p.astype(np.float32))
    Q = np.fft.fft2(q.astype(np.float32))

    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0  # Avoid division by zero at the DC term.
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0  # Fix DC to zero; depth is relative.

    z = np.fft.ifft2(Z).real.astype(np.float32)
    return z


def integrate_normals_poisson(
    p: np.ndarray, q: np.ndarray, margin: int = 2
) -> np.ndarray:
    """Integrates surface-normal gradients into a baseline-aligned depth map.

    Pipeline:
        1. Frankot-Chellappa frequency-domain integration.
        2. Outer-ring baseline alignment: the median depth of the outermost
           ``margin`` pixels is subtracted so that the border is approximately
           zero, and interior values represent relative depth.

    Args:
        p: Gradient field nx/nz of shape ``[H, W]``.
        q: Gradient field ny/nz of shape ``[H, W]``.
        margin: Width (in pixels) of the outer ring used as the zero-plane
            reference. Fixed at 2 by convention.

    Returns:
        Depth map of shape ``[H, W]`` (float32), aligned to the outer-ring
        zero plane.
    """
    z = frankot_chellappa(p, q)

    H, W = z.shape
    if H < 2 * margin or W < 2 * margin:
        # Image too small for outer-ring alignment; return as-is.
        return z

    # Build the outer-ring boolean mask.
    outer = np.zeros_like(z, dtype=bool)
    outer[:margin, :] = True
    outer[-margin:, :] = True
    outer[:, :margin] = True
    outer[:, -margin:] = True

    vals = z[outer]
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
        z = z - np.median(vals)  # Align outer ring to zero.

    return z
