# -*- coding: utf-8 -*-
"""Frankot-Chellappa normal integration (legacy version).

This is an earlier implementation that uses simple mean removal instead of
the outer-ring baseline alignment introduced in the current ``poisson.py``.
Retained for reproducibility of results generated before commit 11050037.
"""

import numpy as np


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Integrates gradient fields (p, q) into a zero-mean depth map.

    Args:
        p: Gradient dz/dx of shape ``[H, W]``.
        q: Gradient dz/dy of shape ``[H, W]``.

    Returns:
        Depth map of shape ``[H, W]`` (float32, zero-mean).
    """
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0  # Avoid division by zero at DC.
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0

    z = np.fft.ifft2(Z).real.astype(np.float32)
    z -= z.mean()
    return z


def integrate_normals_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Unified integration interface (legacy, zero-mean baseline).

    Args:
        p: Gradient dz/dx of shape ``[H, W]``.
        q: Gradient dz/dy of shape ``[H, W]``.

    Returns:
        Zero-mean depth map of shape ``[H, W]`` (float32).
    """
    return frankot_chellappa(p.astype(np.float32), q.astype(np.float32))
