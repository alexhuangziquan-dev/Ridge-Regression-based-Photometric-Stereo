# -*- coding: utf-8 -*-
"""
简单的法向积分（Frankot–Chellappa 频域法）
integrate_normals_poisson(p, q):
  输入 p = dz/dx, q = dz/dy（float32/64，任意尺度）
  输出 z（去均值）
"""
import numpy as np

def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    denom = wx**2 + wy**2
    denom[0, 0] = 1.0  # 避免除零（DC）
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0

    z = np.fft.ifft2(Z).real.astype(np.float32)
    z -= z.mean()
    return z

def integrate_normals_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """外部统一接口"""
    return frankot_chellappa(p.astype(np.float32), q.astype(np.float32))
