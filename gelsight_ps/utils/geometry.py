# -*- coding: utf-8 -*-
"""
几何/法向/积分相关的工具
"""
import numpy as np
import cv2
from typing import Tuple

def sphere_normals_for_contact(
    H: int, W: int,
    cx: float, cy: float,
    r_pix: float,
    pixel_size_mm: float,
    sphere_radius_mm: float,
    rim_shrink_pix: float = 1.5,   # 接触边界内缩（像素），避开 nz≈0 的病态圈
    nz_floor: float = 0.12         # 计算 p,q 前的 nz 下限（稳健）
) -> Tuple[np.ndarray, np.ndarray]:
    """
    由球几何生成每像素“接触区内”的球面法向与几何掩膜（接触半径 a）
    约定：相机朝向 +z，接触法向 z>0。

    重要改变：
    - 掩膜半径用“接触半径 a”（随帧变化），而不是固定的球半径 R。
    - 在掩膜外，法向置零（不参与积分）。
    - 为避免边缘 nz≈0 导致 p,q 奇异，提供 rim_shrink_pix/nz_floor 两个稳健参数。
    """
    # 网格
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx_pix = (xx - cx).astype(np.float64)
    dy_pix = (yy - cy).astype(np.float64)

    # 接触半径（像素/毫米）
    a_pix = max(float(r_pix) - float(rim_shrink_pix), 0.0)
    a_mm  = a_pix * float(pixel_size_mm)

    # 坐标（毫米）
    dx = dx_pix * float(pixel_size_mm)
    dy = dy_pix * float(pixel_size_mm)
    rr = np.sqrt(dx*dx + dy*dy)

    # 掩膜：用接触半径 a（而不是 R）
    mask = (rr <= a_mm).astype(np.uint8)

    # 球面几何（只在接触内定义）
    R = float(sphere_radius_mm)
    z = np.zeros_like(rr, dtype=np.float64)
    inside = rr <= a_mm
    if inside.any():
        z_in = np.sqrt(np.maximum(0.0, R*R - rr[inside]*rr[inside]))
        z[inside] = z_in

    # 法向（单位向量）：(x,y,z)/R，仅在接触内赋值
    nx = np.zeros_like(rr, dtype=np.float64)
    ny = np.zeros_like(rr, dtype=np.float64)
    nz = np.zeros_like(rr, dtype=np.float64)
    if inside.any():
        nx[inside] = dx[inside] / R
        ny[inside] = dy[inside] / R
        nz_raw = z[inside] / R
        # 稳健：给 nz 下限，避免 p=-nx/nz, q=-ny/nz 时奇异放大
        nz[inside] = np.maximum(nz_raw, float(nz_floor))

    # 归一化（理论上已是单位向量，这里稳妥再归一）
    n = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n / norm

    return n, mask


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Frankot-Chellappa：把梯度场 (p,q) 积成 z"""
    H, W = p.shape
    wx = np.fft.fftfreq(W) * 2*np.pi
    wy = np.fft.fftfreq(H) * 2*np.pi
    wx, wy = np.meshgrid(wx, wy, indexing='xy')
    P = np.fft.fft2(p); Q = np.fft.fft2(q)
    denom = wx**2 + wy**2
    denom[0,0] = 1.0
    Z = (-1j*wx*P - 1j*wy*Q) / denom
    Z[0,0] = 0.0
    z = np.fft.ifft2(Z).real
    z = z - z.mean()
    return z
