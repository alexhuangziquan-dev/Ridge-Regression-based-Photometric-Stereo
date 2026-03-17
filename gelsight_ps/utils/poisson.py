# # -*- coding: utf-8 -*-
# """
# 简单的法向积分（Frankot–Chellappa 频域法）
# integrate_normals_poisson(p, q):
#   输入 p = dz/dx, q = dz/dy（float32/64，任意尺度）
#   输出 z（不再进行整幅去均值；由上层选择外环去平面）
# """
# import numpy as np
#
# def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
#     H, W = p.shape
#     wx = np.fft.fftfreq(W) * 2.0 * np.pi
#     wy = np.fft.fftfreq(H) * 2.0 * np.pi
#     wx, wy = np.meshgrid(wx, wy)
#
#     P = np.fft.fft2(p.astype(np.float32))
#     Q = np.fft.fft2(q.astype(np.float32))
#     denom = wx**2 + wy**2
#     denom[0, 0] = 1.0  # 避免除零（DC）
#     Z = (-1j * wx * P - 1j * wy * Q) / denom
#     Z[0, 0] = 0.0      # 固定 DC（基线任意）
#     z = np.fft.ifft2(Z).real.astype(np.float32)
#     # 不再执行 z -= z.mean()，由上层用“外环去平面”统一基线
#
#     return z
#
# def integrate_normals_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
#     """外部统一接口"""
#     return frankot_chellappa(p, q)
# -*- coding: utf-8 -*-
# """
# 简单的法向积分（Frankot–Chellappa 频域法）
#
# integrate_normals_poisson(p, q):
#   输入 p = dz/dx, q = dz/dy（float32/64，任意尺度）
#   步骤：
#     1) 频域 Frankot–Chellappa 积分，得到 z_raw（只确定到一个常数）
#     2) 自动选择“图像最外圈 margin 像素”为零平面，
#        用其中位数作为偏置，执行 z = z_raw - median(z_raw[outer_ring])
#   输出：
#     - 已经对齐到统一零平面的 z（外环 ~0，内部为相对深度）
# """
# import numpy as np
#
# def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
#     H, W = p.shape
#     wx = np.fft.fftfreq(W) * 2.0 * np.pi
#     wy = np.fft.fftfreq(H) * 2.0 * np.pi
#     wx, wy = np.meshgrid(wx, wy)
#
#     P = np.fft.fft2(p.astype(np.float32))
#     Q = np.fft.fft2(q.astype(np.float32))
#
#     denom = wx**2 + wy**2
#     denom[0, 0] = 1.0  # 避免除零（DC）
#     Z = (-1j * wx * P - 1j * wy * Q) / denom
#     Z[0, 0] = 0.0      # 固定 DC（基线任意）
#
#     z = np.fft.ifft2(Z).real.astype(np.float32)
#     # 此时 z 只确定到一个常数偏置 C
#
#     return z
#
# def integrate_normals_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
#     """
#     统一接口：
#     - 先用 Frankot–Chellappa 做积分；
#     - 再以内外环（margin 像素宽）的中位数为 0 做零平面对齐。
#     """
#     z = frankot_chellappa(p, q)
#
#     H, W = z.shape
#     margin = 2  # 你提到的“最外环 2 像素”为零平面
#
#     # 对于太小的图（比如 H/W < 4），直接返回原值，避免 margin 过大
#     if H < 2 * margin or W < 2 * margin:
#         return z
#
#     # 构造最外圈 margin 像素的掩膜
#     outer = np.zeros_like(z, dtype=bool)
#     outer[:margin, :] = True        # 上边缘
#     outer[-margin:, :] = True       # 下边缘
#     outer[:, :margin] = True        # 左边缘
#     outer[:, -margin:] = True       # 右边缘
#
#     vals = z[outer]
#     # 过滤掉 NaN / inf，避免极端数值干扰
#     vals = vals[np.isfinite(vals)]
#
#     if vals.size > 0:
#         z0 = np.median(vals)
#         z = z - z0  # 统一零平面：outer ring 的中位数 → 0
#
#     return z


# -*- coding: utf-8 -*-
"""
简单的法向积分（Frankot–Chellappa 频域法）

本版本根据用户要求统一基准规则：
- 积分后不做均值去除
- 仅做外环 2 像素的中位数归零
- margin = 2（固定）
- 不做 ring-detrend
- 输出的 z 已经对齐到统一零平面
"""

import numpy as np


def frankot_chellappa(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Frankot–Chellappa 频域积分
    输入:
        p, q: dz/dx, dz/dy
    输出:
        z_raw: 仅确定到常数 C 的深度
    """
    H, W = p.shape

    wx = np.fft.fftfreq(W) * 2.0 * np.pi
    wy = np.fft.fftfreq(H) * 2.0 * np.pi
    wx, wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p.astype(np.float32))
    Q = np.fft.fft2(q.astype(np.float32))

    denom = wx**2 + wy**2
    denom[0, 0] = 1.0  # 避免除零（DC项）
    Z = (-1j * wx * P - 1j * wy * Q) / denom
    Z[0, 0] = 0.0      # DC 设为 0 → z_raw 仅差一个常数

    z = np.fft.ifft2(Z).real.astype(np.float32)
    return z


def integrate_normals_poisson(p: np.ndarray, q: np.ndarray,
                              margin: int = 2) -> np.ndarray:
    """
    统一积分接口：
    1) Frankot–Chellappa 积分得到 z_raw
    2) 外环 margin 像素作为零平面，取中位数归零

    参数
    ----
    p, q:
        法向的 p = nx/nz, q = ny/nz
    margin:
        外环厚度（默认 2 像素，已按用户指定固定）

    返回
    ----
    z : 对齐到统一零平面的深度
    """
    z = frankot_chellappa(p, q)

    H, W = z.shape
    if H < 2 * margin or W < 2 * margin:
        # 图像太小：直接返回
        return z

    # ---- 构造最外圈掩膜 ----
    outer = np.zeros_like(z, dtype=bool)
    outer[:margin, :] = True
    outer[-margin:, :] = True
    outer[:, :margin] = True
    outer[:, -margin:] = True

    vals = z[outer]
    vals = vals[np.isfinite(vals)]
    if vals.size > 0:
        z0 = np.median(vals)
        z = z - z0   # 外环中位数归零

    return z
