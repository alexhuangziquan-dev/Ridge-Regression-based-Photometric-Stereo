# gelsight_ps/solve/reconstruct.py — final version with new output structure
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import yaml

# Poisson 积分
from gelsight_ps_ballcalib_minimal.gelsight_ps.utils.poisson import integrate_normals_poisson
from gelsight_ps_ballcalib_minimal.gelsight_ps.image_io.image_loader import Preprocessor


# ---------------------------------------------------------
# 配置部分（所有参数直接在此处设置）
# ---------------------------------------------------------
CONFIG = {
    "camera": {
        "height": 512,  # 相机图像高度
        "width": 512,   # 相机图像宽度
        "fps": 10       # 相机帧率
    },
    "solve": {
        "lut_yaml": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\output/3dim-ref666/out_calib\rgb2n_lut.yaml",  # LUT配置文件路径
    },
    "output": {
        "solve_out_dir": "output/3dim-ref666/out_solve",  # 输出目录路径
    },
    "preproc": {},  # 可选的预处理配置，留空使用默认
}


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def _as_int_scalar(x) -> int:
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise TypeError(f"empty value for int scalar: {x}")
    try:
        return int(arr[0].item())
    except Exception:
        return int(arr[0])


def _build_features_from_rgb(rgb: np.ndarray,
                             feature_names: List[str]) -> np.ndarray:
    """
    构造标定阶段定义的特征
    """
    H, W, _ = rgb.shape
    R = rgb[..., 0].astype(np.float64)
    G = rgb[..., 1].astype(np.float64)
    B = rgb[..., 2].astype(np.float64)

    I = R + G + B
    I_safe = np.clip(I, 1e-6, None)

    # 色度
    r = R / I_safe
    g = G / I_safe
    b = B / I_safe

    feats: List[np.ndarray] = []
    for name in feature_names:
        n = str(name)
        if n in ("r'", "r_norm", "rprime"):
            feats.append(r)
        elif n in ("g'", "g_norm", "gprime"):
            feats.append(g)
        elif n in ("b'", "b_norm", "bprime"):
            feats.append(b)
        elif n in ("I", "intensity"):
            feats.append(I)
        elif n == "1":
            feats.append(np.ones((H, W), dtype=np.float64))
        elif n == "R^2":
            feats.append(R * R)
        elif n == "G^2":
            feats.append(G * G)
        elif n == "B^2":
            feats.append(B * B)
        elif n == "RG":
            feats.append(R * G)
        elif n == "RB":
            feats.append(R * B)
        elif n == "GB":
            feats.append(G * B)
        else:
            raise ValueError(f"Unknown feature name in LUT meta: {name}")

    X = np.stack(feats, axis=-1)
    return X


def _load_lut(lut_yaml: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    读取 LUT
    """
    with open(lut_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    meta = data["meta"]
    npz_name = data["weights_npz"]

    npz_path = os.path.join(os.path.dirname(lut_yaml), npz_name)
    npz = np.load(npz_path)
    W_field = npz["W"].astype(np.float32)
    return W_field, meta


def _lut_predict_normals(bgr_lin: np.ndarray,
                         W_field: np.ndarray,
                         feature_names: List[str],
                         out_dim: int) -> np.ndarray:
    """
    单帧 LUT 推理，输出单位法向
    """
    rgb = bgr_lin[:, :, ::-1]  # BGR→RGB
    H, W, _ = rgb.shape

    X = _build_features_from_rgb(rgb, feature_names)  # [H,W,F]
    Y = np.einsum("hwf,hwfc->hwc", X, W_field).astype(np.float32)  # [H,W,out]

    if out_dim == 2:
        nx, ny = Y[..., 0], Y[..., 1]
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        n = np.stack([nx, ny, nz], axis=-1)
    else:
        n = Y

    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return (n / norm).astype(np.float32)


# ---------------------------------------------------------
# 残差掩膜（与原逻辑一致）
# ---------------------------------------------------------
def _residual_mask(bgr_lin: np.ndarray, ref_lin: np.ndarray,
                   w_chroma: float, w_int: float,
                   thr_rel: float, min_area: int) -> Optional[np.ndarray]:

    rgb = bgr_lin[:, :, ::-1].astype(np.float32)
    rgb0 = ref_lin[:, :, ::-1].astype(np.float32)

    I = np.clip(rgb.sum(axis=-1, keepdims=True), 1e-6, None)
    I0 = np.clip(rgb0.sum(axis=-1, keepdims=True), 1e-6, None)

    chrom = rgb / I
    chrom0 = rgb0 / I0

    d_ch = np.linalg.norm(chrom - chrom0, axis=-1)
    d_I = np.abs(I[..., 0] - I0[..., 0]) / (I0[..., 0] + 1e-6)

    res = w_chroma * (d_ch / np.sqrt(3.0)) + w_int * np.clip(d_I, 0, 1)
    res = (res - res.min()) / (res.max() - res.min() + 1e-9)

    t_auto = float(np.percentile(res, 70)) / 1.3
    t = max(thr_rel, t_auto)

    bw = (res >= t).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < float(min_area):
        return None

    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask


# ---------------------------------------------------------
# 主流程
# ---------------------------------------------------------
def run_reconstruction(cfg: Dict[str, Any]) -> str:
    cam = cfg["camera"]
    H = _as_int_scalar(cam["height"])
    W = _as_int_scalar(cam["width"])

    solve_cfg = cfg["solve"]
    out_cfg = cfg["output"]

    lut_yaml = solve_cfg["lut_yaml"]
    pre = Preprocessor(cfg.get("preproc", {}))

    # 读取 LUT
    W_field, meta = _load_lut(lut_yaml)
    feat_names = [str(x) for x in meta.get("features", [])]
    out_dim = int(meta.get("out_dim", 3))

    # 设置相机帧数
    fps = cam["fps"]
    cap = cv2.VideoCapture(1)  # 使用默认相机
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 初始化基平面深度
    base_depth = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bgr = frame
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        bgr_lin = pre.apply(bgr).astype(np.float32)

        # 法向
        n = _lut_predict_normals(bgr_lin, W_field, feat_names, out_dim)

        # 深度
        p_grad = n[..., 0] / np.maximum(n[..., 2], 0.1)
        q_grad = n[..., 1] / np.maximum(n[..., 2], 0.1)

        depth = -integrate_normals_poisson(p_grad, q_grad)

        # 前50帧计算基平面
        if frame_count < 50:
            if base_depth is None:
                base_depth = depth
            else:
                base_depth += depth
            frame_count += 1
            base_depth /= frame_count  # 平均化

            # 显示前50帧的深度图
            vis_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)  # 使用热力图进行可视化

            # 添加帧编号
            frame_number_text = f"Base Plane Calculation (Frame {frame_count})"
            cv2.putText(vis_map, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 在同一窗口显示基平面计算的深度图
            cv2.imshow('Base Plane Calculation', vis_map)

        else:
            # 后续帧减去基平面深度
            depth_diff = (depth - base_depth).astype(np.float32)  # 计算当前帧深度与基平面深度的差异

            # 可视化深度差异
            vis_map = cv2.normalize(depth_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)  # 使用热力图进行可视化

            # 添加帧编号
            frame_number_text = f"Frame: {frame_count + 1}"
            cv2.putText(vis_map, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示深度图（更新同一窗口）
            cv2.imshow('Depth Map (After Base Plane)', vis_map)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[ok] Reconstructing complete.")
    return "Reconstruction done."


def solve_reconstruction(cfg: Dict[str, Any]) -> str:
    return run_reconstruction(cfg)


# 启动主程序
if __name__ == "__main__":
    solve_reconstruction(CONFIG)
