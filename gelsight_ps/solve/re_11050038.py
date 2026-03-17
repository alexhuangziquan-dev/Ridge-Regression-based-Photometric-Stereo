# gelsight_ps/solve/reconstruct.py
# -*- coding: utf-8 -*-
"""
重建（查表 -> 法向 -> 深度）
升级点（均可配置开关）：
- align_ref_enable: 在做 depth-minus-nopress 前用 ECC 做小位移配准（平移），默认 False
- sat_fix_enable  : 对饱和像素的法向作邻域中位数替换并重归一，默认 False
- contact_mask_enable: 用参考图构造接触掩膜，只在掩膜内保留深度，默认 True
- 修正：warpAffine 不再使用 WARP_INVERSE_MAP（findTransformECC 返回的是 ref->curr）
- 继续输出 depth_minus_nopress_max.json
"""
import os, yaml, json
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List

from ..image_io.image_loader import list_images, imread_color, imwrite, Preprocessor

# —— Poisson 积分；若 utils 下无可用实现，回退到 FC
try:
    from ..utils.poisson import integrate_normals_poisson
except Exception:
    def integrate_normals_poisson(p, q):
        H, W = p.shape
        wx = np.fft.fftfreq(W) * 2.0 * np.pi
        wy = np.fft.fftfreq(H) * 2.0 * np.pi
        wx, wy = np.meshgrid(wx, wy)
        P = np.fft.fft2(p); Q = np.fft.fft2(q)
        denom = wx**2 + wy**2
        denom[0, 0] = 1.0
        Z = (-1j * wx * P - 1j * wy * Q) / denom
        Z[0, 0] = 0.0
        z = np.fft.ifft2(Z).real.astype(np.float32)
        z -= z.mean()
        return z


def _as_int_scalar(x) -> int:
    arr = np.array(x).reshape(-1)
    try:
        return int(arr[0].item())
    except Exception:
        return int(arr[0])


def load_lut(lut_yaml: str):
    with open(lut_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    meta = y["meta"]
    npz = os.path.join(os.path.dirname(lut_yaml), y["weights_npz"])
    W = np.load(npz)["W"]  # [H,W,F,2]
    return W, meta


def rgb_to_normal_xy(rgb_lin: np.ndarray, W_field: np.ndarray) -> np.ndarray:
    """rgb_lin: [H,W,3] (RGB 0..1),  W_field: [H,W,F,2]"""
    H, W, _ = rgb_lin.shape
    R, G, B = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    I = R + G + B
    I_safe = np.clip(I, 1e-6, None)
    r = R / I_safe; g = G / I_safe; b = B / I_safe
    X = np.stack([r, g, b, I, np.ones_like(I),
                  R*R, G*G, B*B, R*G, R*B, G*B], axis=-1).astype(np.float32)  # [H,W,F]
    nxy = (X[..., None, :] @ W_field).squeeze(-2)  # [H,W,2]
    nx, ny = nxy[..., 0], nxy[..., 1]
    nz_sq = np.maximum(1.0 - nx*nx - ny*ny, 1e-6)
    nz = np.sqrt(nz_sq)
    n = np.stack([nx, ny, nz], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    n = -n
    return n.astype(np.float32)  # [H,W,3]


# ---------- 残差掩膜（用于 contact_mask_enable） ----------
def _residual_mask(bgr_lin: np.ndarray, ref_lin: np.ndarray,
                   w_chroma: float, w_int: float,
                   thr_rel: float, min_area: int) -> Optional[np.ndarray]:
    rgb  = bgr_lin[:, :, ::-1].astype(np.float32)
    rgb0 = ref_lin[:, :, ::-1].astype(np.float32)
    I  = np.clip(rgb.sum(axis=-1, keepdims=True), 1e-6, None)
    I0 = np.clip(rgb0.sum(axis=-1, keepdims=True), 1e-6, None)
    chrom  = rgb / I; chrom0 = rgb0 / I0
    d_ch = np.linalg.norm(chrom - chrom0, axis=-1)  # 0..sqrt(3)
    d_I = np.abs(I - I0) / (I0 + 1e-6); d_I = d_I[..., 0]
    res = w_chroma * (d_ch / np.sqrt(3.0)) + w_int * np.clip(d_I, 0, 1)
    res = (res - res.min()) / (res.max() - res.min() + 1e-9)
    t = max(thr_rel, float(np.percentile(res, 70)) / 1.3)
    bw = (res >= t).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0: return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area: return None
    mask = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return (mask > 0).astype(np.uint8)


# ---------- 平移 ECC ----------
def _align_ref_to_curr_translation(ref_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """返回 2x3 warp 矩阵（把 ref 对齐到 curr）。失败返回单位矩阵。"""
    warp = np.array([[1., 0., 0.],
                     [0., 1., 0.]], dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            ref_gray.astype(np.float32), curr_gray.astype(np.float32),
            warp, cv2.MOTION_TRANSLATION,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-6),
            inputMask=None
        )
    except Exception:
        pass
    return warp


def run_reconstruction(cfg: Dict[str, Any]) -> str:
    in_dir  = cfg["solve"]["image_dir"]
    out_dir = cfg["output"]["solve_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    H = _as_int_scalar(cfg["camera"]["height"])
    W = _as_int_scalar(cfg["camera"]["width"])
    pre = Preprocessor(cfg.get("preproc", {}))

    # 开关与参数
    sat_fix_enable       = bool(cfg["solve"].get("sat_fix_enable", False))
    sat_hi               = float(cfg["solve"].get("sat_hi", 0.98))
    align_ref_enable     = bool(cfg["solve"].get("align_ref_enable", False))
    contact_mask_enable  = bool(cfg["solve"].get("contact_mask_enable", True))
    cm_w_chroma          = float(cfg["solve"].get("cm_w_chroma", 0.7))
    cm_w_int             = float(cfg["solve"].get("cm_w_int", 0.3))
    cm_thr_rel           = float(cfg["solve"].get("cm_thr_rel", 0.25))
    cm_min_area          = _as_int_scalar(cfg["solve"].get("cm_min_area", 80))

    # LUT
    W_field, meta = load_lut(cfg["solve"]["lut_yaml"])  # [H,W,F,2]
    assert W_field.shape[0] == H and W_field.shape[1] == W, "LUT H×W != config size"

    # 未按压参考：目录优先（均值）
    ref_dir  = (cfg["solve"].get("ref_nopress_dir") or "").strip()
    ref_path = (cfg["solve"].get("ref_nopress_path") or "").strip()
    ref_lin: Optional[np.ndarray] = None
    ref_depth: Optional[np.ndarray] = None

    if ref_dir:
        paths = list_images(ref_dir)
        if len(paths) == 0:
            raise FileNotFoundError(f"solve.ref_nopress_dir empty: {ref_dir}")
        acc = None
        for pth in paths:
            bgr = imread_color(pth)
            if bgr.shape[0] != H or bgr.shape[1] != W:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            lin = pre.apply(bgr).astype(np.float32)
            acc = lin if acc is None else (acc + lin)
        ref_lin = (acc / float(len(paths))).astype(np.float32)
    elif ref_path:
        bgr = imread_color(ref_path)
        if bgr.shape[0] != H or bgr.shape[1] != W:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        ref_lin = pre.apply(bgr).astype(np.float32)

    # 若提供参考，预先求其“基线深度”（一次）
    if ref_lin is not None and bool(cfg["solve"].get("depth_from_normals", True)):
        n0 = rgb_to_normal_xy(ref_lin[:, :, ::-1], W_field)  # BGR -> RGB
        p0 = n0[..., 0] / np.maximum(n0[..., 2], 1e-6)
        q0 = n0[..., 1] / np.maximum(n0[..., 2], 1e-6)
        ref_depth = integrate_normals_poisson(p0, q0)
        ref_depth -= ref_depth.mean()

    # 统计：每张 depth_minus_nopress 的最大相对深度
    depth_diff_max_stats: Dict[str, float] = {}

    # 逐图处理
    paths = list_images(in_dir)
    assert len(paths) > 0, f"No images in {in_dir}"
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        bgr = imread_color(p)
        if bgr.shape[0] != H or bgr.shape[1] != W:
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        lin = pre.apply(bgr)  # BGR [0,1]
        rgb = lin[:, :, ::-1]

        # —— 法向（查表）
        n = rgb_to_normal_xy(rgb, W_field)        # [H,W,3]

        # 饱和像素法向邻域修复（可关）
        if sat_fix_enable:
            lin_rgb = lin[:, :, ::-1]  # BGR->RGB
            sat = (lin_rgb[...,0] >= sat_hi) | (lin_rgb[...,1] >= sat_hi) | (lin_rgb[...,2] >= sat_hi)
            if np.any(sat):
                for c in (0, 1):  # nx, ny
                    chan = n[..., c]
                    med  = cv2.medianBlur(chan.astype(np.float32), 3)
                    chan[sat] = med[sat]
                    n[..., c] = chan
                n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)

        # —— 深度（相对）
        if bool(cfg["solve"].get("depth_from_normals", True)):
            p_grad = n[..., 0] / np.maximum(n[..., 2], 1e-6)
            q_grad = n[..., 1] / np.maximum(n[..., 2], 1e-6)
            depth = integrate_normals_poisson(p_grad, q_grad).astype(np.float32)
            depth -= depth.mean()

            # (可选) 接触掩膜：仅保留接触区深度
            if contact_mask_enable and (ref_lin is not None):
                mask = _residual_mask(lin, ref_lin, cm_w_chroma, cm_w_int, cm_thr_rel, cm_min_area)
                if mask is not None:
                    depth = depth * (mask.astype(np.float32))

            d_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
            imwrite(os.path.join(out_dir, f"{base}_depth.png"), d_vis)

            # —— 未按压差分（若提供参考）
            if ref_depth is not None:
                ref_depth_used = ref_depth
                if align_ref_enable:
                    # 把参考对齐到当前（只平移）
                    cur_gray = cv2.cvtColor((lin*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    ref_gray = cv2.cvtColor((ref_lin*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    warp = _align_ref_to_curr_translation(ref_gray, cur_gray)
                    ref_depth_used = cv2.warpAffine(
                        ref_depth.astype(np.float32), warp, (W, H),
                        flags=cv2.INTER_LINEAR,  # 注意：不要 WARP_INVERSE_MAP
                        borderMode=cv2.BORDER_REPLICATE
                    )

                d_diff = (depth - ref_depth_used).astype(np.float32)

                # 掩膜也应用到差分
                if contact_mask_enable and (ref_lin is not None):
                    mask = _residual_mask(lin, ref_lin, cm_w_chroma, cm_w_int, cm_thr_rel, cm_min_area)
                    if mask is not None:
                        d_diff = d_diff * (mask.astype(np.float32))

                diff_vis = cv2.normalize(d_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)
                imwrite(os.path.join(out_dir, f"{base}_depth_minus_nopress.png"), diff_vis)

                # 记录该图的最大相对深度
                depth_diff_max_stats[f"{base}_depth_minus_nopress"] = float(np.max(d_diff))

        # —— 法向可视
        n_vis = ((n + 1) / 2.0 * 255).astype(np.uint8)
        imwrite(os.path.join(out_dir, f"{base}_normal.png"), n_vis[..., ::-1])  # BGR 保存

    # 落盘 JSON 统计
    if len(depth_diff_max_stats) > 0:
        json_path = os.path.join(out_dir, "depth_minus_nopress_max.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(depth_diff_max_stats, f, ensure_ascii=False, indent=2)

    print(f"[ok] results saved to {out_dir}")
    return out_dir
