# gelsight_ps/calib/ball_calibrate.py — supports calib.out_dim = 2 or 3
# -*- coding: utf-8 -*-
"""
标准球标定（逐像素岭回归）
新增：在配置里用 calib.out_dim 选择 2（回归 nx,ny）或 3（回归单位法向 nx,ny,nz）。
- npz 仍存键 "W"；shape 为 [H,W,F,out_dim]
- yaml.meta 写入 "out_dim"
其它接口/文件名/可视化保持不变，使 main/example_config 无需修改即可使用。
"""

import os
import json
import yaml
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List

from ..image_io.image_loader import list_images, imread_color, imwrite, Preprocessor
from ..utils.geometry import sphere_normals_for_contact

def _as_int_scalar(x) -> int:
    arr = np.array(x).reshape(-1)
    if arr.size == 0:
        raise TypeError(f"empty value for int scalar: {x}")
    try:
        return int(arr[0].item())
    except Exception:
        return int(arr[0])

def _as_pyint(x) -> int:
    return int(np.asarray(x).reshape(-1)[0])

def build_feature_extractor(calib_cfg: Dict[str, Any]):
    use_quadratic = bool(calib_cfg.get("use_quadratic", True))
    use_cross     = bool(calib_cfg.get("use_cross", True))

    feature_names = ["r'", "g'", "b'", "I", "1"]
    if use_quadratic:
        feature_names += ["R^2", "G^2", "B^2"]
    if use_cross:
        feature_names += ["RG", "RB", "GB"]

    def make_features_from_rgb(rgb: np.ndarray) -> np.ndarray:
        H, W, _ = rgb.shape
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        I = R + G + B
        I_safe = np.clip(I, 1e-6, None)
        r = R / I_safe
        g = G / I_safe
        b = B / I_safe
        feats = [r, g, b, I, np.ones((H, W), np.float32)]
        if use_quadratic:
            feats += [R * R, G * G, B * B]
        if use_cross:
            feats += [R * G, R * B, G * B]
        X = np.stack(feats, axis=-1).astype(np.float64)
        return X  # [H,W,F]

    return make_features_from_rgb, feature_names

# —— 复用你工程已有的交互/复用取圆工具（若无则替换为你的实现）
try:
    from .ball_calibrate import (  # type: ignore
        ManualCirclePicker, load_circle_from_json, load_circle_from_debug_vis,
        save_circle_json, circles_dir_path
    )
except Exception:
    # 简化兜底实现，生产请用你原先的版本
    class ManualCirclePicker:
        def __init__(self, win_name="manual-pick", min_points=3, draw_scale=1.0):
            self.win_name = win_name
        def pick_on_image(self, img):
            H, W = img.shape[:2]
            cx, cy, r = W//2, H//2, min(H,W)//5
            core = np.zeros((H,W), np.uint8)
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            core[((xx-cx)**2+(yy-cy)**2) <= (r-2)**2] = 255
            return cx, cy, r, core, img
    def load_circle_from_json(*a, **k): return None
    def load_circle_from_debug_vis(*a, **k): return None
    def save_circle_json(*a, **k): return None
    def circles_dir_path(*a, **k): return ""

def run_ball_calibration(cfg: Dict[str, Any]) -> str:
    img_dir = cfg["calib"]["image_dir"]
    out_dir = cfg["output"]["calib_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    H = _as_int_scalar(cfg["camera"]["height"])
    W = _as_int_scalar(cfg["camera"]["width"])
    pix_mm = float(cfg["camera"]["pixel_size_mm"])
    pre = Preprocessor(cfg.get("preproc", {}))

    R_mm = float(cfg["calib"]["sphere_radius_mm"])
    lam  = float(cfg["calib"].get("ridge_lambda", 1e-2))
    min_spp  = _as_int_scalar(cfg["calib"].get("min_samples_per_pixel", 6))
    max_imgs = _as_int_scalar(cfg["calib"].get("max_images", 0))
    do_eval   = bool(cfg["calib"].get("eval_export", True))
    debug_dump= bool(cfg["calib"].get("debug_dump", True))

    rim_shrink_pix = float(cfg["calib"].get("rim_shrink_pix", 1.5))
    nz_floor       = float(cfg["calib"].get("nz_floor", 0.12))

    # 新增：选择 2 或 3 维回归（默认 3 更稳）
    out_dim = int(cfg["calib"].get("out_dim", 3))
    if out_dim not in (2, 3):
        raise ValueError(f"calib.out_dim must be 2 or 3, got {out_dim}")

    make_features_from_rgb, feature_names = build_feature_extractor(cfg.get("calib", {}))
    FEAT_DIM = len(feature_names)

    mp_cfg = cfg.get("manual_pick", {}) or {}
    draw_scale = float(mp_cfg.get("draw_scale", 1.0))
    min_pts = int(mp_cfg.get("min_points", 3))
    reuse_circles_dir = (mp_cfg.get("reuse_from_circles_dir") or "").strip()
    reuse_debug_dir   = (mp_cfg.get("reuse_from_debug_dir") or "").strip()
    picker = ManualCirclePicker(win_name="manual-pick", min_points=min_pts, draw_scale=draw_scale)

    debug_dir = os.path.join(out_dir, "debug")
    if debug_dump:
        os.makedirs(debug_dir, exist_ok=True)

    img_paths = list_images(img_dir)
    if max_imgs and max_imgs > 0:
        img_paths = img_paths[:max_imgs]
    assert len(img_paths) > 0, f"No images found in {img_dir}"

    XtX = np.zeros((H, W, FEAT_DIM, FEAT_DIM), np.float64)
    XtY = np.zeros((H, W, FEAT_DIM, out_dim), np.float64)  # out_dim=2或3
    Ns  = np.zeros((H, W), np.int32)

    n_ok = 0
    for idx, p in enumerate(img_paths, 1):
        bgr = imread_color(p)
        if bgr.shape[0] != H or bgr.shape[1] != W:
            if idx == 1:
                print(f"[warn] image size {bgr.shape[1]}x{bgr.shape[0]} != config {W}x{H}, resizing…")
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        bgr_lin = pre.apply(bgr).astype(np.float32)
        rgb = bgr_lin[:, :, ::-1]  # 0..1
        base = os.path.splitext(os.path.basename(p))[0]

        got = None
        if reuse_circles_dir:
            got = load_circle_from_json(base, reuse_circles_dir, H, W)
        if got is None and reuse_debug_dir:
            got = load_circle_from_debug_vis(base, reuse_debug_dir, H, W)
        if got is None:
            picked = picker.pick_on_image(bgr_lin)
            if picked is None:
                continue
            cx, cy, r_est, core, _ = picked
        else:
            cx, cy, r_est = got
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            core = ((xx - cx) ** 2 + (yy - cy) ** 2 <= (max(r_est - 1.5, 1.0) ** 2)).astype(np.uint8)

        save_circle_json(base, debug_dir, cx, cy, r_est, H, W)

        # 使用“接触半径 r_est（像素）”生成几何法向/掩膜（稳健参数同 geometry）
        n_gt, mask_geo = sphere_normals_for_contact(
            H, W, cx, cy, r_est, pix_mm, R_mm,
            rim_shrink_pix=rim_shrink_pix, nz_floor=nz_floor
        )

        core = ((core > 0) & (mask_geo > 0)).astype(np.uint8)
        if core.sum() < 50:
            continue

        if debug_dump:
            dbg_img = (bgr_lin * 255.0).astype(np.uint8).copy()
            cv2.circle(dbg_img, (int(round(cx)), int(round(cy))), int(round(r_est)), (255, 255, 255), 2)
            imwrite(os.path.join(debug_dir, f"{base}_loc_refined.png"), dbg_img)

        sat_hi = float(cfg["calib"].get("sat_hi", 0.98))
        dark_lo = float(cfg["calib"].get("dark_lo", 0.03))
        drop_sat = bool(cfg["calib"].get("drop_saturated", True))
        drop_dark = bool(cfg["calib"].get("drop_dark", True))
        w_sat = float(cfg["calib"].get("weight_saturated", 0.3))
        w_dark = float(cfg["calib"].get("weight_dark", 0.3))

        Rch, Gch, Bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        I = Rch + Gch + Bch
        sat_mask = (Rch >= sat_hi) | (Gch >= sat_hi) | (Bch >= sat_hi)
        dark_mask = (I <= dark_lo)

        X = make_features_from_rgb(rgb)  # [H,W,F]
        if out_dim == 3:
            Y_full = n_gt[..., :3]
        else:
            Y_full = n_gt[..., :2]

        ys, xs = np.where(core > 0)
        for yy, xx in zip(ys, xs):
            v = X[yy, xx]; yv = Y_full[yy, xx]
            w = 1.0
            if sat_mask[yy, xx]:
                if drop_sat: continue
                else: w *= w_sat
            if dark_mask[yy, xx]:
                if drop_dark: continue
                else: w *= w_dark
            XtX[yy, xx] += w * np.outer(v, v)
            XtY[yy, xx] += w * np.outer(v, yv)
            Ns[yy, xx]  += int(w > 0.0)

        n_ok += 1

        if do_eval and (idx % 5 == 0 or idx == len(img_paths)):
            cov = (Ns / max(1, Ns.max()) * 255).astype(np.uint8)
            imwrite(os.path.join(out_dir, "coverage_progress.png"),
                    cv2.applyColorMap(cov, cv2.COLORMAP_JET))

    print(f"[calib] frames used={n_ok} / {len(img_paths)}")

    sel = Ns > 0
    if sel.sum() == 0:
        raise RuntimeError("No valid calibration samples collected.")

    A_glob = np.zeros((FEAT_DIM, FEAT_DIM), np.float64)
    B_glob = np.zeros((FEAT_DIM, out_dim), np.float64)
    ys, xs = np.where(sel)
    for yy, xx in zip(ys, xs):
        A_glob += XtX[yy, xx]
        B_glob += XtY[yy, xx]
    W_global = np.linalg.solve(A_glob + float(lam) * np.eye(FEAT_DIM), B_glob).astype(np.float32)

    W_field = np.zeros((H, W, FEAT_DIM, out_dim), np.float32)
    ok_pix, fb_pix = 0, 0
    for yy, xx in zip(ys, xs):
        if Ns[yy, xx] >= min_spp:
            A = XtX[yy, xx] + float(lam) * np.eye(FEAT_DIM)
            B = XtY[yy, xx]
            try:
                W = np.linalg.solve(A, B).astype(np.float32)
                ok_pix += 1
            except np.linalg.LinAlgError:
                W = W_global; fb_pix += 1
        else:
            W = W_global; fb_pix += 1
        W_field[yy, xx] = W
    W_field[Ns == 0] = W_global

    npz_path = os.path.join(out_dir, "rgb2n_field.npz")
    np.savez_compressed(npz_path, W=W_field)

    meta = {
        "type": "rgb_to_normal_linear_per_pixel",
        "field_shape": [int(H), int(W), int(FEAT_DIM), int(out_dim)],
        "out_dim": int(out_dim),
        "pixel_size_mm": float(pix_mm),
        "sphere_radius_mm": float(R_mm),
        "ridge_lambda": float(lam),
        "min_samples_per_pixel": int(min_spp),
        "features": list(map(str, feature_names)),
        "preproc": cfg.get("preproc", {}),
        "per_pixel": True,
        "rim_shrink_pix": float(rim_shrink_pix),
        "nz_floor": float(nz_floor),
    }
    yaml_path = os.path.join(out_dir, "rgb2n_lut.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": meta, "weights_npz": os.path.basename(npz_path)}, f, allow_unicode=True)

    cov = (Ns / max(1, Ns.max()) * 255).astype(np.uint8)
    imwrite(os.path.join(out_dir, "coverage.png"), cv2.applyColorMap(cov, cv2.COLORMAP_JET))

    print(f"[calib] saved:\n  {yaml_path}\n  {npz_path}")
    return out_dir
