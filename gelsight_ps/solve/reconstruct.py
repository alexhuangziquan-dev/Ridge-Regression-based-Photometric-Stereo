"""Depth reconstruction from photometric-stereo LUT.

Pipeline overview (per frame):
  1. Load a per-pixel weight field (``W_field``) from the calibration LUT.
  2. Build image features from the BGR input.
  3. Predict per-pixel surface normals via a linear feature projection.
  4. Integrate gradients with Poisson integration to obtain a depth map.

Two input modes are supported:

``image_dir`` – batch processing of image files.
    Outputs are written to sub-directories under ``output.solve_out_dir``::

        solve_out_dir/
        ├── depth/   – {base}_depth_minus_ref.npy   (press minus no-press depth)
        ├── normal/  – {base}_normal.npy             (unit normal arrays)
        ├── vis/     – {base}_depth_minus_ref.png, {base}_normal.png
        ├── ref/     – ref_depth.npy, ref_normal.npy
        └── depth_diff_max_stats.json

``camera`` – real-time camera mode (no files written).
    The first ``ref_frame_count`` frames are averaged to build a reference
    plane. Subsequent frames display the subtracted depth map in a live
    OpenCV window. Press ``q`` to quit.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import yaml

# Project I/O utilities.
from ..image_io.image_loader import (
    list_images, imread_color, imwrite, Preprocessor
)

# Poisson depth integration.
from ..utils.poisson import integrate_normals_poisson


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _as_int_scalar(x) -> int:
    """Converts an arbitrary scalar-like value to a Python int."""
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        raise TypeError(f"empty value for int scalar: {x}")
    try:
        return int(arr[0].item())
    except Exception:
        return int(arr[0])


def _build_features_from_rgb(rgb: np.ndarray,
                             feature_names: List[str]) -> np.ndarray:
    """Builds the feature map defined during calibration.

    Args:
        rgb: Float image of shape ``[H, W, 3]`` with channels in R, G, B order.
        feature_names: Ordered list of feature identifiers matching those used
            when fitting the LUT.

    Returns:
        Float64 array of shape ``[H, W, F]`` where ``F = len(feature_names)``.
    """
    H, W, _ = rgb.shape
    R = rgb[..., 0].astype(np.float64)
    G = rgb[..., 1].astype(np.float64)
    B = rgb[..., 2].astype(np.float64)

    I = R + G + B
    I_safe = np.clip(I, 1e-6, None)

    # Chrominance components
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
    """Loads the per-pixel weight field and metadata from a LUT YAML file.

    Args:
        lut_yaml: Path to the ``rgb2n_lut.yaml`` file produced by calibration.

    Returns:
        A tuple ``(W_field, meta)`` where ``W_field`` has shape
        ``[H, W, FEAT_DIM, out_dim]`` (float32) and ``meta`` is the raw
        metadata dict.
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
    """Runs per-pixel LUT inference and returns unit surface normals.

    Args:
        bgr_lin: Linearised BGR image of shape ``[H, W, 3]`` (float32).
        W_field: Per-pixel weight field of shape ``[H, W, F, out_dim]``.
        feature_names: Feature identifiers matching ``W_field``'s feature axis.
        out_dim: Output dimensionality (2 for ``(nx, ny)`` with implicit nz,
            3 for full ``(nx, ny, nz)``).

    Returns:
        Unit normal array of shape ``[H, W, 3]`` (float32).
    """
    rgb = bgr_lin[:, :, ::-1]  # BGR→RGB
    H, W, _ = rgb.shape

    X = _build_features_from_rgb(rgb, feature_names)  # [H,W,F]
    Y = np.einsum("hwf,hwfc->hwc", X, W_field).astype(np.float32)  # [H,W,out]

    if out_dim == 2:
        nx, ny = Y[..., 0], Y[..., 1]
        nz = np.sqrt(1.0 - nx * nx - ny * ny)
        nz = np.clip(nz, 0.0, 1.0)
        n = np.stack([nx, ny, nz], axis=-1)
    else:
        n = Y

    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return (n / norm).astype(np.float32)


# ---------------------------------------------------------
# Residual contact mask (image_dir mode only)
# ---------------------------------------------------------

def _residual_mask(bgr_lin: np.ndarray, ref_lin: np.ndarray,
                   w_chroma: float, w_int: float,
                   thr_rel: float, min_area: int) -> Optional[np.ndarray]:
    """Computes a binary contact mask from the chroma/intensity residual.

    Args:
        bgr_lin: Current frame linearised BGR image (float32).
        ref_lin: No-press reference linearised BGR image (float32).
        w_chroma: Weight for the chrominance residual term.
        w_int: Weight for the intensity residual term.
        thr_rel: Minimum relative threshold for the combined residual.
        min_area: Minimum contour area in pixels to accept as a contact region.

    Returns:
        Uint8 mask (255 = contact, 0 = background), or ``None`` if no valid
        contact region is found.
    """
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
# Contact mask for camera mode – dual-threshold with dilated connected components
# ---------------------------------------------------------

def _contact_mask_with_thresholds(d_diff: np.ndarray,
                                 d_thred_h: float,
                                 d_thred_l: float) -> np.ndarray:
    """Generates a contact mask for camera mode using dual thresholds.

    Regions below the low threshold are masked out. Only regions that
    overlap with the dilated high-threshold connected components are kept.

    Args:
        d_diff: Relative depth difference array.
        d_thred_h: High threshold (core contact region).
        d_thred_l: Low threshold (extended contact region).

    Returns:
        Binary mask of shape ``[H, W]`` (uint8, 255 = keep, 0 = discard).
    """
    # Clip negative values to avoid interference from invalid regions.
    d_diff_clipped = np.clip(d_diff, 0.0, np.max(d_diff))

    # Low-threshold binary map (initial valid region).
    bw_low = (d_diff_clipped > d_thred_l).astype(np.uint8) * 255
    # High-threshold binary map (core contact region).
    bw_high = (d_diff_clipped > d_thred_h).astype(np.uint8) * 255

    # Dilate the high-threshold region to capture the surrounding area.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw_high_dilated = cv2.dilate(bw_high, kernel, iterations=2)

    # Extract connected components of the dilated high-threshold region.
    cnts, _ = cv2.findContours(bw_high_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_mask = np.zeros_like(bw_high_dilated)
    for cnt in cnts:
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

    # Retain only low-threshold pixels that overlap with the connected components.
    final_mask = cv2.bitwise_and(bw_low, cnt_mask)

    # Morphological cleanup to remove noise and fill small holes.
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return final_mask


# ---------------------------------------------------------
# Main reconstruction pipeline
# ---------------------------------------------------------

def run_reconstruction(cfg: Dict[str, Any]) -> str:
    """Runs the depth-reconstruction pipeline in batch or real-time mode.

    Args:
        cfg: Top-level configuration dict. See ``configs/example_config.py``
            for the full schema.

    Returns:
        Output directory path (batch mode) or a status message (camera mode).

    Raises:
        RuntimeError: If the camera device cannot be opened.
        ValueError: If ``solve.input_mode`` is not recognised.
    """
    cam = cfg["camera"]
    H = _as_int_scalar(cam["height"])
    W = _as_int_scalar(cam["width"])

    solve_cfg = cfg["solve"]
    out_cfg = cfg["output"]

    # --- Parse solve configuration. ---
    input_mode = solve_cfg.get("input_mode", "image_dir")
    camera_id = _as_int_scalar(solve_cfg.get("camera_id", 0))
    # Whether to subtract the reference plane in camera mode.
    camera_subtract_ref_plane = bool(solve_cfg.get("camera_subtract_ref_plane", True))
    # Number of initial frames averaged to build the reference plane.
    ref_frame_count = _as_int_scalar(solve_cfg.get("ref_frame_count", 30))
    ref_frame_count = max(1, ref_frame_count)
    # Dual depth thresholds for the camera-mode contact mask.
    d_thred_h = float(solve_cfg.get("d_thred_h", 10.0))  # High threshold.
    d_thred_l = float(solve_cfg.get("d_thred_l", 2.0))   # Low threshold.

    out_dir = out_cfg["solve_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Output sub-directories (image_dir mode only).
    depth_dir = os.path.join(out_dir, "depth")
    normal_dir = os.path.join(out_dir, "normal")
    vis_dir = os.path.join(out_dir, "vis")
    ref_dir = os.path.join(out_dir, "ref")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    pre = Preprocessor(cfg.get("preproc", {}))

    depth_from_normals = bool(solve_cfg.get("depth_from_normals", True))
    smooth_grad_ksize = _as_int_scalar(solve_cfg.get("smooth_grad_ksize", 0))
    depth_flip_sign = bool(solve_cfg.get("depth_flip_sign", False))

    # Contact mask parameters.
    contact_mask_enable = bool(solve_cfg.get("contact_mask_enable", False))
    cm_w_chroma = float(solve_cfg.get("cm_w_chroma", 0.7))
    cm_w_int = float(solve_cfg.get("cm_w_int", 0.3))
    cm_thr_rel = float(solve_cfg.get("cm_thr_rel", 0.25))
    cm_min_area = _as_int_scalar(solve_cfg.get("cm_min_area", 80))

    # Load LUT weight field and metadata.
    W_field, meta = _load_lut(solve_cfg["lut_yaml"])
    feat_names = [str(x) for x in meta.get("features", [])]
    out_dim = int(meta.get("out_dim", 3))

    # No-press reference frame (overridden by live averaging in camera mode).
    ref_lin = None
    ref_depth = None
    if input_mode == "image_dir":
        ref_path = (solve_cfg.get("ref_nopress_path") or "").strip()
        ref_dir_imgs = (solve_cfg.get("ref_nopress_dir") or "").strip()

        if ref_dir_imgs:
            paths = list_images(ref_dir_imgs)
            acc = None
            for p in paths:
                bgr = imread_color(p)
                if bgr.shape[:2] != (H, W):
                    bgr = cv2.resize(bgr, (W, H))
                lin = pre.apply(bgr).astype(np.float32)
                acc = lin if acc is None else acc + lin
            ref_lin = acc / len(paths)
        elif ref_path:
            bgr = imread_color(ref_path)
            if bgr.shape[:2] != (H, W):
                bgr = cv2.resize(bgr, (W, H))
            ref_lin = pre.apply(bgr).astype(np.float32)

        # Compute reference normals and depth for image_dir mode.
        if ref_lin is not None:
            n0 = _lut_predict_normals(ref_lin, W_field, feat_names, out_dim)

            np.save(os.path.join(ref_dir, "ref_normal.npy"), n0)

            p0 = n0[..., 0] / np.maximum(n0[..., 2], 0.1)
            q0 = n0[..., 1] / np.maximum(n0[..., 2], 0.1)
            ref_depth = integrate_normals_poisson(p0, q0)
            if depth_flip_sign:
                ref_depth = -ref_depth

            np.save(os.path.join(ref_dir, "ref_depth.npy"), ref_depth)

    # ---------------------------------------------------------
    # Branch 1: image_dir batch mode
    # ---------------------------------------------------------
    if input_mode == "image_dir":
        img_dir = solve_cfg["image_dir"]
        img_paths = list_images(img_dir)
        depth_diff_max_stats = {}

        for idx, p in enumerate(img_paths, 1):
            bgr = imread_color(p)
            if bgr.shape[:2] != (H, W):
                bgr = cv2.resize(bgr, (W, H))
            bgr_lin = pre.apply(bgr).astype(np.float32)
            base = os.path.splitext(os.path.basename(p))[0]

            # Normal prediction.
            n = _lut_predict_normals(bgr_lin, W_field, feat_names, out_dim)
            np.save(os.path.join(normal_dir, f"{base}_normal.npy"), n)

            # Depth integration.
            p_grad = n[..., 0] / np.maximum(n[..., 2], 0.1)
            q_grad = n[..., 1] / np.maximum(n[..., 2], 0.1)

            if smooth_grad_ksize >= 3:
                k = smooth_grad_ksize // 2 * 2 + 1
                p_grad = cv2.GaussianBlur(p_grad, (k, k), 0)
                q_grad = cv2.GaussianBlur(q_grad, (k, k), 0)

            depth = integrate_normals_poisson(p_grad, q_grad)
            if depth_flip_sign:
                depth = -depth

            # Apply chroma/intensity residual contact mask.
            if contact_mask_enable and ref_lin is not None:
                mask = _residual_mask(bgr_lin, ref_lin,
                                      cm_w_chroma, cm_w_int,
                                      cm_thr_rel, cm_min_area)
                if mask is not None:
                    depth = depth * (mask.astype(np.float32) / 255.0)

            # Compute relative depth and save results.
            if ref_depth is not None:
                d_diff = (depth - ref_depth).astype(np.float32)

                if contact_mask_enable and ref_lin is not None:
                    mask = _residual_mask(bgr_lin, ref_lin,
                                          cm_w_chroma, cm_w_int,
                                          cm_thr_rel, cm_min_area)
                    if mask is not None:
                        d_diff = d_diff * (mask.astype(np.float32) / 255.0)

                np.save(os.path.join(depth_dir,
                                     f"{base}_depth_minus_ref.npy"), d_diff)

                # Record per-frame maximum depth difference.
                depth_diff_max_stats[f"{base}_depth_minus_ref"] = float(np.max(d_diff))

                # Colour-mapped depth visualisation (JET).
                vis_map = cv2.normalize(d_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                imwrite(os.path.join(vis_dir, f"{base}_depth_minus_ref.png"), vis_map)

            # Normal-map visualisation.
            n_vis = ((n + 1) / 2 * 255).astype(np.uint8)
            imwrite(os.path.join(vis_dir, f"{base}_normal.png"), n_vis[:, :, ::-1])

            print(f"[solve] {idx}/{len(img_paths)} processed: {base}")

        # Write per-frame depth statistics as JSON.
        out_json = os.path.join(out_dir, "depth_diff_max_stats.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(depth_diff_max_stats, f, ensure_ascii=False, indent=2)

        print(f"[ok] results saved in {out_dir}")
        return out_dir

    # ---------------------------------------------------------
    # Branch 2: camera real-time mode (reference plane from first n frames)
    # ---------------------------------------------------------
    elif input_mode == "camera":
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {camera_id}. Check that the device exists and permissions are correct.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        # Use a fixed window name for smooth visual transitions.
        if camera_subtract_ref_plane:
            window_name = "Gelsight Real-Time (Subtracted Ref Plane) (press 'q' to quit)"
        else:
            window_name = "Gelsight Real-Time (Raw Depth) (press 'q' to quit)"
        window_created = False

        # State for reference-plane construction from the first N frames.
        frame_idx = 0
        ref_bgr_lin_list = []
        ref_depth_list = []
        ref_plane_generated = False

        print(f"[camera mode] Opened camera {camera_id}, running real-time solve (press 'q' to quit)")
        print(f"[camera mode] Target resolution: {W}x{H}")
        print(f"[camera mode] Subtract reference plane: {camera_subtract_ref_plane}")
        print(f"[camera mode] Collecting first {ref_frame_count} frames for reference plane...")
        if contact_mask_enable:
            print(f"[camera mode] Contact mask enabled, thresholds: d_thred_h={d_thred_h}, d_thred_l={d_thred_l}")

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    print("[warning] Failed to read camera frame, exiting loop")
                    break

                try:
                    # Step 1: Input validation.
                    bgr = bgr.astype(np.uint8) if bgr is not None else np.zeros((H, W, 3), dtype=np.uint8)

                    # Ensure 3-channel BGR.
                    if bgr.ndim == 2:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                    elif bgr.ndim == 3 and bgr.shape[2] != 3:
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGBA2BGR)

                    bgr = np.clip(bgr, 0, 255).astype(np.uint8)

                    # Step 2: Three-stage spatial preprocessing.
                    # Stage 1: Crop 15% border to mitigate lens distortion.
                    h_original, w_original = bgr.shape[:2]
                    crop_margin_h = int(h_original * 0.15)
                    crop_margin_w = int(w_original * 0.15)
                    crop_margin_h = max(1, crop_margin_h)
                    crop_margin_w = max(1, crop_margin_w)
                    bgr_cropped_edge = bgr[crop_margin_h:-crop_margin_h, crop_margin_w:-crop_margin_w, :]

                    # Stage 2: Centre-crop to a square (short-side aligned).
                    h_cropped, w_cropped = bgr_cropped_edge.shape[:2]
                    short_side = min(h_cropped, w_cropped)
                    h_start = (h_cropped - short_side) // 2
                    w_start = (w_cropped - short_side) // 2
                    bgr_cropped_square = bgr_cropped_edge[h_start:h_start+short_side, w_start:w_start+short_side, :]

                    # Stage 3: Resize to the target resolution.
                    bgr_resized = cv2.resize(bgr_cropped_square, (W, H), interpolation=cv2.INTER_LINEAR)

                    # Step 3: Radiometric preprocessing and LUT inference.
                    current_bgr_lin = pre.apply(bgr_resized).astype(np.float32)

                    # Normal prediction.
                    n = _lut_predict_normals(current_bgr_lin, W_field, feat_names, out_dim)

                    # Depth integration.
                    p_grad = n[..., 0] / np.maximum(n[..., 2], 0.1)
                    q_grad = n[..., 1] / np.maximum(n[..., 2], 0.1)

                    if smooth_grad_ksize >= 3:
                        k = smooth_grad_ksize // 2 * 2 + 1
                        p_grad = cv2.GaussianBlur(p_grad, (k, k), 0)
                        q_grad = cv2.GaussianBlur(q_grad, (k, k), 0)

                    current_depth = integrate_normals_poisson(p_grad, q_grad)
                    if depth_flip_sign:
                        current_depth = -current_depth

                    # Step 4: Accumulate frames to build the reference plane.
                    if not ref_plane_generated:
                        ref_bgr_lin_list.append(current_bgr_lin)
                        ref_depth_list.append(current_depth)
                        frame_idx += 1

                        if frame_idx % 5 == 0 or frame_idx == ref_frame_count:
                            print(f"[camera mode] Reference plane progress: {frame_idx}/{ref_frame_count} frames")

                        if frame_idx >= ref_frame_count:
                            ref_lin = np.mean(np.array(ref_bgr_lin_list), axis=0).astype(np.float32)
                            ref_depth = np.mean(np.array(ref_depth_list), axis=0).astype(np.float32)
                            ref_plane_generated = True
                            print("[camera mode] Reference plane ready. Starting live visualisation.")

                    # Step 5: Visualisation.
                    # Fixed normalisation: depth range [0, 100] maps to pixel [0, 225].
                    global_max_depth = 100.0
                    global_pixel_max = 225.0
                    pixel_full_range = 255.0

                    if not ref_plane_generated:
                        vis_data = current_depth.astype(np.float32)
                    else:
                        if camera_subtract_ref_plane:
                            d_diff = (current_depth - ref_depth).astype(np.float32)
                            # Apply dual-threshold connected-component contact mask.
                            if contact_mask_enable and ref_lin is not None:
                                contact_mask = _contact_mask_with_thresholds(d_diff, d_thred_h, d_thred_l)
                                d_diff = d_diff * (contact_mask.astype(np.float32) / 255.0)
                            vis_data = d_diff
                        else:
                            vis_data = current_depth.astype(np.float32)

                    # Linear normalisation followed by JET colour map.
                    vis_data_clipped = np.clip(vis_data, 0.0, global_max_depth)

                    if global_max_depth > 0:
                        vis_map = (vis_data_clipped / global_max_depth) * global_pixel_max
                    else:
                        vis_map = np.zeros_like(vis_data_clipped)

                    vis_map = vis_map.astype(np.uint8)
                    vis_map = np.clip(vis_map, 0, int(pixel_full_range))
                    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

                    # Display in a persistent window for smooth transitions.
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        window_created = True
                    cv2.imshow(window_name, vis_map)

                except Exception as e:
                    print(f"[warning] Frame processing failed: {e}, skipping")
                    continue

                # Poll for quit key.
                if window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[camera mode] User requested exit")
                        break
                else:
                    cv2.waitKey(1)
                    continue

        finally:
            # Release resources. Double waitKey clears the OpenCV event queue on Windows.
            if window_created:
                cv2.destroyWindow(window_name)
            cap.release()
            cv2.waitKey(1)
            cv2.waitKey(1)

        return "Real-time mode execution completed successfully."

    # ---------------------------------------------------------
    # Unsupported mode
    # ---------------------------------------------------------
    else:
        raise ValueError(f"Invalid input_mode '{input_mode}'. Supported: 'image_dir', 'camera'.")
