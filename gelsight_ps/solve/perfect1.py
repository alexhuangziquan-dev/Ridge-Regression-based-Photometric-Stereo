# gelsight_ps/solve/reconstruct.py — final version with new output structure + camera realtime mode
# -*- coding: utf-8 -*-
"""
重建（查表 -> 法向 -> 深度）

输出目录结构（本版本，仅原文件模式生效）：
    solve_out_dir/
        depth/
            {base}_depth_minus_ref.npy             # 仅保存按压-未按压后的相对深度矩阵
        normal/
            {base}_normal.npy                     # 保存按压图像积分前的单位法向矩阵
        vis/
            {base}_depth_minus_ref.png            # 相对深度变化图可视化
            {base}_normal.png                     # 法向可视化图
        ref/
            ref_depth.npy                         # 未按压参考面的深度矩阵
            ref_normal.npy                        # 未按压参考面的法向矩阵
        depth_minus_ref_max.json                  # 保存差值最大深度（逻辑不变）

新增摄像头实时模式：
    1.  配置开关控制，不保存任何文件
    2.  实时读取摄像头帧，执行相同解算逻辑
    3.  实时可视化相对深度结果，按'q'退出
    4.  彻底解决自定义io与内置io命名冲突问题
    5.  新增开关`camera_subtract_ref_plane`，控制实时模式是否减去0平面（未按压参考面）
    6.  实时模式预处理新增三步流程：边缘15%裁剪 → 正方形裁剪（短边对齐） → 指定尺寸缩放
    7.  实时模式前n帧（solve配置中`ref_frame_count`）生成零平面，前n帧可视化当前深度，n帧后可视化做差结果
    8.  保持单个窗口平滑过渡，前后帧预处理逻辑完全一致
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import cv2
import yaml

# 工程内 I/O - 导入路径修改：io → image_io（彻底解决命名冲突，核心修正）
from ..image_io.image_loader import (
    list_images, imread_color, imwrite, Preprocessor
)

# Poisson 积分
from ..utils.poisson import integrate_normals_poisson


# ---------------------------------------------------------
# 工具函数（无修改）
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
        nz = np.sqrt(1.0 - nx * nx - ny * ny)
        nz = np.clip(nz, 0.0, 1.0)
        n = np.stack([nx, ny, nz], axis=-1)
    else:
        n = Y

    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return (n / norm).astype(np.float32)


# ---------------------------------------------------------
# 残差掩膜（与原逻辑一致，无修改）
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
# 主流程（适配image_io目录，新增实时模式前n帧生成零平面）
# ---------------------------------------------------------
def run_reconstruction(cfg: Dict[str, Any]) -> str:
    cam = cfg["camera"]
    H = _as_int_scalar(cam["height"])
    W = _as_int_scalar(cam["width"])

    solve_cfg = cfg["solve"]
    out_cfg = cfg["output"]

    # ---------------------- 配置解析：新增前n帧参数（ref_frame_count） ----------------------
    input_mode = solve_cfg.get("input_mode", "image_dir")
    camera_id = _as_int_scalar(solve_cfg.get("camera_id", 0))
    # 新增：实时模式是否减去0平面（未按压参考面）开关，默认True（保持原有逻辑）
    camera_subtract_ref_plane = bool(solve_cfg.get("camera_subtract_ref_plane", True))
    # 新增：前n帧生成零平面，n从solve配置中读取，默认30帧，确保至少1帧
    ref_frame_count = _as_int_scalar(solve_cfg.get("ref_frame_count", 30))
    ref_frame_count = max(1, ref_frame_count)
    # -----------------------------------------------------------------------------

    out_dir = out_cfg["solve_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 子目录（仅原文件模式生效）
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

    # 掩膜
    contact_mask_enable = bool(solve_cfg.get("contact_mask_enable", False))
    cm_w_chroma = float(solve_cfg.get("cm_w_chroma", 0.7))
    cm_w_int = float(solve_cfg.get("cm_w_int", 0.3))
    cm_thr_rel = float(solve_cfg.get("cm_thr_rel", 0.25))
    cm_min_area = _as_int_scalar(solve_cfg.get("cm_min_area", 80))

    # 读取 LUT
    W_field, meta = _load_lut(solve_cfg["lut_yaml"])
    feat_names = [str(x) for x in meta.get("features", [])]
    out_dim = int(meta.get("out_dim", 3))

    # 未按压参考（仅目录模式生效，实时模式将被前n帧覆盖）
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

        # 计算未按压参考面的 n0, depth0（原逻辑不变）
        if ref_lin is not None:
            n0 = _lut_predict_normals(ref_lin, W_field, feat_names, out_dim)

            # 仅原文件模式保存参考数据
            np.save(os.path.join(ref_dir, "ref_normal.npy"), n0)

            # 深度计算
            p0 = n0[..., 0] / np.maximum(n0[..., 2], 0.1)
            q0 = n0[..., 1] / np.maximum(n0[..., 2], 0.1)
            ref_depth = integrate_normals_poisson(p0, q0)
            if depth_flip_sign:
                ref_depth = -ref_depth

            # 仅原文件模式保存参考深度
            np.save(os.path.join(ref_dir, "ref_depth.npy"), ref_depth)

    # ---------------------- 分支1：原文件批量处理模式（无修改） ----------------------
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

            # 法向
            n = _lut_predict_normals(bgr_lin, W_field, feat_names, out_dim)
            np.save(os.path.join(normal_dir, f"{base}_normal.npy"), n)

            # 深度
            p_grad = n[..., 0] / np.maximum(n[..., 2], 0.1)
            q_grad = n[..., 1] / np.maximum(n[..., 2], 0.1)

            if smooth_grad_ksize >= 3:
                k = smooth_grad_ksize // 2 * 2 + 1
                p_grad = cv2.GaussianBlur(p_grad, (k, k), 0)
                q_grad = cv2.GaussianBlur(q_grad, (k, k), 0)

            depth = integrate_normals_poisson(p_grad, q_grad)
            if depth_flip_sign:
                depth = -depth

            # 掩膜
            if contact_mask_enable and ref_lin is not None:
                mask = _residual_mask(bgr_lin, ref_lin,
                                      cm_w_chroma, cm_w_int,
                                      cm_thr_rel, cm_min_area)
                if mask is not None:
                    depth = depth * (mask.astype(np.float32) / 255.0)

            # 相对深度计算与保存
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

                # 记录最大深度差
                depth_diff_max_stats[f"{base}_depth_minus_ref"] = float(np.max(d_diff))

                # 可视化相对深度
                vis_map = cv2.normalize(d_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                imwrite(os.path.join(vis_dir, f"{base}_depth_minus_ref.png"), vis_map)

            # 法向可视化
            n_vis = ((n + 1) / 2 * 255).astype(np.uint8)
            imwrite(os.path.join(vis_dir, f"{base}_normal.png"), n_vis[:, :, ::-1])

            print(f"[solve] {idx}/{len(img_paths)} processed: {base}")

        # JSON 汇总
        out_json = os.path.join(out_dir, "depth_minus_ref_max.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(depth_diff_max_stats, f, ensure_ascii=False, indent=2)

        print(f"[ok] results saved in {out_dir}")
        return out_dir

    # ---------------------- 分支2：摄像头实时模式（前n帧生成零平面 + 平滑过渡可视化） ----------------------
    elif input_mode == "camera":
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头设备 {camera_id}，请检查设备是否存在或权限是否足够")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        # 窗口名称（保持固定，确保平滑过渡）
        if camera_subtract_ref_plane:
            window_name = "Gelsight Real-Time (Subtracted Ref Plane) (press 'q' to quit)"
        else:
            window_name = "Gelsight Real-Time (Raw Depth) (press 'q' to quit)"
        window_created = False

        # ---------------------- 实时模式零平面生成相关变量初始化 ----------------------
        frame_idx = 0  # 当前帧计数器
        ref_bgr_lin_list = []  # 前n帧bgr_lin收集列表
        ref_depth_list = []  # 前n帧depth收集列表
        ref_plane_generated = False  # 零平面是否已生成标记
        # -----------------------------------------------------------------------------

        print(f"[camera mode] 已打开摄像头 {camera_id}，实时解算中（按'q'退出）")
        print(f"[camera mode] 目标图像尺寸：{W}x{H}")
        print(f"[camera mode] 是否减去0平面：{camera_subtract_ref_plane}")
        print(f"[camera mode] 前{ref_frame_count}帧生成零平面，当前正在采集第1帧...")

        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    print("[warning] 摄像头帧读取失败，退出循环")
                    break

                try:
                    # ========== 步骤1：基础输入校验（前后帧完全一致，无修改） ==========
                    # 1. 强制转换为 uint8 格式，确保像素值范围基础正确
                    bgr = bgr.astype(np.uint8) if bgr is not None else np.zeros((H, W, 3), dtype=np.uint8)

                    # 2. 校验并补全为3通道BGR（与文件模式的imread_color一致）
                    if bgr.ndim == 2:  # 灰度帧转3通道BGR
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
                    elif bgr.ndim == 3 and bgr.shape[2] != 3:  # 非3通道帧（如4通道RGBA）转3通道BGR
                        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGBA2BGR)

                    # 3. 裁剪像素值到0-255范围，避免异常值干扰预处理
                    bgr = np.clip(bgr, 0, 255).astype(np.uint8)

                    # ========== 步骤2：实时模式专属三步预处理（前后帧完全一致，无修改） ==========
                    # 第一步：裁剪边缘15%（去除边缘无效区域和畸变）
                    h_original, w_original = bgr.shape[:2]
                    crop_margin_h = int(h_original * 0.15)
                    crop_margin_w = int(w_original * 0.15)
                    # 确保裁剪后尺寸为正，避免越界
                    crop_margin_h = max(1, crop_margin_h)
                    crop_margin_w = max(1, crop_margin_w)
                    bgr_cropped_edge = bgr[crop_margin_h:-crop_margin_h, crop_margin_w:-crop_margin_w, :]

                    # 第二步：裁剪为与短边边长一致的正方形（中心裁剪，保证区域完整性）
                    h_cropped, w_cropped = bgr_cropped_edge.shape[:2]
                    short_side = min(h_cropped, w_cropped)
                    # 计算中心裁剪坐标
                    h_start = (h_cropped - short_side) // 2
                    w_start = (w_cropped - short_side) // 2
                    bgr_cropped_square = bgr_cropped_edge[h_start:h_start+short_side, w_start:w_start+short_side, :]

                    # 第三步：缩放至指定尺寸（H×W，与目录模式目标尺寸一致）
                    bgr_resized = cv2.resize(bgr_cropped_square, (W, H), interpolation=cv2.INTER_LINEAR)

                    # ========== 步骤3：后续预处理与计算（前后帧完全一致，无修改） ==========
                    # 应用预处理器（与目录模式一致）
                    current_bgr_lin = pre.apply(bgr_resized).astype(np.float32)

                    # 法向推理
                    n = _lut_predict_normals(current_bgr_lin, W_field, feat_names, out_dim)

                    # 深度计算
                    p_grad = n[..., 0] / np.maximum(n[..., 2], 0.1)
                    q_grad = n[..., 1] / np.maximum(n[..., 2], 0.1)

                    if smooth_grad_ksize >= 3:
                        k = smooth_grad_ksize // 2 * 2 + 1
                        p_grad = cv2.GaussianBlur(p_grad, (k, k), 0)
                        q_grad = cv2.GaussianBlur(q_grad, (k, k), 0)

                    current_depth = integrate_normals_poisson(p_grad, q_grad)
                    if depth_flip_sign:
                        current_depth = -current_depth

                    # 掩膜应用（仅对当前深度处理，前后帧一致）
                    if contact_mask_enable and (ref_lin is not None or ref_plane_generated):
                        # 零平面生成前，掩膜暂不生效（无参考）；生成后，使用新参考
                        mask_ref_lin = ref_lin if ref_plane_generated else current_bgr_lin
                        mask = _residual_mask(current_bgr_lin, mask_ref_lin,
                                              cm_w_chroma, cm_w_int,
                                              cm_thr_rel, cm_min_area)
                        if mask is not None:
                            current_depth = current_depth * (mask.astype(np.float32) / 255.0)

                    # ========== 步骤4：前n帧收集数据，生成零平面 ==========
                    if not ref_plane_generated:
                        # 收集前n帧的bgr_lin和depth（用于求均值生成零平面）
                        ref_bgr_lin_list.append(current_bgr_lin)
                        ref_depth_list.append(current_depth)
                        frame_idx += 1

                        # 打印采集进度
                        if frame_idx % 5 == 0 or frame_idx == ref_frame_count:
                            print(f"[camera mode] 零平面采集进度：{frame_idx}/{ref_frame_count} 帧")

                        # 当采集帧数达到n时，计算均值生成零平面
                        if frame_idx >= ref_frame_count:
                            ref_lin = np.mean(np.array(ref_bgr_lin_list), axis=0).astype(np.float32)
                            ref_depth = np.mean(np.array(ref_depth_list), axis=0).astype(np.float32)
                            ref_plane_generated = True
                            print(f"[camera mode] 零平面生成完成！开始可视化做差结果")

                    # ========== 步骤5：可视化逻辑（前n帧显示当前深度，n帧后显示做差结果，平滑过渡） ==========
                    # 全局归一化参数（固定100:225，前后可视化一致）
                    global_max_depth = 100.0
                    global_pixel_max = 225.0
                    pixel_full_range = 255.0

                    if not ref_plane_generated:
                        # 前n帧：可视化当前深度（不做差）
                        vis_data = current_depth.astype(np.float32)
                    else:
                        # n帧后：可视化做差结果（按开关控制是否减零平面）
                        if camera_subtract_ref_plane:
                            d_diff = (current_depth - ref_depth).astype(np.float32)
                            # 掩膜应用（对做差结果处理，保持原有逻辑）
                            if contact_mask_enable and ref_lin is not None:
                                mask = _residual_mask(current_bgr_lin, ref_lin,
                                                      cm_w_chroma, cm_w_int,
                                                      cm_thr_rel, cm_min_area)
                                if mask is not None:
                                    d_diff = d_diff * (mask.astype(np.float32) / 255.0)
                            vis_data = d_diff
                        else:
                            vis_data = current_depth.astype(np.float32)

                    # ========== 全局归一化（前后可视化逻辑一致，保证平滑过渡） ==========
                    # 步骤1：边界裁剪
                    vis_data_clipped = np.clip(vis_data, 0.0, global_max_depth)

                    # 步骤2：固定比例线性归一化（100:225）
                    if global_max_depth > 0:
                        vis_map = (vis_data_clipped / global_max_depth) * global_pixel_max
                    else:
                        vis_map = np.zeros_like(vis_data_clipped)

                    # 步骤3：格式转换与边界保护
                    vis_map = vis_map.astype(np.uint8)
                    vis_map = np.clip(vis_map, 0, int(pixel_full_range))

                    # 步骤4：伪彩色映射（固定JET色图，保持视觉一致）
                    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

                    # ========== 窗口显示（单个窗口，平滑过渡） ==========
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        window_created = True
                    cv2.imshow(window_name, vis_map)

                except Exception as e:
                    print(f"[warning] 单帧解算失败：{str(e)}，跳过当前帧")
                    continue

                # ========== 按键检测（保持原有逻辑） ==========
                if window_created:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[camera mode] 用户主动退出")
                        break
                else:
                    cv2.waitKey(1)
                    continue

        finally:
            # 安全释放资源，避免解释器清理异常
            if window_created:
                cv2.destroyWindow(window_name)
            cap.release()
            cv2.waitKey(1)  # 清空OpenCV事件队列
            cv2.waitKey(1)  # 双重清空，确保无残留（Windows环境适配）

        return "Real-time mode execution completed successfully."

    # ---------------------- 无效模式处理 ----------------------
    else:
        raise ValueError(f"无效的输入模式 {input_mode}，支持：'image_dir' / 'camera'")