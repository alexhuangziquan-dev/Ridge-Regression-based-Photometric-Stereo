"""
图像/掩膜加载与保存工具 + 预处理（暗/白/γ/白平衡）
Author: You
"""
from typing import List, Tuple, Optional
import os
import cv2
import numpy as np

def list_images(root_dir: str, exts=(".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    files = []
    for fn in sorted(os.listdir(root_dir)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(root_dir, fn))
    return files

def imread_color(path: str) -> np.ndarray:
    """读取彩色图像（BGR，uint8）"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"Failed to write image: {path}")

def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def percentile_threshold(img_gray: np.ndarray, p: float) -> float:
    return float(np.percentile(img_gray.reshape(-1), p))

# ---------- 预处理：暗/白场 + 线性化 + 白平衡 ----------
class Preprocessor:
    def __init__(self, cfg_preproc: dict):
        self.dark = None
        self.white = None
        self.gamma_inv = float(cfg_preproc.get("gamma_inv", 1.0))
        self.wb_gain = np.array(cfg_preproc.get("wb_gain", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1,1,3)  # BGR 顺序
        self.eps = float(cfg_preproc.get("clip_eps", 1e-6))

        dpath = (cfg_preproc.get("dark_path") or "").strip()
        wpath = (cfg_preproc.get("white_path") or "").strip()
        if dpath:
            d = cv2.imread(dpath, cv2.IMREAD_COLOR)
            if d is None:
                raise FileNotFoundError(f"dark frame not found: {dpath}")
            self.dark = d.astype(np.float32)
        if wpath:
            w = cv2.imread(wpath, cv2.IMREAD_COLOR)
            if w is None:
                raise FileNotFoundError(f"white frame not found: {wpath}")
            self.white = w.astype(np.float32)

    def apply(self, img_bgr: np.ndarray) -> np.ndarray:
        """返回 float32 的 BGR，范围 [0,1]，并做线性化/白平衡"""
        I = img_bgr.astype(np.float32)
        if self.dark is not None and self.white is not None:
            I = (I - self.dark) / (self.white - self.dark + self.eps)
            I = np.clip(I, 0.0, 1.0)
        else:
            I = I / 255.0
        # 线性化（sRGB -> 线性近似）
        if self.gamma_inv and self.gamma_inv != 1.0:
            I = np.power(I, self.gamma_inv).astype(np.float32)
        # 白平衡（BGR 增益）
        I = I * self.wb_gain
        I = np.clip(I, 0.0, 1.0)
        return I  # BGR, float32, [0,1]

# ---------- 掩膜 ----------
def auto_contact_mask(img_bgr: np.ndarray, p_lo=60, p_hi=99.5) -> np.ndarray:
    """
    基于分位数阈值的简易接触掩膜估计（先粗筛）
    """
    gray = ensure_gray(img_bgr)
    t1 = percentile_threshold(gray, p_lo)
    t2 = percentile_threshold(gray, p_hi)
    _, m1 = cv2.threshold(gray, t1, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(gray, t2, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(m1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
    mask = cv2.bitwise_or(mask, m2)
    return (mask > 0).astype(np.uint8)

def detect_circle_with_prior(gray: np.ndarray, r_pix: float,
                             dp=1.2, minDist=30, p1=80, p2=20) -> Optional[Tuple[float,float,float]]:
    """
    用半径先验做 HoughCircle 检测，失败返回 None
    """
    gray8 = gray if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    circles = cv2.HoughCircles(gray8, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=p1, param2=p2,
                               minRadius=int(0.8*r_pix), maxRadius=int(1.2*r_pix))
    if circles is None:
        return None
    cx, cy, r = circles[0,0]
    return float(cx), float(cy), float(r)
