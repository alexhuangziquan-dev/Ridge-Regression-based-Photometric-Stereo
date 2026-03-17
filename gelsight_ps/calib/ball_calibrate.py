"""Per-pixel photometric stereo calibration using a steel ball pressed into the sensor.

This module implements the full calibration pipeline that regresses a per-pixel
lookup table (LUT) mapping RGB responses to surface normals. Multiple ball radii
can be used simultaneously by placing calibration images in subdirectories named
after their radius (e.g. ``1.0/``, ``2.0/``).

Typical usage:
    python -m gelsight_ps.main --config configs/example_config.py --mode calib

Directory layout for multi-radius calibration::

    calib_data/
    ├── 1.0/          # images captured with a 1 mm radius ball
    │   ├── 000000.png
    │   └── ...
    └── 2.0/          # images captured with a 2 mm radius ball
        ├── 000000.png
        └── ...

The regression uses a two-level strategy:
  1. Global ridge regression over all pixels.
  2. 8x8 block-level residual correction, Gaussian-blended into a per-pixel
     weight field ``W_field[H, W, FEAT_DIM, out_dim]``.

Output artifacts (written to ``output.calib_out_dir``):
  - ``rgb2n_lut.yaml``   – metadata and pointer to the weight file.
  - ``rgb2n_field.npz``  – compressed per-pixel weight field.
  - ``coverage.png``     – heatmap of sample counts per pixel.
  - ``debug/``           – per-frame circle overlays and JSON circle data.
  - ``geom_depth/``      – integrated depth maps for qualitative inspection.
"""

import os
import csv
import json
import yaml
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List

from ..image_io.image_loader import (
    list_images, imread_color, imwrite, Preprocessor
)
from ..utils.geometry import sphere_normals_for_contact


def _as_int_scalar(x) -> int:
    """Converts an arbitrary scalar-like value to a Python int.

    Args:
        x: Any value that can be reshaped into a 1-D numpy array.

    Returns:
        The first element as a Python int.

    Raises:
        TypeError: If ``x`` is empty after reshaping.
    """
    arr = np.array(x).reshape(-1)
    if arr.size == 0:
        raise TypeError(f"empty value for int scalar: {x}")
    try:
        return int(arr[0].item())
    except Exception:
        return int(arr[0])


def _as_pyint(x) -> int:
    """Converts the first element of an array-like value to a Python int."""
    return int(np.asarray(x).reshape(-1)[0])


def build_feature_extractor(calib_cfg: Dict[str, Any]):
    """Builds a feature extractor function based on calibration config flags.

    Constructs a closure that converts an RGB image into a feature map used
    as the design matrix for ridge regression. The base features are the
    normalized chrominance channels (r, g, b), total intensity (I), and a
    bias term (1). Optional quadratic and cross-product terms can be enabled.

    Args:
        calib_cfg: Calibration sub-dict from the config.  Reads the keys
            ``use_quadratic`` (default True) and ``use_cross`` (default True).

    Returns:
        A tuple ``(make_features_from_rgb, feature_names)`` where
        ``make_features_from_rgb`` is a callable ``(rgb: ndarray) -> ndarray``
        that maps an ``[H, W, 3]`` float32 image (values in [0, 1]) to an
        ``[H, W, F]`` float64 feature array, and ``feature_names`` is the
        corresponding list of string labels.
    """
    use_quadratic = bool(calib_cfg.get("use_quadratic", True))
    use_cross = bool(calib_cfg.get("use_cross", True))

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
        return np.stack(feats, axis=-1).astype(np.float64)

    return make_features_from_rgb, feature_names


def fit_circle_least_squares(pts_xy: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Fits a circle to 2-D points using algebraic least squares.

    Implements the Pratt-style algebraic fit reduced to a 2x2 linear system
    by centering the point cloud first.

    Args:
        pts_xy: Array of shape ``[N, 2]`` with (x, y) coordinates.

    Returns:
        A tuple ``(cx, cy, r)`` on success, or ``None`` if fewer than 3
        points are provided or the system is singular.
    """
    if pts_xy.ndim != 2 or pts_xy.shape[1] != 2 or pts_xy.shape[0] < 3:
        return None
    pts = pts_xy.astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    x_m, y_m = x.mean(), y.mean()
    u, v = x - x_m, y - y_m
    Suu = (u * u).sum()
    Svv = (v * v).sum()
    Suv = (u * v).sum()
    Suuu = (u * u * u).sum()
    Svvv = (v * v * v).sum()
    Suvv = (u * v * v).sum()
    Svuu = (v * u * u).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu], dtype=np.float64)
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx = x_m + uc
    cy = y_m + vc
    r = float(np.sqrt(max(uc * uc + vc * vc + (Suu + Svv) / len(pts), 0.0)))
    return float(cx), float(cy), r


def _detect_circle_from_vis(img_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Detects the largest circle in a debug visualization image.

    Tries Hough circle detection first; falls back to contour-based algebraic
    fitting if Hough returns no result.

    Args:
        img_bgr: BGR image, typically a previously saved debug overlay.

    Returns:
        ``(cx, cy, r)`` in pixel coordinates, or ``None`` if detection fails.
    """
    if img_bgr is None or img_bgr.ndim != 3:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    minR = max(5, min(h, w) // 20)
    maxR = max(min(h, w) // 2, minR + 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 8,
        param1=120,
        param2=30,
        minRadius=minR,
        maxRadius=maxR,
    )
    if circles is not None and len(circles) > 0:
        cs = circles[0]
        idx = int(np.argmax(cs[:, 2]))
        x, y, r = cs[idx]
        return float(x), float(y), float(r)

    edges = cv2.Canny(gray, 80, 180)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    return fit_circle_least_squares(pts)


def circles_dir_path(base_dir: str) -> str:
    """Returns the circles sub-directory path, creating it if necessary."""
    d = os.path.join(base_dir, "circles")
    os.makedirs(d, exist_ok=True)
    return d


def circle_json_path(base_name: str, base_dir: str) -> str:
    """Returns the JSON path for a given frame's circle data."""
    return os.path.join(circles_dir_path(base_dir), f"{base_name}.json")


def save_circle_json(
    base_name: str, base_dir: str, cx: float, cy: float, r: float, H: int, W: int
) -> None:
    """Saves circle parameters for one frame as a JSON file.

    Args:
        base_name: Stem of the source image filename (no extension).
        base_dir:  Directory that contains the ``circles/`` sub-directory.
        cx: Circle centre x in pixels.
        cy: Circle centre y in pixels.
        r:  Circle radius in pixels.
        H:  Image height used when the circle was detected.
        W:  Image width used when the circle was detected.
    """
    meta = {"cx": float(cx), "cy": float(cy), "r": float(r), "H": int(H), "W": int(W)}
    with open(circle_json_path(base_name, base_dir), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_circle_from_json(
    base_name: str, circles_dir: str, H: int, W: int
) -> Optional[Tuple[float, float, float]]:
    """Loads circle parameters from a previously saved JSON file.

    Coordinates are rescaled automatically when the stored image resolution
    differs from the requested ``(H, W)``.

    Args:
        base_name:   Stem of the source image filename (no extension).
        circles_dir: Directory containing ``<base_name>.json``.
        H:           Target image height.
        W:           Target image width.

    Returns:
        ``(cx, cy, r)`` rescaled to ``(H, W)``, or ``None`` if the file does
        not exist or cannot be parsed.
    """
    p = os.path.join(circles_dir, f"{base_name}.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        r = float(meta["r"])
        H0 = int(meta.get("H", H))
        W0 = int(meta.get("W", W))
    except Exception:
        return None
    if (H0, W0) != (H, W):
        sx = W0 / float(W)
        sy = H0 / float(H)
        cx /= sx
        cy /= sy
        r /= 0.5 * (sx + sy)
    return float(cx), float(cy), float(r)


def load_circle_from_debug_vis(
    base_name: str, debug_dir: str, H: int, W: int
) -> Optional[Tuple[float, float, float]]:
    """Recovers a circle from a saved debug visualization image.

    Looks for ``<base_name>_manual_selected.png`` then
    ``<base_name>_loc_refined.png`` in ``debug_dir`` and runs Hough / contour
    circle detection on the first match.

    Args:
        base_name: Stem of the source image filename (no extension).
        debug_dir: Directory containing debug overlay images.
        H:         Target image height.
        W:         Target image width.

    Returns:
        ``(cx, cy, r)`` rescaled to ``(H, W)``, or ``None`` on failure.
    """
    cand_paths = [
        os.path.join(debug_dir, f"{base_name}_manual_selected.png"),
        os.path.join(debug_dir, f"{base_name}_loc_refined.png"),
    ]
    for vis_path in cand_paths:
        if not os.path.isfile(vis_path):
            continue
        vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)
        if vis is None:
            continue
        sol = _detect_circle_from_vis(vis)
        if sol is None:
            continue
        cx_vis, cy_vis, r_vis = sol
        h_vis, w_vis = vis.shape[:2]
        sx = w_vis / float(W)
        sy = h_vis / float(H)
        cx = cx_vis / sx
        cy = cy_vis / sy
        r = r_vis / (0.5 * (sx + sy))
        return float(cx), float(cy), float(r)
    return None


class ManualCirclePicker:
    """Interactive OpenCV window for manually annotating a ball contact circle.

    The user left-clicks to add points on the ring edge; right-click removes
    the last point. Once at least ``min_points`` points are placed, pressing
    Enter confirms the fitted circle. Middle-drag pans; scroll-wheel zooms.

    Attributes:
        win: OpenCV window name.
        min_points: Minimum number of clicked points required to confirm.
        draw_scale: Window display scale relative to the source image size.
    """

    def __init__(
        self, win_name: str = "manual-pick", min_points: int = 3, draw_scale: float = 1.0
    ):
        self.win = win_name
        self.min_points = int(min_points)
        self.draw_scale = float(draw_scale)
        self._pts: List[Tuple[float, float]] = []
        self._img_base: Optional[np.ndarray] = None
        self._ann_full: Optional[np.ndarray] = None
        self._view: Optional[np.ndarray] = None
        self._cur_fit: Optional[Tuple[float, float, float]] = None
        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._dragging_mid = False
        self._last_mouse = (0, 0)
        self._progress_text = ""

    def _clamp_view(self):
        H, W = self._img_base.shape[:2]
        vw = W / self._zoom
        vh = H / self._zoom
        self._ox = float(np.clip(self._ox, 0, max(W - vw, 0)))
        self._oy = float(np.clip(self._oy, 0, max(H - vh, 0)))

    def _img_to_view(self, x_im: float, y_im: float) -> Tuple[int, int]:
        H, W = self._img_base.shape[:2]
        win_w = int(W * self.draw_scale)
        win_h = int(H * self.draw_scale)
        vw = W / self._zoom
        vh = H / self._zoom
        x_rel = (x_im - self._ox) / vw
        y_rel = (y_im - self._oy) / vh
        return int(x_rel * win_w + 0.5), int(y_rel * win_h + 0.5)

    def _view_to_img(self, x_win: int, y_win: int) -> Tuple[float, float]:
        H, W = self._img_base.shape[:2]
        win_w = int(W * self.draw_scale)
        win_h = int(H * self.draw_scale)
        vw = W / self._zoom
        vh = H / self._zoom
        x_im = self._ox + (x_win / max(win_w, 1.0)) * vw
        y_im = self._oy + (y_win / max(win_h, 1.0)) * vh
        return float(x_im), float(y_im)

    def _compose_ann_full(self):
        vis = self._img_base.copy()
        for p in self._pts:
            cv2.circle(
                vis,
                (int(round(p[0])), int(round(p[1]))),
                3,
                (0, 255, 255),
                -1,
            )
        self._cur_fit = None
        if len(self._pts) >= 3:
            sol = fit_circle_least_squares(np.array(self._pts, dtype=np.float64))
            if sol is not None:
                cx, cy, r = sol
                self._cur_fit = (cx, cy, r)
                cv2.circle(
                    vis,
                    (int(round(cx)), int(round(cy))),
                    int(round(r)),
                    (0, 255, 0),
                    1,
                )
                cv2.circle(
                    vis,
                    (int(round(cx)), int(round(cy))),
                    2,
                    (0, 0, 255),
                    -1,
                )
        self._ann_full = vis

    def _render_view(self):
        H, W = self._img_base.shape[:2]
        win_w = int(W * self.draw_scale)
        win_h = int(H * self.draw_scale)
        vw = int(round(W / self._zoom))
        vh = int(round(H / self._zoom))
        x0 = int(round(self._ox))
        y0 = int(round(self._oy))
        x1 = int(np.clip(x0 + vw, 0, W))
        y1 = int(np.clip(y0 + vh, 0, H))
        roi = self._ann_full[y0:y1, x0:x1]
        if roi.size == 0:
            roi = self._ann_full
        view = cv2.resize(roi, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
        if self._progress_text:
            cv2.putText(
                view,
                self._progress_text,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                view,
                self._progress_text,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        self._view = view

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self._dragging_mid:
            xi, yi = self._view_to_img(x, y)
            self._pts.append((xi, yi))
            self._compose_ann_full()
            self._render_view()
        elif event == cv2.EVENT_RBUTTONDOWN and not self._dragging_mid:
            if self._pts:
                self._pts.pop()
                self._compose_ann_full()
                self._render_view()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._dragging_mid = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self._dragging_mid = False
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_mid:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._last_mouse = (x, y)
            H, W = self._img_base.shape[:2]
            win_w = int(W * self.draw_scale)
            win_h = int(H * self.draw_scale)
            vw = W / self._zoom
            vh = H / self._zoom
            self._ox -= dx * (vw / max(win_w, 1.0))
            self._oy -= dy * (vh / max(win_h, 1.0))
            self._clamp_view()
            self._render_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            zoom_factor = 1.15 if flags > 0 else 1.0 / 1.15
            old_zoom = self._zoom
            new_zoom = float(np.clip(old_zoom * zoom_factor, 1.0, 12.0))
            if abs(new_zoom - old_zoom) < 1e-6:
                return
            xi, yi = self._view_to_img(x, y)
            self._zoom = new_zoom
            self._clamp_view()
            H, W = self._img_base.shape[:2]
            win_w = int(W * self.draw_scale)
            win_h = int(H * self.draw_scale)
            vw = W / self._zoom
            vh = H / self._zoom
            self._ox = xi - (x / max(win_w, 1.0)) * vw
            self._oy = yi - (y / max(win_h, 1.0)) * vh
            self._clamp_view()
            self._render_view()

    def set_progress(self, done: int, total: int):
        """Updates the progress overlay shown in the picker window.

        Args:
            done:  Number of frames already processed.
            total: Total number of frames in the calibration run.
        """
        left = max(total - done, 0)
        self._progress_text = f"Progress: {done}/{total}  (Left: {left})"

    def pick_on_image(
        self, bgr_lin: np.ndarray
    ) -> Optional[Tuple[float, float, float, np.ndarray, np.ndarray]]:
        """Opens the interactive window and waits for the user to mark the circle.

        Args:
            bgr_lin: Linear-light BGR image in [0, 1] float32.

        Returns:
            A tuple ``(cx, cy, r, core_mask, vis)`` where ``core_mask`` is a
            uint8 binary mask of the contact disc (rim inset by 1.5 px) and
            ``vis`` is the annotated BGR uint8 overlay.  Returns ``None`` if
            the user closes the window without confirming.
        """
        self._pts.clear()
        self._cur_fit = None
        base8 = (bgr_lin * 255.0).astype(np.uint8)
        self._img_base = base8
        self._zoom = 1.0
        self._ox = 0.0
        self._oy = 0.0
        self._compose_ann_full()
        self._render_view()

        H, W = self._img_base.shape[:2]
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.win, int(W * self.draw_scale), int(H * self.draw_scale)
        )
        cv2.setMouseCallback(self.win, self._mouse)

        cx = cy = r = None
        while True:
            cv2.imshow(self.win, self._view)
            k = cv2.waitKey(30) & 0xFF
            if k == 13:  # Enter
                if len(self._pts) >= self.min_points and self._cur_fit is not None:
                    cx, cy, r = self._cur_fit
                    break
                else:
                    tmp = self._view.copy()
                    cv2.putText(
                        tmp,
                        f"Need >= {self.min_points} points",
                        (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(self.win, tmp)
                    cv2.waitKey(600)

        cv2.destroyWindow(self.win)

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        r_eff = max(r - 1.5, 1.0)
        core = (rr2 <= (r_eff * r_eff)).astype(np.uint8)

        vis = self._img_base.copy()
        cv2.circle(
            vis,
            (int(round(cx)), int(round(cy))),
            int(round(r)),
            (0, 255, 0),
            2,
        )
        cv2.circle(
            vis,
            (int(round(cx)), int(round(cy))),
            2,
            (0, 0, 255),
            -1,
        )
        return float(cx), float(cy), float(r), core, vis


def _read_radius_csv(csv_path: str) -> List[Optional[float]]:
    """Reads per-frame contact radii from a CSV file.

    Parses the first numeric token on each row.  Rows that cannot be parsed
    (e.g. header lines, blank lines) are stored as ``None`` and trigger a
    fallback to the manually picked radius at call time.  No unit conversion
    is performed here; callers apply ``pixel_size_mm`` as needed.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A list with one entry per CSV row; each entry is a float radius or
        ``None`` for unparseable rows.
    """
    vals: List[Optional[float]] = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            v: Optional[float] = None
            for tok in row:
                tok = (tok or "").strip()
                if not tok:
                    continue
                try:
                    v = float(tok)
                    break
                except Exception:
                    continue
            vals.append(v)
    return vals


def _collect_radius_subdirs(img_dir: str, max_imgs: int) -> List[Tuple[float, List[str]]]:
    """Scans ``img_dir`` for radius-named subdirectories and collects image paths.

    Each subdirectory whose name can be parsed as a float is treated as a
    distinct ball radius (in mm).  This allows a single calibration run to
    consume images from multiple ball sizes, improving LUT coverage.

    Args:
        img_dir:  Parent directory containing radius-named subdirectories.
        max_imgs: If positive, each subdirectory is capped at this many images.

    Returns:
        A list of ``(R_mm, image_paths)`` tuples sorted by ascending radius.
        Only subdirectories that contain at least one image are included.

    Raises:
        ValueError: If ``img_dir`` does not exist or contains no valid radius
            subdirectories.
    """
    result = []
    try:
        entries = sorted(os.listdir(img_dir))
    except FileNotFoundError:
        raise ValueError(f"image_dir does not exist: {img_dir}")
    for name in entries:
        subdir = os.path.join(img_dir, name)
        if not os.path.isdir(subdir):
            continue
        try:
            R_mm = float(name)
        except ValueError:
            continue
        paths = list_images(subdir)
        if max_imgs and max_imgs > 0:
            paths = paths[:max_imgs]
        if paths:
            result.append((R_mm, paths))
    if not result:
        raise ValueError(
            f"No valid radius subdirectories found under {img_dir}.\n"
            f"Place calibration images in subdirectories named by radius (mm), e.g.:\n"
            f"  {img_dir}/1.0/000000.png\n"
            f"  {img_dir}/2.0/000000.png"
        )
    return result


def run_ball_calibration(cfg: Dict[str, Any]) -> str:
    """Runs the full ball-press photometric stereo calibration pipeline.

    Reads calibration images from radius-named subdirectories under
    ``cfg["calib"]["image_dir"]``, accumulates per-pixel normal observations,
    solves a global + block-residual ridge regression, and writes the resulting
    weight field and metadata to ``cfg["output"]["calib_out_dir"]``.

    Args:
        cfg: Top-level configuration dict.  See ``configs/example_config.py``
            for the full schema.

    Returns:
        Path to the written ``rgb2n_lut.yaml`` file.

    Raises:
        ValueError:   If no valid radius subdirectories are found.
        RuntimeError: If no calibration samples are collected after processing
            all frames.
    """
    img_dir = cfg["calib"]["image_dir"]
    out_dir = cfg["output"]["calib_out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    H = _as_int_scalar(cfg["camera"]["height"])
    W = _as_int_scalar(cfg["camera"]["width"])
    pix_mm = float(cfg["camera"]["pixel_size_mm"])
    pre = Preprocessor(cfg.get("preproc", {}))

    lam = float(cfg["calib"].get("ridge_lambda", 1e-2))
    min_spp = _as_int_scalar(cfg["calib"].get("min_samples_per_pixel", 6))
    delta_alpha = float(cfg["calib"].get("delta_alpha", 1.0))
    if delta_alpha < 0.0:
        delta_alpha = 0.0
    if delta_alpha > 1.0:
        delta_alpha = 1.0
    do_eval = bool(cfg["calib"].get("eval_export", True))
    debug_dump = bool(cfg["calib"].get("debug_dump", True))

    rim_shrink_pix = float(cfg["calib"].get("rim_shrink_pix", 1.5))
    nz_floor = float(cfg["calib"].get("nz_floor", 0.12))

    # Regression output dimension: 2 → (nx/nz, ny/nz); 3 → full unit normal.
    out_dim = int(cfg["calib"].get("out_dim", 2))
    if out_dim not in (2, 3):
        raise ValueError(f"calib.out_dim must be 2 or 3, got {out_dim}")

    # Contact-radius CSV mode: per-frame contact radius supplied externally.
    use_manual_radius_csv = bool(cfg["calib"].get("use_manual_radius_csv", False))
    radius_csv_cfg = cfg.get("manual_radius_csv", {}) or {}
    radius_csv_path = (radius_csv_cfg.get("csv_path") or "").strip()
    radius_csv_unit = (radius_csv_cfg.get("unit") or "mm").strip().lower()  # "mm" | "pix"

    if use_manual_radius_csv:
        assert (
            radius_csv_path
        ), "use_manual_radius_csv=True but manual_radius_csv.csv_path is not set"

    make_features_from_rgb, feature_names = build_feature_extractor(cfg.get("calib", {}))
    FEAT_DIM = len(feature_names)

    mp_cfg = cfg.get("manual_pick", {}) or {}
    draw_scale = float(mp_cfg.get("draw_scale", 1.0))
    min_pts = int(mp_cfg.get("min_points", 3))
    reuse_circles_dir = (mp_cfg.get("reuse_from_circles_dir") or "").strip()
    reuse_debug_dir = (mp_cfg.get("reuse_from_debug_dir") or "").strip()
    picker = ManualCirclePicker(
        win_name="manual-pick", min_points=min_pts, draw_scale=draw_scale
    )

    debug_dir = os.path.join(out_dir, "debug")
    if debug_dump:
        os.makedirs(debug_dir, exist_ok=True)
        print(
            f"[calib] (info) reuse_from_circles_dir = {reuse_circles_dir or '(none)'}"
        )
        print(f"[calib] (info) reuse_from_debug_dir   = {reuse_debug_dir or '(none)'}")
        print(f"[calib] (info) save circles to        = {circles_dir_path(debug_dir)}")

    # Collect all (R_mm, path) pairs from radius-named subdirectories.
    max_imgs = _as_int_scalar(cfg["calib"].get("max_images", 0))
    radius_entries = _collect_radius_subdirs(img_dir, max_imgs)
    img_paths_with_R: List[Tuple[float, str]] = [
        (R_mm_i, p) for R_mm_i, paths in radius_entries for p in paths
    ]
    assert len(img_paths_with_R) > 0, f"No images found under {img_dir}"
    img_paths = [p for _, p in img_paths_with_R]
    all_R_mm = sorted({R for R, _ in img_paths_with_R})
    print(f"[calib] radius subdirs detected: {all_R_mm} mm, total {len(img_paths_with_R)} frames")

    # Load per-frame contact radii from CSV when that mode is enabled.
    radii_csv_vals: List[Optional[float]] = []
    if use_manual_radius_csv:
        radii_csv_vals = _read_radius_csv(radius_csv_path)
        if len(radii_csv_vals) < len(img_paths_with_R):
            print(
                f"[calib][warn] CSV rows ({len(radii_csv_vals)}) < images "
                f"({len(img_paths_with_R)}); missing frames will fall back to manual radius."
            )
        elif len(radii_csv_vals) > len(img_paths_with_R):
            print(
                f"[calib][warn] CSV rows ({len(radii_csv_vals)}) > images "
                f"({len(img_paths_with_R)}); extra rows will be ignored."
            )

    # Accumulation matrices for per-pixel normal least-squares.
    XtX = np.zeros((H, W, FEAT_DIM, FEAT_DIM), np.float64)
    XtY = np.zeros((H, W, FEAT_DIM, out_dim), np.float64)
    Ns = np.zeros((H, W), np.int32)

    n_ok = 0
    total_frames = len(img_paths_with_R)
    done_frames = 0

    for idx, (R_mm, p) in enumerate(img_paths_with_R, 1):
        bgr = imread_color(p)
        if bgr.shape[0] != H or bgr.shape[1] != W:
            if idx == 1:
                print(
                    f"[warn] image size {bgr.shape[1]}x{bgr.shape[0]} != config {W}x{H}, resizing…"
                )
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        bgr_lin = pre.apply(bgr).astype(np.float32)
        rgb = bgr_lin[:, :, ::-1]
        base = os.path.splitext(os.path.basename(p))[0]

        cx = cy = r_est = None
        core = None
        vis_pick = None

        picker.set_progress(done_frames, total_frames)

        # Branch A: contact-radius CSV mode — centre from JSON cache or manual
        # pick, contact radius from the CSV row for this frame index.
        if use_manual_radius_csv:
            got_center = None
            if reuse_circles_dir:
                got_center = load_circle_from_json(base, reuse_circles_dir, H, W)
            if got_center is None:
                got_center = load_circle_from_json(
                    base, os.path.join(debug_dir, "circles"), H, W
                )

            if got_center is not None:
                cx, cy, _r_ignored = got_center
            else:
                picked = picker.pick_on_image(bgr_lin)
                if picked is None:
                    if debug_dump:
                        print(f"[manual] pick failed on frame {idx:06d}, skip")
                    done_frames += 1
                    continue
                cx, cy, r_manual, core, vis_pick = picked

            if (
                idx - 1 < len(radii_csv_vals)
                and radii_csv_vals[idx - 1] is not None
            ):
                r_val = float(radii_csv_vals[idx - 1])
                if radius_csv_unit == "mm":
                    r_est = r_val / max(pix_mm, 1e-12)
                else:
                    r_est = r_val
            else:
                if "r_manual" in locals():
                    r_est = float(r_manual)
                    if debug_dump:
                        print(
                            f"[calib][warn] CSV row {idx} missing/invalid, "
                            f"falling back to manual radius r={r_est:.3f} px"
                        )
                else:
                    if debug_dump:
                        print(
                            f"[calib][warn] CSV row {idx} missing/invalid and "
                            f"no manual radius available; skipping frame"
                        )
                    done_frames += 1
                    continue

            Hh, Ww = H, W
            yy, xx = np.meshgrid(np.arange(Hh), np.arange(Ww), indexing="ij")
            core = (
                (xx - cx) ** 2 + (yy - cy) ** 2
                <= (max(r_est - 1.5, 1.0) ** 2)
            ).astype(np.uint8)

            save_circle_json(base, debug_dir, cx, cy, float(r_est), H, W)
            if debug_dump:
                vis2 = (bgr_lin * 255.0).astype(np.uint8).copy()
                cv2.circle(
                    vis2,
                    (int(round(cx)), int(round(cy))),
                    int(round(r_est)),
                    (0, 255, 0),
                    2,
                )
                cv2.circle(
                    vis2,
                    (int(round(cx)), int(round(cy))),
                    2,
                    (0, 0, 255),
                    -1,
                )
                imwrite(
                    os.path.join(debug_dir, f"{base}_manual_selected.png"),
                    vis2,
                )
        else:
            # Priority A: reuse cached circle JSON from a previous run.
            if reuse_circles_dir:
                got = load_circle_from_json(base, reuse_circles_dir, H, W)
                if got is not None:
                    cx, cy, r_est = got
                    yy, xx = np.meshgrid(
                        np.arange(H), np.arange(W), indexing="ij"
                    )
                    core = (
                        (xx - cx) ** 2 + (yy - cy) ** 2
                        <= (max(r_est - 1.5, 1.0) ** 2)
                    ).astype(np.uint8)
                    vis_pick = (bgr_lin * 255.0).astype(np.uint8).copy()
                    cv2.circle(
                        vis_pick,
                        (int(round(cx)), int(round(cy))),
                        int(round(r_est)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(
                        vis_pick,
                        (int(round(cx)), int(round(cy))),
                        2,
                        (0, 0, 255),
                        -1,
                    )

            # Priority B: auto-detect circle from a saved debug visualization.
            if (cx is None or cy is None or r_est is None) and reuse_debug_dir:
                got = load_circle_from_debug_vis(
                    base, reuse_debug_dir, H, W
                )
                if got is not None:
                    cx, cy, r_est = got
                    yy, xx = np.meshgrid(
                        np.arange(H), np.arange(W), indexing="ij"
                    )
                    core = (
                        (xx - cx) ** 2 + (yy - cy) ** 2
                        <= (max(r_est - 1.5, 1.0) ** 2)
                    ).astype(np.uint8)
                    vis_pick = (bgr_lin * 255.0).astype(np.uint8).copy()
                    cv2.circle(
                        vis_pick,
                        (int(round(cx)), int(round(cy))),
                        int(round(r_est)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(
                        vis_pick,
                        (int(round(cx)), int(round(cy))),
                        2,
                        (0, 0, 255),
                        -1,
                    )

            # Priority C: interactive manual picking as last resort.
            if cx is None or cy is None or r_est is None:
                picked = picker.pick_on_image(bgr_lin)
                if picked is None:
                    if debug_dump:
                        print(f"[manual] pick failed on frame {idx:06d}, skip")
                    done_frames += 1
                    continue
                cx, cy, r_est, core, vis_pick = picked

            save_circle_json(base, debug_dir, cx, cy, r_est, H, W)
            if debug_dump and vis_pick is not None:
                imwrite(
                    os.path.join(debug_dir, f"{base}_manual_selected.png"),
                    vis_pick,
                )

        # Compute ground-truth surface normals from sphere geometry.
        n_gt, mask_geo = sphere_normals_for_contact(
            H,
            W,
            cx,
            cy,
            float(r_est),
            pix_mm,
            R_mm,
            rim_shrink_pix=rim_shrink_pix,
            nz_floor=nz_floor,
        )

        core = ((core > 0) & (mask_geo > 0)).astype(np.uint8)
        if core.sum() < 50:
            if debug_dump:
                dbg_img = (bgr_lin * 255.0).astype(np.uint8).copy()
                cv2.putText(
                    dbg_img,
                    f"CORE TOO SMALL: {int(core.sum())}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                imwrite(
                    os.path.join(debug_dir, f"{base}_loc_refined.png"), dbg_img
                )
            done_frames += 1
            continue

        if debug_dump:
            dbg_img = (bgr_lin * 255.0).astype(np.uint8).copy()
            cv2.circle(
                dbg_img,
                (int(round(cx)), int(round(cy))),
                int(round(r_est)),
                (255, 255, 255),
                2,
            )
            imwrite(
                os.path.join(debug_dir, f"{base}_loc_refined.png"), dbg_img
            )

        sat_hi = float(cfg["calib"].get("sat_hi", 0.98))
        dark_lo = float(cfg["calib"].get("dark_lo", 0.03))
        drop_sat = bool(cfg["calib"].get("drop_saturated", True))
        drop_dark = bool(cfg["calib"].get("drop_dark", True))
        w_sat = float(cfg["calib"].get("weight_saturated", 0.3))
        w_dark = float(cfg["calib"].get("weight_dark", 0.3))

        Rch, Gch, Bch = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        I = Rch + Gch + Bch
        sat_mask = (Rch >= sat_hi) | (Gch >= sat_hi) | (Bch >= sat_hi)
        dark_mask = I <= dark_lo

        X = make_features_from_rgb(rgb)

        if out_dim == 3:
            Y = n_gt[..., :3]
        else:
            Y = n_gt[..., :2]

        ys_core, xs_core = np.where(core > 0)
        for yy, xx in zip(ys_core, xs_core):
            v = X[yy, xx]
            yv = Y[yy, xx]
            w = 1.0
            if sat_mask[yy, xx]:
                if drop_sat:
                    continue
                else:
                    w *= w_sat
            if dark_mask[yy, xx]:
                if drop_dark:
                    continue
                else:
                    w *= w_dark
            XtX[yy, xx] += w * np.outer(v, v)
            XtY[yy, xx] += w * np.outer(v, yv)
            Ns[yy, xx] += int(w > 0.0)

        n_ok += 1
        done_frames += 1

        if do_eval and (idx % 5 == 0 or idx == len(img_paths_with_R)):
            cov = (Ns / max(1, Ns.max()) * 255).astype(np.uint8)
            imwrite(
                os.path.join(out_dir, "coverage_progress.png"),
                cv2.applyColorMap(cov, cv2.COLORMAP_JET),
            )

    print(f"[calib] frames used={n_ok} / {len(img_paths_with_R)}")

    # -------------------------------------------------------------------------
    # Solve: global + 8x8 block-residual ridge regression with Gaussian blending
    # -------------------------------------------------------------------------
    sel = Ns > 0
    if sel.sum() == 0:
        raise RuntimeError("No valid calibration samples collected.")

    FEAT_DIM = len(feature_names)
    I = np.eye(FEAT_DIM, dtype=np.float64)

    # Step 1: global ridge regression over all pixels.
    A_glob = np.zeros((FEAT_DIM, FEAT_DIM), np.float64)
    B_glob = np.zeros((FEAT_DIM, out_dim), np.float64)

    ys, xs = np.where(sel)
    for yy, xx in zip(ys, xs):
        A_glob += XtX[yy, xx]
        B_glob += XtY[yy, xx]

    W_global = np.linalg.solve(
        A_glob + float(lam) * I,
        B_glob
    ).astype(np.float32)

    # Step 2: per-block residual correction (8x8 grid).
    # Stronger regularisation prevents block residuals from overfitting.
    K = 8
    bh = (H + K - 1) // K
    bw = (W + K - 1) // K
    lam_local = 20.0 * float(lam)

    dW_blocks = np.zeros((K, K, FEAT_DIM, out_dim), np.float32)
    block_valid = np.zeros((K, K), dtype=bool)

    for by in range(K):
        y0 = by * bh
        y1 = min((by + 1) * bh, H)
        if y0 >= y1:
            continue

        for bx in range(K):
            x0 = bx * bw
            x1 = min((bx + 1) * bw, W)
            if x0 >= x1:
                continue

            A_blk = np.zeros((FEAT_DIM, FEAT_DIM), np.float64)
            B_blk = np.zeros((FEAT_DIM, out_dim), np.float64)

            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if Ns[yy, xx] > 0:
                        A_blk += XtX[yy, xx]
                        B_blk += XtY[yy, xx]

            if A_blk.trace() < 1e-8:
                continue

            B_res = B_blk - A_blk @ W_global  # residual regression target

            try:
                dW = np.linalg.solve(
                    A_blk + lam_local * I,
                    B_res
                ).astype(np.float32)
                dW_blocks[by, bx] = dW
                block_valid[by, bx] = True
            except np.linalg.LinAlgError:
                continue

    # Step 3: Gaussian-blend block residuals into a per-pixel weight field.
    W_field = np.zeros((H, W, FEAT_DIM, out_dim), np.float32)

    sigma_x = bw * 0.75
    sigma_y = bh * 0.75

    for y in range(H):
        for x in range(W):
            W_acc = W_global.copy()
            wsum = 0.0

            by_f = (y + 0.5) / bh
            bx_f = (x + 0.5) / bw

            for by in range(max(0, int(by_f) - 1), min(K, int(by_f) + 2)):
                for bx in range(max(0, int(bx_f) - 1), min(K, int(bx_f) + 2)):
                    if not block_valid[by, bx]:
                        continue

                    cy = (by + 0.5) * bh
                    cx = (bx + 0.5) * bw

                    wy = np.exp(-((y + 0.5 - cy) ** 2) / (2 * sigma_y ** 2))
                    wx = np.exp(-((x + 0.5 - cx) ** 2) / (2 * sigma_x ** 2))
                    w = wy * wx

                    W_acc += w * dW_blocks[by, bx]
                    wsum += w

            if wsum > 1e-6:
                W_acc = W_global + (W_acc - W_global) / wsum

            W_field[y, x] = W_acc

    # smooth_delta_w is no longer applied; the config key is read only to
    # avoid breaking existing config files that still contain it.
    smooth_delta = bool(cfg["calib"].get("smooth_delta_w", False))
    _ = smooth_delta

    npz_path = os.path.join(out_dir, "rgb2n_field.npz")
    np.savez_compressed(npz_path, W=W_field)

    meta = {
        "type": "rgb_to_normal_linear_per_pixel",
        "field_shape": [
            _as_pyint(H),
            _as_pyint(W),
            int(FEAT_DIM),
            int(out_dim),
        ],
        "out_dim": int(out_dim),
        "pixel_size_mm": float(pix_mm),
        "sphere_radius_mm_list": [float(r) for r in all_R_mm],
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
        yaml.safe_dump(
            {"meta": meta, "weights_npz": os.path.basename(npz_path)},
            f,
            allow_unicode=True,
        )

    cov = (Ns / max(1, Ns.max()) * 255).astype(np.uint8)
    imwrite(
        os.path.join(out_dir, "coverage.png"),
        cv2.applyColorMap(cov, cv2.COLORMAP_JET),
    )
    np.save(os.path.join(out_dir, "coverage.npy"), Ns)

    # -------------------------------------------------------------------------
    # Evaluation: hold-out 10% of frames to report angular accuracy.
    # Does not affect the trained weights.
    # -------------------------------------------------------------------------
    try:
        N_imgs = len(img_paths_with_R)
        k_test = max(1, int(round(0.10 * N_imgs)))
        test_indices = np.linspace(
            0, N_imgs - 1, num=k_test, dtype=int
        ).tolist()

        test_recs: List[dict] = []
        acc_all: List[float] = []

        for i in test_indices:
            R_mm, p = img_paths_with_R[i]
            base = os.path.splitext(os.path.basename(p))[0]

            bgr = imread_color(p)
            if bgr is None or bgr.size == 0:
                test_recs.append({"image": base, "skip": "imread_failed"})
                continue

            try:
                H_t = int(float(np.asarray(H).ravel()[0]))
            except Exception:
                H_t = int(bgr.shape[0])
            try:
                W_t = int(float(np.asarray(W).ravel()[0]))
            except Exception:
                W_t = int(bgr.shape[1])
            if H_t <= 0:
                H_t = int(bgr.shape[0])
            if W_t <= 0:
                W_t = int(bgr.shape[1])

            if int(bgr.shape[0]) != H_t or int(bgr.shape[1]) != W_t:
                bgr = cv2.resize(bgr, (W_t, H_t), interpolation=cv2.INTER_LINEAR)

            bgr_lin = pre.apply(bgr).astype(np.float32)
            rgb = bgr_lin[:, :, ::-1]

            got = load_circle_from_json(
                base, os.path.join(out_dir, "debug", "circles"), H_t, W_t
            )
            if got is None:
                test_recs.append(
                    {"image": base, "skip": "circle_json_missing"}
                )
                continue
            cx, cy, r_est = got

            n_gt, mask_geo = sphere_normals_for_contact(
                H_t,
                W_t,
                cx,
                cy,
                float(r_est),
                pix_mm,
                R_mm,
                rim_shrink_pix=rim_shrink_pix,
                nz_floor=nz_floor,
            )
            mask = mask_geo > 0
            if mask.sum() == 0:
                test_recs.append({"image": base, "skip": "empty_mask"})
                continue

            X = make_features_from_rgb(rgb)
            Yp = np.einsum("ijk,ijkl->ijl", X, W_field)
            if out_dim == 2:
                nx, ny = Yp[..., 0], Yp[..., 1]
                nz = np.sqrt(
                    np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0)
                )
                n_pred = np.stack([nx, ny, nz], axis=-1)
            else:
                n_pred = Yp[..., :3]

            def _norm(v):
                return v / (
                    np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
                )

            n_pred_u = _norm(n_pred)
            n_gt_u = _norm(n_gt[..., :3])

            cos = (n_pred_u[mask] * n_gt_u[mask]).sum(axis=-1)
            cos = np.clip(cos, -1.0, 1.0)
            acc = (1.0 + cos) * 0.5
            acc_mean = float(acc.mean())
            test_recs.append(
                {
                    "image": base,
                    "pixels": int(mask.sum()),
                    "acc_pct": round(acc_mean * 100.0, 4),
                    "err_pct": round((1.0 - acc_mean) * 100.0, 4),
                    "acc_median_pct": round(float(np.median(acc) * 100.0), 4),
                }
            )
            acc_all.append(acc_mean)

        summary = {
            "num_test_images": len(test_recs),
            "acc_mean_pct": round(
                (float(np.mean(acc_all)) * 100.0) if acc_all else 0.0, 4
            ),
            "err_mean_pct": round(
                (100.0 - float(np.mean(acc_all)) * 100.0) if acc_all else 0.0,
                4,
            ),
        }
        with open(
            os.path.join(out_dir, "test_eval.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"summary": summary, "details": test_recs},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"[eval] test results saved: {os.path.join(out_dir, 'test_eval.json')}"
        )
    except Exception as _e:
        print(f"[eval][warn] evaluation failed: {_e}")

    # -------------------------------------------------------------------------
    # Geometric depth integration: integrate normals → depth for each training
    # frame and record per-frame depth statistics for qualitative inspection.
    # -------------------------------------------------------------------------
    from ..utils.poisson import integrate_normals_poisson

    geom_dir = os.path.join(out_dir, "geom_depth")
    os.makedirs(geom_dir, exist_ok=True)

    press_stats: Dict[str, Dict[str, float]] = {}

    H_int, W_int = Ns.shape

    for R_mm, p2 in img_paths_with_R:
        base = os.path.splitext(os.path.basename(p2))[0]
        got = load_circle_from_json(
            base, os.path.join(out_dir, "debug", "circles"), H_int, W_int
        )
        if got is None:
            continue
        cx, cy, r_est = got

        n_gt, mask_geo = sphere_normals_for_contact(
            H_int,
            W_int,
            cx,
            cy,
            float(r_est),
            pix_mm,
            R_mm,
            rim_shrink_pix=rim_shrink_pix,
            nz_floor=nz_floor,
        )
        mask = (mask_geo > 0)
        if mask.sum() == 0:
            continue

        nx = n_gt[..., 0]
        ny = n_gt[..., 1]
        nz = n_gt[..., 2]
        p_core = -nx / (nz + 1e-9)
        q_core = -ny / (nz + 1e-9)

        depth_core = integrate_normals_poisson(p_core, q_core).astype(np.float32)

        mask_u8 = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        boundary = mask & (eroded == 0)

        valid_boundary = boundary & np.isfinite(depth_core)
        if not np.any(valid_boundary):
            valid_boundary = mask & np.isfinite(depth_core)

        if np.any(valid_boundary):
            min_edge = float(depth_core[valid_boundary].min())
        else:
            min_edge = 0.0

        shift = -min_edge + 1e-6
        depth_shifted = depth_core + shift
        depth_shifted[~mask] = 0.0

        if np.any(mask):
            max_depth = float(depth_shifted[mask].max())
            min_depth = float(depth_shifted[mask].min())
            absmax_depth = float(np.max(np.abs(depth_shifted[mask])))
        else:
            max_depth = min_depth = absmax_depth = 0.0

        press_stats[base] = {
            "max": max_depth,
            "min": min_depth,
            "absmax": absmax_depth,
        }

        np.save(
            os.path.join(geom_dir, f"{base}_press_geom.npy"),
            depth_shifted.astype(np.float32),
        )

    with open(os.path.join(geom_dir, "press_geom_stats.json"), "w", encoding="utf-8") as f_js:
        json.dump(press_stats, f_js, indent=2, ensure_ascii=False)

    return yaml_path
