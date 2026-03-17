# GelSight Mini — 标准球标定的近场 Lambert 光度立体

> 基于 MIT `gelsight_driver` 思路做的**最小可用**版本：支持**从目录读取图像**完成"**标准球标定 → RGB→法向 LUT（YAML+NPZ） → 解算（法向/深度）**"，同时支持**实时摄像头模式**。

## 目录结构

```
GDLT-NEW/
├── configs/
│   └── example_config.py      # 配置文件模板
├── gelsight_ps/
│   ├── main.py                # 程序入口
│   ├── config.py              # 配置加载器
│   ├── calib/
│   │   └── ball_calibrate.py  # 球压标定流程
│   ├── solve/
│   │   └── reconstruct.py     # 法向/深度解算流程（含实时模式）
│   ├── image_io/
│   │   └── image_loader.py    # 图像 I/O 与预处理
│   └── utils/
│       ├── geometry.py        # 球面法向计算
│       └── poisson.py         # Poisson 深度积分
└── README.md
```

---

## 依赖安装

- Python 3.8+

```bash
pip install numpy opencv-python pyyaml scipy
```

---

## 一、数据准备

### 1.1 参考图（无压痕）

采集一组**未施压**时的传感器图像，放入同一目录，用于计算参考基准平面：

```
plain/
├── 000000.png
├── 000001.png
└── ...
```

在配置中将该目录填入 `calib.ref_nopress_dir` 和 `solve.ref_nopress_dir`。

---

### 1.2 标定图（球压痕）

用已知半径的标准钢球在传感器上多位置、多压深压痕，采集图像。**目录以球的半径（mm）命名**，支持多种半径同时标定：

```
calib_data/
├── 1.0/          # 使用半径 1.0 mm 的球采集的图像
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── 2.0/          # 使用半径 2.0 mm 的球采集的图像
    ├── 000000.png
    └── ...
```

将该根目录填入 `calib.image_dir`。

> **建议**：每种半径采集 20 张以上，尽量覆盖传感器视场的不同区域，以保证 LUT 精度。

---

### 1.3 待解算图（solve 模式）

将需要解算法向/深度的压痕图像放入同一目录：

```
test_data/
├── 000000.png
├── 000001.png
└── ...
```

将该目录填入 `solve.image_dir`。

---

## 二、Config 配置

复制 `configs/example_config.py` 为你自己的配置文件，按实际情况修改以下字段。

配置文件支持 `.py`、`.yaml`、`.json` 三种格式；`.py` 格式中需定义名为 `CFG`、`CONFIG` 或 `config` 的字典变量。

### 2.1 `camera` — 相机参数

| 字段 | 说明 |
|------|------|
| `width` / `height` | 图像分辨率（像素） |
| `pixel_size_mm` | 单个像素的物理尺寸（mm/pixel） |

### 2.2 `preproc` — 预处理参数

| 字段 | 说明 |
|------|------|
| `dark_path` | 暗场图路径，为空则跳过暗场矫正 |
| `gamma_inv` | Gamma 逆校正指数（`1.0` = 不校正） |
| `wb_gain` | 白平衡增益 `[R, G, B]`（`[1.0,1.0,1.0]` = 不校正） |
| `clip_eps` | 归一化时的下界裁剪值（防止除零） |

### 2.3 `calib` — 标定参数

| 字段 | 说明 |
|------|------|
| `image_dir` | 球压痕标定图根目录（子目录以球半径命名） |
| `ref_nopress_dir` | 无压痕参考图目录 |
| `ridge_lambda` | 岭回归正则化系数（建议 `1e-2`） |
| `min_samples_per_pixel` | 每像素最少有效样本数（不足则用全局模型填充） |
| `max_images` | 每个半径目录最多加载的图像数（`0` = 全部加载） |
| `use_quadratic` | 是否启用二次特征（R²、G²、B²） |
| `use_cross` | 是否启用交叉特征（RG、RB、GB） |
| `out_dim` | 输出维度：`2` = (nx, ny)；`3` = (nx, ny, nz) |
| `rim_shrink_pix` | 圆边缘向内缩减像素数（避免边缘噪声） |
| `nz_floor` | 法向 z 分量下限（过滤掠射区域） |
| `eval_export` | 是否导出标定评估图 |
| `use_manual_radius_csv` | 是否从 CSV 文件读取每张图的精确球半径 |

### 2.4 `solve` — 解算参数

| 字段 | 说明 |
|------|------|
| `image_dir` | 待解算图像目录（`image_dir` 模式） |
| `lut_yaml` | 标定输出的 `rgb2n_lut.yaml` 路径 |
| `ref_nopress_dir` | 无压痕参考图目录 |
| `depth_from_normals` | 是否由法向图积分深度 |
| `smooth_grad_ksize` | 深度积分前梯度平滑核大小（奇数） |
| `depth_flip_sign` | 是否翻转深度符号 |
| `contact_mask_enable` | 是否启用接触区域掩膜 |
| `input_mode` | 输入模式：`"image_dir"`（批量文件）或 `"camera"`（实时摄像头） |
| `camera_id` | 实时模式下摄像头设备号（`0` = 内置，`1`/`2` = 外接） |
| `ref_frame_count` | 实时模式下用于建立参考基准的帧数 |
| `camera_subtract_ref_plane` | 实时模式下是否减去参考平面 |
| `d_thred_h` / `d_thred_l` | 实时模式深度显示的高/低阈值（mm） |

### 2.5 `output` — 输出路径

| 字段 | 说明 |
|------|------|
| `calib_out_dir` | 标定产物输出目录（LUT、覆盖图、调试图等） |
| `solve_out_dir` | 解算结果输出目录（法向图、深度图等） |

### 2.6 `manual_radius_csv` — 手动半径 CSV（可选）

当 `calib.use_manual_radius_csv = True` 时，从 CSV 文件读取每张图对应的精确球半径：

| 字段 | 说明 |
|------|------|
| `csv_path` | CSV 文件路径（需含图像文件名与对应半径列） |
| `unit` | 半径单位，`"mm"` 或 `"m"` |

---

## 三、启动方式

### 3.1 标定模式（calib）

生成每像素 RGB→法向 LUT，输出到 `output.calib_out_dir`：

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode calib
```

**标定输出文件：**

```
<calib_out_dir>/
├── rgb2n_lut.yaml        # LUT 元数据（指向权重文件）
├── rgb2n_field.npz       # 每像素权重场
├── coverage.png          # 各像素样本数热力图
├── debug/                # 每帧圆检测叠加图与 JSON 数据
└── geom_depth/           # 定性深度图（供目视检查）
```

---

### 3.2 解算模式（solve）— 批量文件

将 `solve.input_mode` 设为 `"image_dir"`，然后运行：

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode solve
```

**解算输出文件：**

```
<solve_out_dir>/
├── depth/    # {base}_depth_minus_ref.npy   （压痕减参考深度）
├── normal/   # {base}_normal.npy             （单位法向数组）
├── vis/      # {base}_depth_minus_ref.png, {base}_normal.png
├── ref/      # ref_depth.npy, ref_normal.npy
└── depth_diff_max_stats.json
```

---

### 3.3 解算模式（solve）— 实时摄像头

将 `solve.input_mode` 设为 `"camera"`，并设置 `solve.camera_id`，然后运行：

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode solve
```

启动后程序先采集 `ref_frame_count` 帧建立参考平面，随后在 OpenCV 窗口中实时显示减去参考后的深度图。按 **`q`** 退出。

---

## 四、注意事项

- **圆检测为启发式方法**：适合快速验证流程，工程落地建议使用更稳健的接触区域分割。
- **Lambert 模型为近似**：Mini 传感器近场多色光照存在 BRDF/渐晕等非理想项，LUT 会部分吸收这些误差。
- **Frankot–Chellappa / Poisson 深度积分**：仅作演示，实际应用可用正则化形貌融合替代。
- **逐像素回归**：当前使用全局共享 + 8×8 分块残差修正策略；若需完整逐像素模型，注意 I/O 体量。

---

## 许可

本示例为演示用途，按你的项目协议整合到原仓库时请遵守上游开源协议。
