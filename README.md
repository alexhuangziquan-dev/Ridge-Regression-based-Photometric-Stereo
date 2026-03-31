# Ridge Regression-based Photometric Stereo

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight photometric stereo pipeline for
[GelSight Mini](https://www.gelsight.com/) tactile sensors. It recovers
per-pixel surface normals and depth maps from RGB contact images using
**ridge regression** on sphere-calibration data under near-field,
multi-color illumination.

## Highlights

- **Ball calibration** — automatically detects pressed spheres of known radii,
  builds a per-pixel RGB-to-normal lookup table (LUT), and exports it as
  portable YAML + NPZ files.
- **Ridge regression model** — supports linear, quadratic, and cross-term
  features with configurable regularization, plus block-wise residual
  correction for spatial non-uniformity.
- **Batch and real-time reconstruction** — reconstructs surface normals and
  integrates depth via Frankot-Chellappa / Poisson solver, from either a
  directory of images or a live camera feed.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Notes](#notes)
- [License](#license)

## Project Structure

```text
.
├── configs/
│   └── example_config.py        # Example configuration template
├── gelsight_ps/
│   ├── main.py                  # Entry point (CLI dispatcher)
│   ├── config.py                # Configuration loader (.py / .yaml / .json)
│   ├── calib/
│   │   └── ball_calibrate.py    # Sphere-press calibration pipeline
│   ├── solve/
│   │   └── reconstruct.py       # Normal & depth reconstruction (batch / live)
│   ├── image_io/
│   │   └── image_loader.py      # Image I/O and preprocessing
│   └── utils/
│       ├── geometry.py          # Sphere normal computation
│       └── poisson.py           # Poisson depth integration
├── utils/                       # Shared utilities
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- A GelSight Mini sensor (or equivalent) for data acquisition

## Installation

```bash
pip install numpy opencv-python pyyaml scipy
```

> **Tip:** It is recommended to use a virtual environment or conda environment
> to avoid dependency conflicts.

## Data Preparation

### 1. Reference Images (No Contact)

Capture a set of images with **no object pressed** against the sensor.
These serve as the reference baseline for background subtraction.

```text
plain/
├── 000000.png
├── 000001.png
└── ...
```

Set this directory as `calib.ref_nopress_dir` and `solve.ref_nopress_dir` in
your configuration file.

### 2. Calibration Images (Sphere Presses)

Press calibration spheres of known radii at multiple positions and depths on
the sensor surface. Organize the images into **subdirectories named by sphere
radius in millimeters**:

```text
calib_data/
├── 1.0/            # Images captured with a 1.0 mm radius sphere
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── 2.0/            # Images captured with a 2.0 mm radius sphere
    ├── 000000.png
    └── ...
```

Set the root directory as `calib.image_dir`.

> **Recommendation:** Capture at least 20 images per radius, covering different
> regions of the sensor field of view, to ensure LUT accuracy.

### 3. Test Images (For Reconstruction)

Place the images you want to reconstruct into a single directory:

```text
test_data/
├── 000000.png
├── 000001.png
└── ...
```

Set this directory as `solve.image_dir`.

## Configuration

Copy `configs/example_config.py` to create your own configuration file and
modify the fields as needed. The configuration loader supports `.py`, `.yaml`,
and `.json` formats. For `.py` files, define a dictionary named `CFG`,
`CONFIG`, or `config`.

<details>
<summary><strong>Camera parameters</strong> — <code>camera</code></summary>

| Field | Description |
|---|---|
| `width` / `height` | Image resolution in pixels |
| `pixel_size_mm` | Physical size of a single pixel (mm/pixel) |

</details>

<details>
<summary><strong>Preprocessing</strong> — <code>preproc</code></summary>

| Field | Description |
|---|---|
| `dark_path` | Path to a dark-field image; leave empty to skip |
| `gamma_inv` | Inverse gamma correction exponent (`1.0` = disabled) |
| `wb_gain` | White balance gain `[R, G, B]` (`[1.0, 1.0, 1.0]` = disabled) |
| `clip_eps` | Lower-bound clipping value during normalization |

</details>

<details>
<summary><strong>Calibration</strong> — <code>calib</code></summary>

| Field | Description |
|---|---|
| `image_dir` | Root directory of sphere-press calibration images |
| `ref_nopress_dir` | Directory of no-contact reference images |
| `ridge_lambda` | Ridge regression regularization coefficient (suggested: `1e-2`) |
| `min_samples_per_pixel` | Minimum valid samples per pixel; pixels below this threshold fall back to the global model |
| `max_images` | Maximum images to load per radius subdirectory (`0` = all) |
| `use_quadratic` | Enable quadratic features (R², G², B²) |
| `use_cross` | Enable cross-term features (RG, RB, GB) |
| `out_dim` | Output dimensions: `2` = (nx, ny); `3` = (nx, ny, nz) |
| `rim_shrink_pix` | Inward shrink of detected circle rim in pixels |
| `nz_floor` | Minimum normal z-component (filters grazing regions) |
| `eval_export` | Export calibration evaluation plots |
| `use_manual_radius_csv` | Read per-image sphere radii from a CSV file |

</details>

<details>
<summary><strong>Reconstruction</strong> — <code>solve</code></summary>

| Field | Description |
|---|---|
| `image_dir` | Directory of images to reconstruct |
| `lut_yaml` | Path to the calibration output `rgb2n_lut.yaml` |
| `ref_nopress_dir` | Directory of no-contact reference images |
| `depth_from_normals` | Integrate depth from the recovered normal map |
| `smooth_grad_ksize` | Gradient smoothing kernel size before depth integration (odd) |
| `depth_flip_sign` | Flip the sign of the reconstructed depth |
| `contact_mask_enable` | Enable contact-region masking |
| `input_mode` | `"image_dir"` for batch processing; `"camera"` for live feed |
| `camera_id` | Camera device index for live mode (`0` = built-in) |
| `ref_frame_count` | Number of frames for building the reference baseline in live mode |
| `d_thred_h` / `d_thred_l` | High / low depth display thresholds (mm) for live mode |

</details>

<details>
<summary><strong>Output paths</strong> — <code>output</code></summary>

| Field | Description |
|---|---|
| `calib_out_dir` | Output directory for calibration artifacts |
| `solve_out_dir` | Output directory for reconstruction results |

</details>

## Usage

### Calibration

Generate the per-pixel RGB-to-normal LUT:

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode calib
```

### Reconstruction — Batch Mode

Set `solve.input_mode` to `"image_dir"`, then run:

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode solve
```

### Reconstruction — Live Camera

Set `solve.input_mode` to `"camera"` and configure `solve.camera_id`, then run:

```bash
python -m gelsight_ps.main --config configs/example_config.py --mode solve
```

The system first captures `ref_frame_count` frames to establish a reference
plane, then displays the depth map in real time. Press **`q`** to exit.

## Output

### Calibration Artifacts

```text
<calib_out_dir>/
├── rgb2n_lut.yaml          # LUT metadata (points to the weight file)
├── rgb2n_field.npz         # Per-pixel regression weight field
├── coverage.png            # Sample-count heatmap per pixel
├── debug/                  # Per-frame circle detection overlays and JSON data
└── geom_depth/             # Qualitative depth maps for visual inspection
```

### Reconstruction Results

```text
<solve_out_dir>/
├── depth/                  # .npy depth arrays (indentation minus reference)
├── normal/                 # .npy unit normal arrays
├── vis/                    # Visualization PNGs (depth and normal maps)
├── ref/                    # Reference depth and normal arrays
└── depth_diff_max_stats.json
```

## Notes

- **Circle detection is heuristic.** It works well for rapid prototyping;
  consider more robust contact-region segmentation for production use.
- **Lambertian model is approximate.** The GelSight Mini's near-field,
  multi-color illumination introduces BRDF and vignetting effects that the
  per-pixel LUT partially absorbs.
- **Frankot-Chellappa / Poisson integration** is provided for demonstration.
  Regularized shape-from-normal methods may yield better results in practice.
- **Per-pixel regression** currently uses a global shared model with 8x8
  block-wise residual correction. A fully per-pixel model is possible but
  significantly increases I/O overhead.

## License

This project is released under the [MIT License](LICENSE).

## Disclaimer

*This is not an officially supported Google product.*
