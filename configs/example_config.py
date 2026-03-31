# -*- coding: utf-8 -*-
"""Example configuration for the gelsight_ps pipeline.

Copy this file and modify the paths and parameters to match your setup.
Only fields that the pipeline actually reads are included.
"""

CFG = {
    "camera": {
        "width": 512,
        "height": 512,
        "pixel_size_mm": 0.041015,
    },

    "preproc": {
        "dark_path": "",
        "gamma_inv": 1.0,
        "wb_gain": [1.0, 1.0, 1.0],
        "clip_eps": 1e-6,
    },

    "calib": {
        # Root directory with radius-named subdirectories (e.g. 1.0/, 2.0/).
        "image_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\calib_multi",

        "ridge_lambda": 1e-2,
        "min_samples_per_pixel": 4,
        "max_images": 0,
        "eval_export": True,

        "use_quadratic": True,
        "use_cross": True,

        # No-press reference images for baseline subtraction.
        "ref_nopress_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\plain",

        "rim_shrink_pix": 0.5,
        "nz_floor": 0.1,
        "out_dim": 3,

        "use_manual_radius_csv": False,
    },

    "solve": {
        # Directory of images to reconstruct.
        "image_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\4mm_3_3",
        # Path to the calibration LUT produced by the calib step.
        "lut_yaml": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\output\3dim-ref26314\out_calib\rgb2n_lut.yaml",

        # No-press reference images.
        "ref_nopress_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\plain",
        "ref_nopress_path": "",

        "depth_from_normals": True,
        "smooth_grad_ksize": 3,
        "depth_flip_sign": True,

        "contact_mask_enable": True,
        "cm_w_chroma": 0.7,
        "cm_w_int": 0.3,
        "cm_thr_rel": 0.25,
        "cm_min_area": 80,

        # Input mode: "image_dir" for batch file processing,
        # "camera" for real-time camera feed.
        "input_mode": "image_dir",
        # Camera device index (0 = built-in, 1/2 = external).
        "camera_id": 1,

        "camera_subtract_ref_plane": True,
        # Number of frames averaged to build the reference plane.
        "ref_frame_count": 50,

        # Depth display thresholds for camera mode (mm).
        "d_thred_h": 10,
        "d_thred_l": 5,
    },

    "output": {
        "calib_out_dir": "output/3dim-ref26314/out_calib",
        "solve_out_dir": "output/3dim-ref26314/out_solve2",
    },

    "manual_pick": {
        "draw_scale": 1.0,
        "min_points": 3,
    },

    "manual_radius_csv": {
        "csv_path": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\251111_data\2026-1-23real.csv",
        "unit": "mm",
    },
}
