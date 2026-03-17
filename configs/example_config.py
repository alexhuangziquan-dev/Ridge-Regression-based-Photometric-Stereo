# -*- coding: utf-8 -*-
"""
最终精简版配置（仅保留你的代码真实会读取的字段）
保持你原本填写的所有路径不变。
"""

CFG = {
    "camera": {
        "width":  512,
        "height": 512,
        "pixel_size_mm": 0.041015,#0.0315
    },

    "preproc": {
        "dark_path": "",
        "gamma_inv": 1.0,
        "wb_gain": [1.0, 1.0, 1.0],
        "clip_eps": 1e-6,
    },

    "calib": {
        "image_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\calib_multi",  # 子目录以半径命名，如 1.0/ 2.0/

        "ridge_lambda": 1e-2,
        "min_samples_per_pixel": 4,
        "max_images": 0,
        "eval_export": True,

        "use_quadratic": True,
        "use_cross": True,

        "ref_nopress_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\plain",

        "rim_shrink_pix": 0.5,
        "nz_floor": 0.1,
        "out_dim": 3,

        "use_manual_radius_csv": False,
    },

    "solve": {
        "image_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\test_data\4mm_3_3",#r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\251111_data\251111\1mm_25_5"
        "lut_yaml":  r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\output\3dim-ref26314\out_calib\rgb2n_lut.yaml",

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

        # 新增：输入模式开关，image_dir=原文件模式，camera=新实时模式
        "input_mode" : "image_dir",
        # 新增：摄像头设备号，默认0（内置摄像头），外接摄像头可改为1/2等
        "camera_id": 1,

        "camera_subtract_ref_plane" : True ,
        "ref_frame_count" : 50,

        "d_thred_h" : 10,
        "d_thred_l" : 5,
    },

    "output": {
        "calib_out_dir": "output/3dim-ref26314/out_calib",
        "solve_out_dir": "output/3dim-ref26314/out_solve2",
    },

    "manual_pick": {
        # "reuse_from_circles_dir":
        #     r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\output\2dim_00315\out_calib\debug\circles",
        # "reuse_from_debug_dir": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\output\3dim-ref99\out_calib\debug",
        "draw_scale": 1.0,
        "min_points": 3,
    },

    "manual_radius_csv": {
        "csv_path": r"F:\PS_GEL\gelsight_ps_ballcalib_minimal\251111_data\2026-1-23real.csv",
        "unit": "mm",
    },
}