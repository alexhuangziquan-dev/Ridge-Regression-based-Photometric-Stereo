"""Utility script to rename and renumber image files sequentially.

Copies images from a source directory to a destination directory, renaming
them with zero-padded six-digit indices starting from a configurable offset.
"""

import os
import shutil


def rename_images(src_dir: str, dst_dir: str, start_index: int = 0) -> None:
    """Copies and sequentially renumbers images from src_dir to dst_dir.

    Args:
        src_dir: Source directory containing the original images.
        dst_dir: Destination directory for the renamed copies.
        start_index: First index number (e.g. 24 produces ``000024.png``).
    """
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    for i, filename in enumerate(files):
        new_name = f"{start_index + i:06d}" + os.path.splitext(filename)[1]
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy2(src_path, dst_path)

    print(f"Processed {len(files)} images, saved to {dst_dir}")


if __name__ == "__main__":
    src_dir = r"G:\chrome\gsrobotics-main\gsrobotics-main\baoding-bu\images"
    dst_dir = r"G:\chrome\gelsight_ps_ballcalib_minimal\gelsight_ps_ballcalib_minimal\data\calib_ball"
    start_num = 68

    rename_images(src_dir, dst_dir, start_index=start_num)
