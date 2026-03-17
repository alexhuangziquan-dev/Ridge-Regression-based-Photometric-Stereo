import os
import shutil

def rename_images(src_dir, dst_dir, start_index=0):
    """
    将指定目录下的图片按顺序重新编号并保存到新目录
    :param src_dir: 原始图片目录
    :param dst_dir: 目标目录
    :param start_index: 起始编号（int，例如24表示从000024开始）
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 读取目录下的图片文件并排序
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, filename in enumerate(files):
        # 新编号（六位数）
        new_name = f"{start_index + i:06d}" + os.path.splitext(filename)[1]
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)

        # 复制并重命名
        shutil.copy2(src_path, dst_path)

    print(f"已处理 {len(files)} 张图片，保存到 {dst_dir}")

if __name__ == "__main__":
    # 示例：从 000024 开始编号
    src_dir = r"G:\chrome\gsrobotics-main\gsrobotics-main\baoding-bu\images"
    dst_dir = r"G:\chrome\gelsight_ps_ballcalib_minimal\gelsight_ps_ballcalib_minimal\data\calib_ball"
    start_num = 68 # 表示从 000024 开始

    rename_images(src_dir, dst_dir, start_index=start_num)
