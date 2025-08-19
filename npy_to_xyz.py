import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

# 兼容直接脚本运行路径
import os as _os
import sys as _sys
_FILE_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJECT_ROOT = _os.path.dirname(_FILE_DIR)
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)


def setup_logging(name: str, log_dir: str = 'Log') -> logging.Logger:
    """
    创建并返回一个 Logger：
    - 同时输出到控制台与文件 `Log/<name>.log`
    - 文件以覆盖写入（mode='w'），UTF-8 编码
    - 避免重复添加 handler
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # 文件日志（覆盖写）
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 控制台日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _load_points_and_normals_from_file(file_path: str, logger) -> tuple[np.ndarray, np.ndarray | None]:
    """
    从 .npy 或 .npz 文件中读取点阵与法向量。

    返回 (points_xyz, normals)；若不存在法向量，第二个返回值为 None。

    - .npy: 直接读取数组，要求二维且列数>=3；不包含 normals
    - .npz: 优先读取键 'points'；不存在则读取第一个数组；若包含键 'normals' 则一并返回
    """
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == '.npy':
        arr = np.load(file_path)
        normals = None
    elif suffix == '.npz':
        with np.load(file_path) as data:
            key = 'points' if 'points' in data.files else (data.files[0] if data.files else None)
            if key is None:
                raise ValueError("NPZ 文件中未找到任何数组")
            arr = data[key]
            normals = data['normals'] if 'normals' in data.files else None
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"形状无效: shape={arr.shape}")
    points_xyz = arr[:, :3].astype(np.float64, copy=False)

    if normals is not None:
        if normals.ndim != 2 or normals.shape[1] < 3 or normals.shape[0] != points_xyz.shape[0]:
            logger.warning(
                f"  跳过法向量（形状不匹配）: normals.shape={None if normals is None else normals.shape} vs points={points_xyz.shape}"
            )
            normals = None
        else:
            normals = normals[:, :3].astype(np.float64, copy=False)

    return points_xyz, normals


def npy_to_xyz(
    input_dir: str,
    output_dir: str,
    logger,
    include_normals: bool = False,
    generate_template: bool = False,
) -> None:
    """
    将输入目录下的所有 .npy 点云文件转换为 .xyz 文本文件，仅输出前3列 (x, y, z)。

    - 递归遍历 input_dir
    - 对每个 .npy 文件，读取为数组，取前3列作为坐标
    - 在 output_dir 中按相对路径写出 .xyz（保留子目录结构）
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    input_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            f_lower = fname.lower()
            if f_lower.endswith('.npy') or f_lower.endswith('.npz'):
                input_files.append(os.path.join(root, fname))

    if not input_files:
        logger.warning(f"在目录 {input_dir} 中未找到 .npy/.npz 文件")
        return

    num_npy = 0
    num_npz = 0
    total_points = 0
    files_with_normals = 0
    templates_written = 0

    for src_path in tqdm(input_files, desc="转换 npy/npz -> xyz"):
        try:
            points_xyz, normals = _load_points_and_normals_from_file(src_path, logger)
            if src_path.lower().endswith('.npy'):
                num_npy += 1
            else:
                num_npz += 1

            rel_path = os.path.relpath(src_path, input_dir)
            rel_base, _ = os.path.splitext(rel_path)
            out_path = os.path.join(output_dir, rel_base + '.xyz')

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # 组合输出：根据 include_normals 与数据可用性决定是否写 6 列
            if include_normals and normals is not None:
                output_data = np.hstack([points_xyz, normals])
                files_with_normals += 1
            else:
                if include_normals and normals is None:
                    logger.warning("  源文件未提供 normals，退回仅写出 xyz")
                output_data = points_xyz

            # 写出为 ascii 文本
            fmt = '%.8f'
            np.savetxt(out_path, output_data, fmt=fmt)
            num_points = int(points_xyz.shape[0])
            total_points += num_points
            if output_data.shape[1] == 6:
                logger.info(f"  写出: {out_path} | 点数={num_points} | 含法向量")
            else:
                logger.info(f"  写出: {out_path} | 点数={num_points}")

            # 若需要生成模板鞋垫：对 insole 文件生成扁平模板（z=0；法向量为 0,0,1）
            is_insole = ('insole' in os.path.basename(rel_base).lower()) or ('insoles' in rel_path.replace('\\', '/').lower())
            if generate_template and is_insole:
                template_points = points_xyz.copy()
                # 扁平化到 z=0
                template_points[:, 2] = 0.0

                if include_normals and normals is not None:
                    template_normals = np.zeros_like(template_points)
                    template_normals[:, 2] = 1.0  # 指向 +Z
                    template_output = np.hstack([template_points, template_normals])
                else:
                    template_output = template_points

                template_out_path = os.path.join(output_dir, 'templates', rel_base + '.xyz')
                os.makedirs(os.path.dirname(template_out_path), exist_ok=True)
                np.savetxt(template_out_path, template_output, fmt=fmt)
                templates_written += 1
                logger.info(
                    f"  模板写出: {template_out_path} | 点数={template_output.shape[0]}"
                    + (" | 含法向量" if template_output.shape[1] == 6 else "")
                )
        except Exception as e:
            logger.error(f"  转换失败: {src_path} -> {e}")

    logger.info(
        f"完成：共处理 {len(input_files)} 个文件（npy={num_npy}, npz={num_npz}），总点数={total_points}，"
        f"含法向量文件数={files_with_normals}，模板鞋垫数量={templates_written}，输出目录: {output_dir}"
    )


def _prompt_input(prompt: str, default: str | None = None) -> str:
    """简单命令行输入封装，支持默认值（回车选择）。"""
    if default:
        user = input(f"{prompt} [默认: {default}]: ").strip()
        return user or default
    return input(f"{prompt}: ").strip()


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Y/N 提示，默认值由 default_yes 控制。"""
    default_hint = 'Y/n' if default_yes else 'y/N'
    while True:
        ans = input(f"{prompt} ({default_hint}): ").strip().lower()
        if not ans:
            return default_yes
        if ans in ('y', 'yes'):
            return True
        if ans in ('n', 'no'):
            return False
        print("请输入 y 或 n。")


def main():
    # 设置日志
    logger = setup_logging("npy_to_xyz")
    
    parser = argparse.ArgumentParser(description="将指定目录下的 .npy 点云批量转换为 .xyz（仅 x y z 列）")
    parser.add_argument('--input', type=str, default=os.path.join('data', 'pointcloud'),
                        help='输入根目录（默认: data/pointcloud）')
    parser.add_argument('--output', type=str, default='xyz_output',
                        help='输出根目录（默认: xyz）')
    parser.add_argument('--no-interactive', action='store_true',
                        help='关闭交互式模式，直接使用命令行参数执行')
    parser.add_argument('--with-normals', action='store_true',
                        help='若源含 normals，则将法向量一并写出为 6 列 (x y z nx ny nz)')
    parser.add_argument('--generate-template', action='store_true',
                        help='为鞋垫(insole)生成扁平模板 (z=0；若含法向量则为 0,0,1)，输出到 <output>/templates')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    interactive = not args.no_interactive
    include_normals = args.with_normals
    generate_template = args.generate_template

    if interactive:
        print("=== 交互模式：npy -> xyz 转换 ===")
        # 输入目录
        input_dir = _prompt_input("请选择输入目录（递归查找 .npy）", default=input_dir)
        while not os.path.isdir(input_dir):
            print(f"目录不存在: {input_dir}")
            input_dir = _prompt_input("请重新输入输入目录", default=None)

        # 输出目录
        output_dir = _prompt_input("请选择输出目录（将生成 .xyz 文件）", default=output_dir)
        if not output_dir:
            output_dir = 'xyz_output'

        # 是否包含法向量
        include_normals = _prompt_yes_no("若源含 normals，是否一并写出?", default_yes=True)

        # 是否生成模板（默认: 否）
        generate_template = _prompt_yes_no("是否为鞋垫生成扁平模板?", default_yes=False)

        print("\n配置预览：")
        print(f"  输入目录: {input_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  含法向量: {'是' if include_normals else '否'}")
        print(f"  生成模板: {'是' if generate_template else '否'}")
        if not _prompt_yes_no("确认执行?", default_yes=True):
            print("已取消。")
            return

    npy_to_xyz(
        input_dir,
        output_dir,
        logger,
        include_normals=include_normals,
        generate_template=generate_template,
    )


if __name__ == '__main__':
    main()


