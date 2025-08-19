import argparse
import os
from typing import Tuple, List

import numpy as np
import trimesh


def load_pointcloud(path: str) -> np.ndarray:
    """
    加载点云文件，支持 .npy / .npz / .stl。
    - 若为 (N,6) 数据，将只取前 3 维 xyz。
    返回: (N,3) float32
    """
    path_lower = path.lower()
    if path_lower.endswith('.npy'):
        pts = np.load(path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f'invalid array shape in npy: {pts.shape}')
        pts = pts[:, :3].astype(np.float32)
        return pts
    if path_lower.endswith('.npz'):
        with np.load(path) as data:
            if 'points' in data.files:
                pts = data['points']
            else:
                # 取第一个数组作为点云
                pts = data[data.files[0]]
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f'invalid array shape in npz: {pts.shape}')
        pts = pts[:, :3].astype(np.float32)
        return pts
    if path_lower.endswith('.stl'):
        mesh = trimesh.load_mesh(path)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError('stl vertices must be (N,3)')
        return verts
    raise ValueError('Unsupported format: only .npy, .npz and .stl')


def compute_stats(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    计算点云基础统计：min/max/centroid/最大半径/最大绝对坐标。
    返回: (pmin, pmax, centroid, max_radius, max_abs)
    """
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    centroid = points.mean(axis=0)
    max_radius = float(np.max(np.linalg.norm(points - centroid, axis=1)))
    max_abs = float(np.max(np.abs(points)))
    return pmin, pmax, centroid, max_radius, max_abs


def check_normalization(points: np.ndarray, mode: str = 'sphere', atol_centroid: float = 0.05) -> bool:
    """
    检查是否标准化：
    - sphere: 质心接近 0，最大半径 ~ 1
    - cube:   质心接近 0，坐标绝对值最大 ~ 1（范围基本在 [-1, 1]）
    """
    pmin, pmax, centroid, max_radius, max_abs = compute_stats(points)
    print('点云信息:')
    print(f'  坐标范围: min={pmin}, max={pmax}')
    print(f'  质心: {centroid}')
    print(f'  最大半径 (for sphere): {max_radius:.6f}')
    print(f'  最大绝对坐标 (for cube): {max_abs:.6f}')

    center_ok = np.allclose(centroid, 0.0, atol=atol_centroid)
    if mode == 'sphere':
        scale_ok = (0.9 <= max_radius <= 1.1)
        if center_ok and scale_ok:
            print('✅ 该点云已基本标准化 (居中 + 缩放到单位球)')
            return True
        else:
            print('⚠️ 该点云未标准化为单位球，建议进行处理')
            return False
    elif mode == 'cube':
        scale_ok = (0.9 <= max_abs <= 1.1)
        if center_ok and scale_ok:
            print('✅ 该点云已基本标准化 (居中 + 缩放到单位立方)')
            return True
        else:
            print('⚠️ 该点云未标准化为单位立方，建议进行处理')
            return False
    else:
        raise ValueError("mode must be 'sphere' or 'cube'")


def _gather_paths(inputs: List[str], recursive: bool) -> List[str]:
    """
    将输入的文件/目录展开为文件列表。仅保留 .npy/.npz/.stl。
    - inputs: 路径列表，既可以是文件也可以是目录
    - recursive: 若为目录，是否递归遍历
    """
    exts = {'.npy', '.npz', '.stl'}
    files: List[str] = []
    for p in inputs:
        if not os.path.exists(p):
            print(f'跳过: 路径不存在 -> {p}')
            continue
        if os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in exts:
                files.append(p)
            else:
                print(f'跳过: 不支持的文件类型 -> {p}')
        else:
            if recursive:
                for root, _, fnames in os.walk(p):
                    for fn in fnames:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in exts:
                            files.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(p):
                    full = os.path.join(p, fn)
                    if os.path.isfile(full) and os.path.splitext(full)[1].lower() in exts:
                        files.append(full)
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description='检查点云是否已标准化到单位球/单位立方')
    # 允许省略路径，运行后交互式输入；支持一次输入多个文件或目录
    parser.add_argument('paths', type=str, nargs='*', default=None, help='点云文件或目录路径（可多个）')
    parser.add_argument('--mode', type=str, choices=['sphere', 'cube'], default='sphere',
                        help='检查标准: sphere=单位球, cube=单位立方 (默认 sphere)')
    parser.add_argument('--atol-centroid', type=float, default=0.05, help='质心接近 0 的容差')
    parser.add_argument('--recursive', action='store_true', help='当输入为目录时，递归遍历子目录')
    args = parser.parse_args()

    if not args.paths:
        try:
            line = input('请输入点云路径（可多个，空格分隔，支持文件或目录）：').strip()
        except EOFError:
            line = ''
        if not line:
            raise SystemExit('未提供路径，已退出。')
        args.paths = line.split()

    files = _gather_paths(args.paths, recursive=args.recursive)
    if not files:
        raise SystemExit('未找到任何待检查的点云文件（支持 .npy/.npz/.stl）。')

    total = 0
    ok = 0
    for f in files:
        print('=' * 60)
        print(f'文件: {f}')
        try:
            pts = load_pointcloud(f)
            res = check_normalization(pts, mode=args.mode, atol_centroid=args.atol_centroid)
            ok += int(res)
        except Exception as e:
            print(f'❌ 处理失败: {e}')
        total += 1

    print('=' * 60)
    print(f'完成。总计: {total} | 通过: {ok} | 未通过/失败: {total - ok}')


if __name__ == '__main__':
    main()


