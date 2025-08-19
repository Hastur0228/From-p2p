import os
import sys
from pathlib import Path
import argparse
import numpy as np
import trimesh

# 为了复用已有的采样函数（FPS 与表面采样），将 project root 加入 sys.path
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_FILE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    # 复用已有的采样逻辑，保持与工程内其他流程一致
    from npy_to_xyz import farthest_point_sampling, sample_points_on_mesh  # type: ignore
except Exception:
    # 最小兜底：若导入失败，定义简化版 FPS 与采样
    def farthest_point_sampling(points: np.ndarray, num_samples: int, logger=None) -> np.ndarray:
        n = points.shape[0]
        if num_samples >= n:
            return np.arange(n, dtype=np.int64)
        start = int(np.random.randint(0, n))
        selected = np.empty(num_samples, dtype=np.int64)
        selected[0] = start
        diff = points - points[start]
        min_d2 = np.einsum('ij,ij->i', diff, diff)
        for i in range(1, num_samples):
            j = int(np.argmax(min_d2))
            selected[i] = j
            diff = points - points[j]
            d2 = np.einsum('ij,ij->i', diff, diff)
            min_d2 = np.minimum(min_d2, d2)
        return selected

    def sample_points_on_mesh(mesh: trimesh.Trimesh, desired_points: int, candidate_multiplier: int = 8, logger=None):
        m = max(desired_points * max(1, candidate_multiplier), desired_points * 2)
        pts, face_ids = trimesh.sample.sample_surface(mesh, m)
        nors = mesh.face_normals[face_ids]
        keep = farthest_point_sampling(pts, desired_points, logger)
        return pts[keep], nors[keep]


def _load_points_from_file(path: Path) -> np.ndarray:
    """
    读取 .npy/.npz 中的点云，优先使用键 'points'；若无则读取第一个数组。
    仅返回 (N, 3) 的坐标。
    """
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(str(path))
    elif suffix == '.npz':
        with np.load(str(path)) as data:
            key = 'points' if 'points' in data.files else (data.files[0] if data.files else None)
            if key is None:
                raise ValueError(f"NPZ 内无数组: {path}")
            arr = data[key]
    else:
        raise ValueError(f"不支持的文件类型: {path}")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"数组形状无效: {path} shape={arr.shape}")
    return arr[:, :3].astype(np.float64, copy=False)


def _rasterize_heightmap(points_xyz: np.ndarray,
                         grid_x: int,
                         grid_y: int,
                         x_min: float,
                         x_max: float,
                         y_min: float,
                         y_max: float,
                         min_points_per_cell: int = 1) -> np.ndarray:
    """
    将点云投影到规则网格，按网格单元对 z 求平均，得到高度图。
    - 对于样本不足的格元，填充为 NaN，稍后统一做邻域填充。
    返回 H (grid_y, grid_x)。
    """
    # 计算每个点所在的格元索引
    # 注意：将连续坐标映射到 [0, grid-1] 的闭区间，然后截断
    px = points_xyz[:, 0]
    py = points_xyz[:, 1]
    pz = points_xyz[:, 2]

    # 避免除零
    dx = (x_max - x_min) if (x_max - x_min) > 1e-12 else 1.0
    dy = (y_max - y_min) if (y_max - y_min) > 1e-12 else 1.0

    ix = np.floor((px - x_min) / dx * (grid_x - 1)).astype(np.int64)
    iy = np.floor((py - y_min) / dy * (grid_y - 1)).astype(np.int64)
    ix = np.clip(ix, 0, grid_x - 1)
    iy = np.clip(iy, 0, grid_y - 1)

    # 聚合 z 到格元
    z_sum = np.zeros((grid_y, grid_x), dtype=np.float64)
    z_cnt = np.zeros((grid_y, grid_x), dtype=np.int64)
    # 累加
    np.add.at(z_sum, (iy, ix), pz)
    np.add.at(z_cnt, (iy, ix), 1)

    # 计算平均值
    H = np.full((grid_y, grid_x), np.nan, dtype=np.float64)
    mask = z_cnt >= min_points_per_cell
    H[mask] = z_sum[mask] / np.maximum(1, z_cnt[mask])
    return H


def _fill_nan_by_neighbor_mean(heightmap: np.ndarray, max_iters: int = 64) -> np.ndarray:
    """
    使用 8 邻域均值迭代填充 NaN。最多迭代 max_iters 次，或直至无 NaN。
    若仍残留 NaN，最终用 0 填充。
    """
    H = heightmap.copy()
    gy, gx = H.shape
    # 邻域偏移（8 邻域）
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1)]

    for _ in range(max_iters):
        nan_mask = np.isnan(H)
        if not np.any(nan_mask):
            break
        neighbor_sum = np.zeros_like(H)
        neighbor_cnt = np.zeros_like(H)
        for dy, dx in offsets:
            sl_y_src = slice(max(0, -dy), min(gy, gy - dy))
            sl_x_src = slice(max(0, -dx), min(gx, gx - dx))
            sl_y_dst = slice(max(0, dy), min(gy, gy + dy))
            sl_x_dst = slice(max(0, dx), min(gx, gx + dx))
            nb = H[sl_y_src, sl_x_src]
            nb_valid = ~np.isnan(nb)
            neighbor_sum[sl_y_dst, sl_x_dst] += np.where(nb_valid, nb, 0.0)
            neighbor_cnt[sl_y_dst, sl_x_dst] += nb_valid.astype(np.int32)

        # 仅填充那些原本是 NaN 且邻居中有有效值的位置
        can_fill = nan_mask & (neighbor_cnt > 0)
        H[can_fill] = neighbor_sum[can_fill] / neighbor_cnt[can_fill]

    # 仍残留的 NaN 用 0 填充（对应无数据区域）
    H[np.isnan(H)] = 0.0
    return H


def _build_mesh_from_heightmap(H: np.ndarray,
                               x_min: float,
                               x_max: float,
                               y_min: float,
                               y_max: float) -> trimesh.Trimesh:
    """
    根据高度图构建规则网格三角面片：
    - 顶点排列为行优先 (y 维在前，x 维在后)
    - 每个网格单元分解为两个三角形
    """
    gy, gx = H.shape
    xs = np.linspace(x_min, x_max, gx, dtype=np.float64)
    ys = np.linspace(y_min, y_max, gy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)  # 形状 (gy, gx)

    vertices = np.stack([XX.reshape(-1), YY.reshape(-1), H.reshape(-1)], axis=1)

    faces = []
    for j in range(gy - 1):
        base0 = j * gx
        base1 = (j + 1) * gx
        for i in range(gx - 1):
            v00 = base0 + i
            v01 = base0 + i + 1
            v10 = base1 + i
            v11 = base1 + i + 1
            # 两个三角形（保持右手坐标顺序，Z 由高度定义）
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def find_insole_files(insole_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, names in os.walk(insole_dir):
        for n in names:
            nl = n.lower()
            if nl.endswith('.npy') or nl.endswith('.npz'):
                files.append(Path(root) / n)
    return sorted(files)


def filter_side(files: list[Path], side: str | None) -> list[Path]:
    if side is None:
        return files
    s = side.upper()
    if s not in ("L", "R"):
        raise ValueError("side 必须是 'L' 或 'R' 或 None")
    suffix = f"_{s.lower()}."
    return [f for f in files if suffix in f.name.lower()]


def compute_average_insole(
    insole_dir: Path,
    grid_x: int,
    grid_y: int,
    min_points_per_cell: int,
    side: str | None = None,
):
    """
    从多个鞋垫点云生成平均高度图与对应三角网格。
    返回 (mesh, H_avg, (x_min,x_max,y_min,y_max))。
    """
    files = filter_side(find_insole_files(insole_dir), side)
    if not files:
        raise FileNotFoundError(f"未找到点云文件: {insole_dir} (side={side})")

    # 统计全局 XY 范围（点云已在项目流程中做过归一化/对齐，这里仍按数据实际范围计算）
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    clouds: list[np.ndarray] = []
    for fp in files:
        P = _load_points_from_file(fp)
        clouds.append(P)
        x_min = min(x_min, float(P[:, 0].min()))
        x_max = max(x_max, float(P[:, 0].max()))
        y_min = min(y_min, float(P[:, 1].min()))
        y_max = max(y_max, float(P[:, 1].max()))

    # 若范围过小，给定容差，避免退化
    if x_max - x_min < 1e-6:
        x_min, x_max = -1.0, 1.0
    if y_max - y_min < 1e-6:
        y_min, y_max = -1.0, 1.0

    # 对每个鞋垫生成高度图，再做逐点平均（忽略 NaN）
    H_sum = np.zeros((grid_y, grid_x), dtype=np.float64)
    H_cnt = np.zeros((grid_y, grid_x), dtype=np.int64)

    for P in clouds:
        H = _rasterize_heightmap(P, grid_x, grid_y, x_min, x_max, y_min, y_max, min_points_per_cell)
        H = _fill_nan_by_neighbor_mean(H, max_iters=64)
        valid = ~np.isnan(H)
        H_sum[valid] += H[valid]
        H_cnt[valid] += 1

    # 求平均高度图
    H_avg = np.zeros_like(H_sum)
    nonzero = H_cnt > 0
    H_avg[nonzero] = H_sum[nonzero] / np.maximum(1, H_cnt[nonzero])
    # 若某处从未被任何鞋垫覆盖，则高度保持为 0

    # 构建平均网格
    mesh = _build_mesh_from_heightmap(H_avg, x_min, x_max, y_min, y_max)
    return mesh, H_avg, (x_min, x_max, y_min, y_max)


def save_outputs(mesh: trimesh.Trimesh,
                 H: np.ndarray,
                 bounds: tuple[float, float, float, float],
                 output_dir: Path,
                 sampled_points: int,
                 candidate_multiplier: int = 8,
                 side: str | None = None):
    """
    保存 STL、高度图，以及从平均网格采样得到的模板点云（含法向量）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 侧别后缀（若指定 L/R 则追加 _L/_R，以避免覆盖）
    suffix = ''
    if side is not None:
        s = side.upper()
        if s in ('L', 'R'):
            suffix = f'_{s}'

    # 1) 保存 STL
    stl_path = output_dir / f'average_insole_mesh{suffix}.stl'
    mesh.export(stl_path)

    # 2) 采样模板点云（均匀）
    pts, nors = sample_points_on_mesh(mesh, desired_points=sampled_points, candidate_multiplier=candidate_multiplier)
    template_path = output_dir / f'average_insole_template{suffix}.npz'
    np.savez_compressed(template_path, points=pts.astype(np.float32), normals=nors.astype(np.float32))

    # 3) 备份保存高度图及网格 bounds，便于后续复用或可视化
    hm_path = output_dir / f'average_insole_heightmap{suffix}.npz'
    x_min, x_max, y_min, y_max = bounds
    np.savez_compressed(hm_path,
                        height=H.astype(np.float32),
                        x_min=np.float32(x_min), x_max=np.float32(x_max),
                        y_min=np.float32(y_min), y_max=np.float32(y_max))

    return {
        'stl': stl_path,
        'template': template_path,
        'heightmap': hm_path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='从多个鞋垫点云生成平均鞋垫网格，并从中采样模板点云保存到 Templates。'
    )
    parser.add_argument('--insoles-dir', type=str, default=str(Path('data') / 'pointcloud' / 'insoles'),
                        help='鞋垫点云目录（支持 .npy/.npz）。默认: data/pointcloud/insoles')
    parser.add_argument('--grid-x', type=int, default=256, help='高度图 X 方向分辨率（列数），默认 256')
    parser.add_argument('--grid-y', type=int, default=128, help='高度图 Y 方向分辨率（行数），默认 128')
    parser.add_argument('--min-per-cell', type=int, default=1, help='每格最少点数阈值（不足则视为缺失），默认 1')
    parser.add_argument('--sampled-points', type=int, default=4096, help='从平均网格采样的点数，默认 4096')
    parser.add_argument('--candidate-multiplier', type=int, default=8, help='表面候选采样倍数（用于 FPS 前的候选点集），默认 8')
    parser.add_argument('--output-dir', type=str, default=str(Path('Templates')),
                        help='输出目录（将写出 STL、模板点云和高度图），默认 Templates')
    parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR',
                        help='侧别：L 或 R 或 LR（左右都生成，依次 L 后 R）。默认 LR')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    insole_dir = Path(args.insoles_dir)
    output_dir = Path(args.output_dir)

    side_arg = (args.side or '').upper()
    sides_to_run = ['L', 'R'] if side_arg == 'LR' else [side_arg] if side_arg in ('L', 'R') else []
    if not sides_to_run:
        raise SystemExit("--side 必须是 L/R/LR")

    for s in sides_to_run:
        print(f"开始生成侧别: {s}")
        mesh, H_avg, bounds = compute_average_insole(
            insole_dir=insole_dir,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            min_points_per_cell=args.min_per_cell,
            side=s,
        )

        paths = save_outputs(
            mesh=mesh,
            H=H_avg,
            bounds=bounds,
            output_dir=output_dir,
            sampled_points=args.sampled_points,
            candidate_multiplier=args.candidate_multiplier,
            side=s,
        )

        print(f"{s} 侧平均鞋垫已生成:")
        print(f"  STL: {paths['stl']}")
        print(f"  模板点云: {paths['template']}")
        print(f"  高度图: {paths['heightmap']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


