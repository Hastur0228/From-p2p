import os
import sys
from pathlib import Path
import argparse
import numpy as np
import trimesh
try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    MiniBatchKMeans = None  # type: ignore
    _HAS_SKLEARN = False

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
    - 仅使用有效区域（H 中非 NaN 的顶点）。任何包含无效顶点的三角形都会被跳过。
    """
    gy, gx = H.shape
    xs = np.linspace(x_min, x_max, gx, dtype=np.float64)
    ys = np.linspace(y_min, y_max, gy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)  # 形状 (gy, gx)

    # 顶点：对无效高度（NaN）先填 0，但随后会通过面片筛选排除
    H_valid_mask = ~np.isnan(H)
    H_filled = np.nan_to_num(H, nan=0.0)
    vertices = np.stack([XX.reshape(-1), YY.reshape(-1), H_filled.reshape(-1)], axis=1)

    faces = []
    for j in range(gy - 1):
        base0 = j * gx
        base1 = (j + 1) * gx
        for i in range(gx - 1):
            v00 = base0 + i
            v01 = base0 + i + 1
            v10 = base1 + i
            v11 = base1 + i + 1

            # 四个格点的有效性
            if not (H_valid_mask[j, i] and H_valid_mask[j, i + 1] and H_valid_mask[j + 1, i] and H_valid_mask[j + 1, i + 1]):
                continue

            # 两个三角形（保持右手坐标顺序，Z 由高度定义）
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh


def find_np_files(root_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, names in os.walk(root_dir):
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


def compute_average_direct_points(
    root_dir: Path,
    side: str,
    num_points: int,
    seed: int | None = None,
    max_iter: int = 200,
    batch_size: int = 10000,
    n_init: int = 3,
) -> np.ndarray:
    """
    直接在三维空间对多个鞋垫点云取“平均模板”（不投影）：
    - 收集所有鞋垫点云（指定侧别）并拼接
    - 通过 MiniBatchKMeans 将全部点聚成 num_points 个簇
    - 以簇中心作为模板点（3D）
    注意：该方法仅由输入点云的联合支撑决定，不涉及无效区域。
    """
    files = filter_side(find_np_files(root_dir), side)
    if not files:
        raise FileNotFoundError(f"未找到点云文件: {root_dir} (side={side})")

    clouds: list[np.ndarray] = []
    for fp in files:
        P = _load_points_from_file(fp)
        clouds.append(P)
    all_points = np.vstack(clouds).astype(np.float64, copy=False)
    if all_points.shape[0] < num_points:
        # 若总点数少于目标点数，则重复补齐以避免 KMeans 失败
        repeat = int(np.ceil(num_points / max(1, all_points.shape[0])))
        all_points = np.tile(all_points, (repeat, 1))

    if _HAS_SKLEARN:
        rng = None if seed is None else int(seed)
        kmeans = MiniBatchKMeans(
            n_clusters=int(num_points),
            init='k-means++',
            n_init=max(1, int(n_init)),
            random_state=rng,
            max_iter=max(1, int(max_iter)),
            batch_size=max(1000, int(batch_size)),
            verbose=False,
        )
        kmeans.fit(all_points)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
        return centers
    # Fallback: 体素平均 + FPS 精简
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-9)
    n_per_axis = max(1, int(round(np.cbrt(float(num_points)))))
    voxel_size = ranges / n_per_axis
    # 避免过小体素导致过多单元
    voxel_size = np.maximum(voxel_size, ranges / max(1, n_per_axis))
    idx = np.floor((all_points - mins) / voxel_size).astype(np.int64)
    # 聚合
    from collections import defaultdict
    sum_map: dict[tuple[int, int, int], np.ndarray] = {}
    cnt_map: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for p, key in zip(all_points, map(tuple, idx)):
        if key in sum_map:
            sum_map[key] += p
        else:
            sum_map[key] = p.copy()
        cnt_map[key] += 1
    centers = np.vstack([sum_map[k] / max(1, cnt_map[k]) for k in sum_map.keys()])
    # 调整到目标点数
    M = centers.shape[0]
    if M > num_points:
        keep = farthest_point_sampling(centers, num_points)
        centers = centers[keep]
    elif M < num_points:
        reps = int(np.ceil(num_points / max(1, M)))
        centers = np.tile(centers, (reps, 1))[:num_points]
    return centers


def compute_average_heightmap(
    root_dir: Path,
    grid_x: int,
    grid_y: int,
    min_points_per_cell: int,
    side: str | None = None,
    min_coverage_count: int = 1,
):
    """
    从多个鞋垫点云生成平均高度图与对应三角网格。
    返回 (mesh, H_avg, (x_min,x_max,y_min,y_max))。
    """
    files = filter_side(find_np_files(root_dir), side)
    if not files:
        raise FileNotFoundError(f"未找到点云文件: {root_dir} (side={side})")

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

    # 对每个鞋垫生成高度图，再做逐点平均（严格忽略无效区域，不进行邻域填充）
    H_sum = np.zeros((grid_y, grid_x), dtype=np.float64)
    H_cnt = np.zeros((grid_y, grid_x), dtype=np.int64)

    for P in clouds:
        H = _rasterize_heightmap(P, grid_x, grid_y, x_min, x_max, y_min, y_max, min_points_per_cell)
        valid = ~np.isnan(H)
        H_sum[valid] += H[valid]
        H_cnt[valid] += 1

    # 求平均高度图（仅在覆盖次数达到阈值处求平均，其他位置保持 NaN 表示无效区域）
    H_avg = np.full_like(H_sum, np.nan, dtype=np.float64)
    enough = H_cnt >= int(max(1, min_coverage_count))
    H_avg[enough] = H_sum[enough] / np.maximum(1, H_cnt[enough])

    # 构建平均网格
    mesh = _build_mesh_from_heightmap(H_avg, x_min, x_max, y_min, y_max)
    return mesh, H_avg, (x_min, x_max, y_min, y_max)


def save_outputs(mesh: trimesh.Trimesh,
                 H: np.ndarray,
                 bounds: tuple[float, float, float, float],
                 output_dir: Path,
                 sampled_points: int,
                 candidate_multiplier: int = 8,
                 side: str | None = None,
                 category_label: str = 'insole'):
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
    stl_path = output_dir / f'average_{category_label}_mesh{suffix}.stl'
    mesh.export(stl_path)

    # 2) 采样模板点云（均匀）
    pts, nors = sample_points_on_mesh(mesh, desired_points=sampled_points, candidate_multiplier=candidate_multiplier)
    template_path = output_dir / f'average_{category_label}_template{suffix}.npz'
    np.savez_compressed(template_path, points=pts.astype(np.float32), normals=nors.astype(np.float32))

    # 3) 备份保存高度图及网格 bounds，便于后续复用或可视化
    hm_path = output_dir / f'average_{category_label}_heightmap{suffix}.npz'
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
        description='从多个点云生成平均模板（支持 insoles 或 feet），并导出模板/高度图。'
    )
    parser.add_argument('--category', type=str, choices=['insoles', 'feet'], default=None,
                        help='平均的类别：insoles 或 feet；若不提供，将进入交互选择')
    parser.add_argument('--root-dir', type=str, default=str(Path('data') / 'pointcloud'),
                        help='点云根目录，脚本会在其中寻找对应类别子目录（insoles/feet）')
    parser.add_argument('--method', type=str, choices=['direct', 'heightmap'], default='direct',
                        help='模板生成方法：direct=点云直接聚类平均（不投影），heightmap=投影到高度图后建网格。默认 direct')
    # direct 方法参数
    parser.add_argument('--sampled-points', type=int, default=4096, help='direct: 期望模板点数（聚类簇数），默认 4096')
    parser.add_argument('--seed', type=int, default=None, help='direct: 随机种子')
    parser.add_argument('--kmeans-max-iter', type=int, default=200, help='direct: KMeans 最大迭代次数，默认 200')
    parser.add_argument('--kmeans-batch-size', type=int, default=10000, help='direct: MiniBatchKMeans 批大小，默认 10000')
    parser.add_argument('--kmeans-n-init', type=int, default=3, help='direct: 不同初始的尝试次数，默认 3')
    # heightmap 方法参数（仅在选择 heightmap 时生效）
    parser.add_argument('--grid-x', type=int, default=256, help='heightmap: X 方向分辨率（列数），默认 256')
    parser.add_argument('--grid-y', type=int, default=128, help='heightmap: Y 方向分辨率（行数），默认 128')
    parser.add_argument('--min-per-cell', type=int, default=1, help='heightmap: 每格最少点数阈值（不足则视为缺失），默认 1')
    parser.add_argument('--min-coverage-count', type=int, default=1, help='heightmap: 一个网格位置至少被多少个鞋垫覆盖才算有效，默认 1')
    parser.add_argument('--candidate-multiplier', type=int, default=8, help='heightmap: 表面候选采样倍数（用于 FPS 前的候选点集），默认 8')
    parser.add_argument('--output-dir', type=str, default=str(Path('Templates')),
                        help='输出目录（将写出 STL、模板点云和高度图），默认 Templates')
    parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR',
                        help='侧别：L 或 R 或 LR（左右都生成，依次 L 后 R）。默认 LR')
    return parser.parse_args(argv)


def _interactive_choose_category(default: str = 'insoles') -> str:
    print('请选择平均类别:')
    print('  [1] insoles (默认)')
    print('  [2] feet')
    choice = input('选择 1 或 2: ').strip()
    if choice == '2':
        return 'feet'
    return default


def main(argv=None) -> int:
    args = parse_args(argv)
    category = args.category or _interactive_choose_category()
    if category not in ('insoles', 'feet'):
        raise SystemExit('无效类别，必须为 insoles 或 feet')

    root_dir = Path(args.root_dir)
    data_dir = root_dir / category
    output_dir = Path(args.output_dir) / category

    side_arg = (args.side or '').upper()
    sides_to_run = ['L', 'R'] if side_arg == 'LR' else [side_arg] if side_arg in ('L', 'R') else []
    if not sides_to_run:
        raise SystemExit('--side 必须是 L/R/LR')

    for s in sides_to_run:
        print(f"开始生成 {category}，侧别: {s}")
        if args.method == 'direct':
            pts = compute_average_direct_points(
                root_dir=data_dir,
                side=s,
                num_points=args.sampled_points,
                seed=args.seed,
                max_iter=args.kmeans_max_iter,
                batch_size=args.kmeans_batch_size,
                n_init=args.kmeans_n_init,
            )
            suffix = f"_{s}"
            template_path = output_dir / f'average_{category}_template{suffix}.npz'
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(template_path, points=pts.astype(np.float32))
            print(f"{category} {s} 侧平均模板已生成:")
            print(f"  模板点云: {template_path}")
        else:
            mesh, H_avg, bounds = compute_average_heightmap(
                root_dir=data_dir,
                grid_x=args.grid_x,
                grid_y=args.grid_y,
                min_points_per_cell=args.min_per_cell,
                side=s,
                min_coverage_count=args.min_coverage_count,
            )

            paths = save_outputs(
                mesh=mesh,
                H=H_avg,
                bounds=bounds,
                output_dir=output_dir,
                sampled_points=args.sampled_points,
                candidate_multiplier=args.candidate_multiplier,
                side=s,
                category_label=category[:-1] if category.endswith('s') else category,
            )

            print(f"{category} {s} 侧平均模板已生成:")
            print(f"  STL: {paths['stl']}")
            print(f"  模板点云: {paths['template']}")
            print(f"  高度图: {paths['heightmap']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


