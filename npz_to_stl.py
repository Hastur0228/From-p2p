import os
import sys
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

 


def load_points_from_npz(npz_path: Path) -> np.ndarray:
    """
    Load (N,3+) float array from .npz/.npy and return xyz as float64 (N,3).
    Accepts files with key 'points' or the first available array.
    """
    suffix = npz_path.suffix.lower()
    if suffix == '.npz':
        with np.load(str(npz_path)) as data:
            key = 'points' if 'points' in data.files else (data.files[0] if data.files else None)
            if key is None:
                raise ValueError(f"No arrays found in NPZ: {npz_path}")
            arr = data[key]
    elif suffix == '.npy':
        arr = np.load(str(npz_path))
    else:
        raise ValueError(f"Unsupported file type: {npz_path}")

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Invalid point array shape: {arr.shape} for {npz_path}")
    return arr[:, :3].astype(np.float64, copy=False)


 


def save_mesh_as_stl(mesh: o3d.geometry.TriangleMesh, out_path: Path, write_ascii: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=bool(write_ascii))
    if not ok:
        raise IOError(f"Failed to write STL: {out_path}")


 


def estimate_alpha_from_knn(point_cloud: o3d.geometry.PointCloud, k: int = 10, scale: float = 2.5) -> float:
    """
    Heuristic alpha estimator based on median 1-NN distance. Fallback to 3% of bbox diagonal.
    """
    k = max(2, int(k))
    scale = float(scale)
    kdt = o3d.geometry.KDTreeFlann(point_cloud)
    dists = []
    pts = point_cloud.points
    for i in range(len(pts)):
        _, _, d2 = kdt.search_knn_vector_3d(pts[i], k)
        if len(d2) >= 2:
            dists.append(float(np.sqrt(d2[1])))
    if len(dists) == 0:
        bbox = point_cloud.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        return max(1e-6, 0.03 * float(diag))
    median_nn = float(np.median(np.asarray(dists, dtype=np.float64)))
    return max(1e-6, scale * median_nn)


def reconstruct_alpha(points_xyz: np.ndarray, alpha: float | None, k: int, scale: float) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    if alpha is None or float(alpha) <= 0:
        alpha = estimate_alpha_from_knn(pcd, k=k, scale=scale)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, float(alpha))
    mesh.compute_vertex_normals()
    return mesh


 


def process_one(
    input_path: Path,
    output_path: Path,
    alpha: float | None,
    alpha_k: int,
    alpha_scale: float,
    smooth_iters: int,
    overwrite: bool,
    ascii_stl: bool,
) -> bool:
    try:
        if output_path.exists() and not overwrite:
            return True

        points = load_points_from_npz(input_path)
        if points.shape[0] < 4:
            raise ValueError("Too few points (<4) for surface reconstruction")

        mesh = reconstruct_alpha(points_xyz=points, alpha=alpha, k=alpha_k, scale=alpha_scale)
        # Optional single-stage Taubin smoothing
        if smooth_iters and int(smooth_iters) > 0:
            mesh_smooth = mesh.filter_smooth_taubin(number_of_iterations=int(smooth_iters))
            mesh_smooth.compute_vertex_normals()
            mesh = mesh_smooth
        if len(mesh.triangles) == 0:
            raise RuntimeError("Alpha shape produced empty mesh; try increasing --alpha or adjusting --alpha-scale")

        save_mesh_as_stl(mesh, output_path, write_ascii=ascii_stl)
        return True
    except Exception:
        return False


def find_inputs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fl = fn.lower()
            if fl.endswith('.npz') or fl.endswith('.npy'):
                files.append(Path(dirpath) / fn)
    files.sort()
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Convert .npz/.npy point clouds to STL using Alpha Shape.')
    parser.add_argument('--input', type=str, default=str(Path('output') / 'pointcloud'), help='Input file or directory (default: output/pointcloud)')
    parser.add_argument('--output-dir', type=str, default=str(Path('output') / 'stl'), help='Output directory for STL files')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for alpha shape (if omitted, auto-estimate)')
    parser.add_argument('--alpha-k', type=int, default=10, help='K for KNN in alpha auto-estimation (default: 10)')
    parser.add_argument('--alpha-scale', type=float, default=2.5, help='Scale multiplier on median NN distance (default: 2.5)')
    parser.add_argument('--ascii', action='store_true', help='Write ASCII STL (default False -> binary)')
    parser.add_argument('--smooth-iters', type=int, default=50, help='Taubin smoothing iterations (0 to disable)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing STL files')
    args = parser.parse_args(argv)
    input_root = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        return 1

    inputs = find_inputs(input_root)
    if not inputs:
        return 0

    ok_count = 0
    for i, src in enumerate(inputs, start=1):
        rel = src.name if input_root.is_file() else os.path.relpath(src, input_root)
        base, _ = os.path.splitext(rel)
        out_path = output_dir / (base + '.stl')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = process_one(
            input_path=src,
            output_path=out_path,
            alpha=args.alpha,
            alpha_k=args.alpha_k,
            alpha_scale=args.alpha_scale,
            overwrite=args.overwrite,
            smooth_iters=args.smooth_iters,
            ascii_stl=args.ascii,
        )
        if ok:
            ok_count += 1

    return 0 if ok_count == len(inputs) else 2


if __name__ == '__main__':
    sys.exit(main())


