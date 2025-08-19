import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import platform
import trimesh


def configure_per_file_logger(log_path: Path) -> logging.Logger:
	"""
	Create a per-file logger that writes to the given log_path.
	The file is opened in write mode to satisfy the overwrite requirement.
	"""
	logger = logging.getLogger(str(log_path))
	# Avoid duplicate handlers if called multiple times for the same file
	if logger.handlers:
		return logger
	logger.setLevel(logging.INFO)
	log_path.parent.mkdir(parents=True, exist_ok=True)
	file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
	file_handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger


def choose_output_format_interactive() -> str:
	"""
	Interactive CLI for choosing output format when not specified by flag.
	Returns 'npy' or 'npz'. Default is 'npz'.
	"""
	print("请选择输出格式 (Enter 选择默认):")
	print("  [1] npy (仅保存 points)")
	print("  [2] npz (保存 points/normals/centroid/scale/R/var_ratio) [默认]")
	choice = input("选择 1 或 2: ").strip()
	if choice == '1':
		return 'npy'
	return 'npz'


def choose_output_directory_interactive() -> Path:
	"""
	Interactive CLI for choosing output directory.
	"""
	print("请选择输出目录:")
	print("  [1] data/pointcloud (默认)")
	print("  [2] 自定义路径")
	choice = input("选择 1 或 2: ").strip()
	if choice == '2':
		custom_path = input("请输入自定义路径: ").strip()
		return Path(custom_path)
	return Path('data') / 'pointcloud'


def choose_save_parameters_interactive() -> tuple[bool, Path, str]:
	"""
	Interactive CLI for choosing save parameters.
	Returns (save_output, output_dir, format).
	"""
	print("是否保存处理结果?")
	print("  [1] 是")
	print("  [2] 否 (默认)")
	choice = input("选择 1 或 2: ").strip()
	
	if choice == '1':
		output_dir = choose_output_directory_interactive()
		output_format = choose_output_format_interactive()
		return True, output_dir, output_format
	else:
		return False, Path('data') / 'pointcloud', 'npz'  # 默认值，不会使用


def list_stl_files(input_dirs: List[Path]) -> List[Path]:
	"""Recursively list all STL files under the provided directories."""
	stl_files: List[Path] = []
	for d in input_dirs:
		if d.is_file() and d.suffix.lower() == '.stl':
			stl_files.append(d)
		elif d.is_dir():
			stl_files.extend([p for p in d.rglob('*.stl') if p.is_file()])
	return sorted(stl_files)


def farthest_point_sampling(points: np.ndarray, num_samples: int, logger: Optional[logging.Logger] = None) -> np.ndarray:
	"""
	Greedy Farthest Point Sampling (FPS) on a point set.

	Parameters
	----------
	points : (N, 3)
		Candidate point set.
	num_samples : int
		Number of points to sample.
	logger : Logger, optional
		Logger for progress.

	Returns
	-------
	indices : (num_samples,)
		Indices of selected points.
	"""
	N = points.shape[0]
	if num_samples >= N:
		if logger:
			logger.info(f"Requested samples {num_samples} >= candidates {N}; returning all indices.")
		return np.arange(N, dtype=np.int64)

	# Choose a random starting index to reduce bias
	start_index = int(np.random.randint(0, N))
	selected_indices = np.empty(num_samples, dtype=np.int64)
	selected_indices[0] = start_index

	# Maintain the distance to the sampled set for each point
	# Initialize with distances to the first selected point
	diff = points - points[start_index]
	min_dist_sq = np.einsum('ij,ij->i', diff, diff)

	for i in range(1, num_samples):
		# Pick the farthest point from current sampled set
		far_index = int(np.argmax(min_dist_sq))
		selected_indices[i] = far_index
		# Update min distances with the newly added point
		diff = points - points[far_index]
		dist_sq = np.einsum('ij,ij->i', diff, diff)
		min_dist_sq = np.minimum(min_dist_sq, dist_sq)
		if logger and (i % max(1, num_samples // 10) == 0):
			logger.info(f"FPS progress: {i}/{num_samples}")

	return selected_indices


def sample_points_on_mesh(
	mesh: trimesh.Trimesh,
	desired_points: int,
	candidate_multiplier: int,
	logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Sample approximately even points on the mesh surface by first drawing a larger
	set uniformly by area, then running FPS down to the target size.

	Returns points (P,3) and normals (P,3).
	"""
	# Ensure we have at least desired_points candidates
	num_candidates = max(desired_points * max(1, candidate_multiplier), desired_points * 2)
	if logger:
		logger.info(
			f"Surface sampling: candidates={num_candidates} (target={desired_points}, multiplier={candidate_multiplier})"
		)
	points_raw, face_ids = trimesh.sample.sample_surface(mesh, num_candidates)
	# Face normals assigned to sampled points
	normals_raw = mesh.face_normals[face_ids]
	# FPS down to desired_points
	if logger:
		logger.info("Running Farthest Point Sampling (FPS) to enforce even spacing")
	keep_idx = farthest_point_sampling(points_raw, desired_points, logger)
	return points_raw[keep_idx], normals_raw[keep_idx]


def filter_outliers_by_height(
	points: np.ndarray,
	low_percent: Optional[float],
	high_percent: Optional[float],
	logger: Optional[logging.Logger] = None,
) -> np.ndarray:
	"""
	Optionally filter points by z-height percentiles (e.g., remove overly low bottom or overly high spikes).
	Returns a boolean mask of kept points.
	"""
	if low_percent is None and high_percent is None:
		return np.ones(points.shape[0], dtype=bool)
	z = points[:, 2]
	keep = np.ones(points.shape[0], dtype=bool)
	if low_percent is not None:
		low_thr = np.percentile(z, low_percent)
		keep &= z >= low_thr
		if logger:
			logger.info(f"Height filter: removed below {low_percent:.2f}th percentile (z >= {low_thr:.6f})")
	if high_percent is not None:
		high_thr = np.percentile(z, high_percent)
		keep &= z <= high_thr
		if logger:
			logger.info(f"Height filter: removed above {high_percent:.2f}th percentile (z <= {high_thr:.6f})")
	return keep


def center_and_unit_normalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
	"""
	Center only (no scaling). Return (points_centered, centroid, scale=1.0).
	"""
	centroid = points.mean(axis=0, keepdims=True)
	points_centered = points - centroid
	return points_centered, centroid.reshape(-1), 1.0


def pca_align(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Align the point cloud to principal axes using PCA.
	Return (aligned_points, rotation_matrix R, explained_variance_ratio).
	"""
	# points are assumed centered (mean ~ 0)
	cov = np.cov(points.T)
	vals, vecs = np.linalg.eigh(cov)
	# Sort eigenvalues descending
	order = np.argsort(vals)[::-1]
	vals = vals[order]
	vecs = vecs[:, order]
	# Ensure right-handed coordinate system
	det_before = np.linalg.det(vecs)
	if det_before < 0:
		vecs[:, -1] *= -1.0
	R = vecs  # Columns are principal directions
	aligned = points @ R
	var_ratio = vals / (vals.sum() + 1e-12)
	return aligned, R, var_ratio


def _compute_region_widths_xy(points_aligned: np.ndarray, q_low: float = 0.10, q_high: float = 0.90) -> Tuple[float, float]:
	"""Compute Y-span at low/high X quantiles as a proxy for heel/toe widths."""
	x = points_aligned[:, 0]
	y = points_aligned[:, 1]
	if points_aligned.shape[0] == 0:
		return 0.0, 0.0
	low_thr = np.quantile(x, q_low)
	high_thr = np.quantile(x, q_high)
	y_low = y[x <= low_thr]
	y_high = y[x >= high_thr]
	width_low = float(y_low.max() - y_low.min()) if y_low.size > 0 else 0.0
	width_high = float(y_high.max() - y_high.min()) if y_high.size > 0 else 0.0
	return width_low, width_high


def enforce_foot_orientation(
	points_aligned: np.ndarray,
	normals_rotated: np.ndarray,
	logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Enforce consistent anatomical orientation after PCA alignment:
	- Plantar in XY plane (Z up): flip Z if mean normal Z is negative
	- Toes toward +X: compare Y-width at x-high vs x-low, flip X if needed
	- Preserve right-handedness by choosing Y flip accordingly

	Returns (points_oriented, normals_oriented, diag_signs)
	where diag_signs is a 3-vector [sx, sy, sz].
	"""
	# Decide Z flip based on average normal direction
	mean_nz = float(normals_rotated[:, 2].mean()) if normals_rotated.size else 0.0
	sz = 1.0 if mean_nz >= 0.0 else -1.0

	# Apply tentative Z flip to evaluate toe/heel widths on flattened sense of up
	points_tmp = points_aligned.copy()
	points_tmp[:, 2] *= sz

	# Toe vs heel heuristic using Y-span at extreme X
	width_low, width_high = _compute_region_widths_xy(points_tmp)
	# Toe region expected wider than heel; want toe at +X
	sx = 1.0 if width_high >= width_low else -1.0

	# Choose sy to preserve right-handedness (det > 0)
	# det of diag(sx, sy, sz) = sx*sy*sz, want +1 => sy = sign(+1 / (sx*sz))
	sy = 1.0 if (sx * sz) > 0 else -1.0

	if logger:
		logger.info(
			f"Orientation enforcement: mean_nz={mean_nz:.6f} -> sz={int(sz)}, "
			f"width_low={width_low:.6f}, width_high={width_high:.6f} -> sx={int(sx)}, sy={int(sy)}"
		)

	S = np.array([sx, sy, sz], dtype=np.float64)
	points_out = points_aligned * S
	normals_out = normals_rotated * S
	return points_out, normals_out, S


def ensure_enough_points(
	mesh: trimesh.Trimesh,
	desired_points: int,
	candidate_multiplier: int,
	low_percent: Optional[float],
	high_percent: Optional[float],
	logger: Optional[logging.Logger],
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Repeatedly sample until, after optional filtering, we have at least desired_points.
	Then FPS down to desired_points.
	"""
	accum_points: Optional[np.ndarray] = None
	accum_normals: Optional[np.ndarray] = None
	max_rounds = 5
	for round_idx in range(max_rounds):
		round_t0 = time.perf_counter()
		pts, nors = sample_points_on_mesh(mesh, desired_points, candidate_multiplier, logger)
		if accum_points is None:
			accum_points = pts
			accum_normals = nors
		else:
			accum_points = np.vstack([accum_points, pts])
			accum_normals = np.vstack([accum_normals, nors])
		mask = filter_outliers_by_height(accum_points, low_percent, high_percent, logger)
		kept = int(mask.sum())
		if logger:
			elapsed = (time.perf_counter() - round_t0) * 1000.0
			pct = 100.0 * kept / max(1, accum_points.shape[0])
			logger.info(
				f"Round {round_idx+1}: accumulated={accum_points.shape[0]}, kept={kept} ({pct:.2f}%), "
				f"time={elapsed:.1f} ms"
			)
		if kept >= desired_points:
			break
	
	if accum_points is None or accum_points.shape[0] == 0:
		raise RuntimeError("Sampling produced no points")
	
	mask = filter_outliers_by_height(accum_points, low_percent, high_percent, logger)
	points_kept = accum_points[mask]
	normals_kept = accum_normals[mask]
	if points_kept.shape[0] < desired_points:
		# Fallback: disable filtering if insufficient
		if logger:
			logger.warning(
				f"Insufficient points after filtering ({points_kept.shape[0]} < {desired_points}). "
				"Disabling filters as fallback."
			)
		points_kept = accum_points
		normals_kept = accum_normals
	
	# FPS to target size
	keep_idx = farthest_point_sampling(points_kept, desired_points, logger)
	return points_kept[keep_idx], normals_kept[keep_idx]


def process_single_stl(
	stl_path: Path,
	root_dir: Path,
	logs_root: Path,
	num_points: int,
	candidate_multiplier: int,
	z_low_percent: Optional[float],
	z_high_percent: Optional[float],
	seed: Optional[int],
	save_output: bool,
	output_base_dir: Path,
	output_format: Optional[str],
) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
	"""
	Process a single STL file end-to-end and write a log file summarizing all steps.
	"""
	# Derive relative path to keep folder structure in logs directory
	rel = stl_path.relative_to(root_dir)
	log_path = logs_root / rel.with_suffix('.log')
	logger = configure_per_file_logger(log_path)
	logger.info(f"Input STL: {stl_path}")
	logger.info(f"Output log: {log_path}")
	logger.info(
		f"Parameters: num_points={num_points}, candidate_multiplier={candidate_multiplier}, "
		f"z_low_percent={z_low_percent}, z_high_percent={z_high_percent}, seed={seed}"
	)
	# Environment & file metadata
	try:
		stat = stl_path.stat()
		logger.info(
			f"File: size_bytes={stat.st_size}, mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}"
		)
	except Exception:
		logger.info("File: size_bytes=?, mtime=? (stat failed)")
	logger.info(
		f"Env: python={sys.version.split()[0]}, numpy={np.__version__}, trimesh={trimesh.__version__}, "
		f"platform={platform.platform()}"
	)

	try:
		stage_t0 = time.perf_counter()
		mesh = trimesh.load(str(stl_path), force='mesh')
		if not isinstance(mesh, trimesh.Trimesh):
			raise ValueError("Loaded object is not a Trimesh mesh")
		logger.info(f"Mesh: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
		if not mesh.is_watertight:
			logger.warning("Mesh is not watertight; surface sampling may include boundary artifacts")
		# Mesh geometry stats
		try:
			area = float(mesh.area)
			logger.info(f"Mesh surface area: {area:.6f}")
		except Exception:
			logger.info("Mesh surface area: unavailable")
		try:
			vol = float(mesh.volume)
			logger.info(f"Mesh volume: {vol:.6f}")
		except Exception:
			logger.info("Mesh volume: unavailable (likely non-watertight)")
		try:
			bmin = mesh.bounds[0].tolist()
			bmax = mesh.bounds[1].tolist()
			logger.info(f"Mesh AABB min={bmin} max={bmax}")
		except Exception:
			logger.info("Mesh AABB: unavailable")
		try:
			boundary_edges = getattr(mesh, 'edges_boundary', None)
			num_boundary_edges = int(len(boundary_edges)) if boundary_edges is not None else 0
			logger.info(f"Boundary edges: {num_boundary_edges}")
		except Exception:
			logger.info("Boundary edges: unavailable")
		logger.info(f"Load mesh time: {(time.perf_counter() - stage_t0)*1000.0:.1f} ms")

		# 1) Sampling with FPS refinement
		stage_t0 = time.perf_counter()
		points_sampled, normals_sampled = ensure_enough_points(
			mesh=mesh,
			desired_points=num_points,
			candidate_multiplier=candidate_multiplier,
			low_percent=z_low_percent,
			high_percent=z_high_percent,
			logger=logger,
		)
		logger.info(f"Sampled points: {points_sampled.shape}, normals: {normals_sampled.shape}")
		logger.info(f"Sampling time: {(time.perf_counter() - stage_t0)*1000.0:.1f} ms")

		# 2) Centering only (no scaling)
		stage_t0 = time.perf_counter()
		points_normed, centroid, scale = center_and_unit_normalize(points_sampled)
		logger.info(
			f"Centering only: centroid={centroid.tolist()} | scale fixed to {scale:.6f} (no scaling)"
		)
		for axis, name in enumerate(['x','y','z']):
			coord = points_normed[:, axis]
			logger.info(
				f"Axis {name} stats (centered): min={float(coord.min()):.6f}, mean={float(coord.mean()):.6f}, "
				f"std={float(coord.std()):.6f}, max={float(coord.max()):.6f}"
			)
		logger.info(f"Centering time: {(time.perf_counter() - stage_t0)*1000.0:.1f} ms")

		# 3) PCA alignment
		stage_t0 = time.perf_counter()
		points_aligned, R, var_ratio = pca_align(points_normed)
		logger.info(
			"PCA alignment: explained variance ratio="
			+ ", ".join([f"{v:.4f}" for v in var_ratio.tolist()])
		)
		logger.info("Rotation matrix (columns are principal axes):\n" + np.array2string(R, precision=6))
		# Re-check right-handedness
		logger.info(f"Rotation matrix det={np.linalg.det(R):.6f}")

		# 3.5) Enforce consistent anatomical orientation
		normals_rotated = normals_sampled @ R
		points_oriented, normals_oriented, diag_signs = enforce_foot_orientation(points_aligned, normals_rotated, logger)
		logger.info(f"Orientation signs diag(S)={diag_signs.tolist()}")
		logger.info(f"PCA time: {(time.perf_counter() - stage_t0)*1000.0:.1f} ms")

		# 4) Ensure exact point count via FPS (final uniformity)
		stage_t0 = time.perf_counter()
		final_idx = farthest_point_sampling(points_oriented, num_points, logger)
		points_final = points_oriented[final_idx]
		normals_final = normals_oriented[final_idx]
		logger.info(f"Final points: {points_final.shape}, normals: {normals_final.shape}")
		# Final cloud stats
		bbox_min = points_final.min(axis=0)
		bbox_max = points_final.max(axis=0)
		center_dist = float(np.linalg.norm(points_final.mean(axis=0)))
		radii_final = np.linalg.norm(points_final, axis=1)
		p5, p50, p95 = np.percentile(radii_final, [5, 50, 95]).tolist()
		logger.info(f"Aligned normalized AABB min={bbox_min.tolist()} max={bbox_max.tolist()}")
		logger.info(
			f"Final radii percentiles: p5={p5:.6f}, p50={p50:.6f}, p95={p95:.6f}; centroid_norm={center_dist:.6f}"
		)
		for axis, name in enumerate(['x','y','z']):
			coord = points_final[:, axis]
			logger.info(
				f"Axis {name} stats (final): min={float(coord.min()):.6f}, mean={float(coord.mean()):.6f}, "
				f"std={float(coord.std()):.6f}, max={float(coord.max()):.6f}"
			)
		# Normals statistics
		norm_lengths = np.linalg.norm(normals_final, axis=1)
		logger.info(
			f"Normal lengths: min={float(norm_lengths.min()):.6f}, "
			f"mean={float(norm_lengths.mean()):.6f}, max={float(norm_lengths.max()):.6f}"
		)
		logger.info(f"Final FPS time: {(time.perf_counter() - stage_t0)*1000.0:.1f} ms")
		# Save outputs if requested
		if save_output:
			# Determine output directory based on source type
			rel_to_root = stl_path.relative_to(root_dir)
			if 'feet' in str(rel_to_root):
				output_dir = output_base_dir / 'feet'
			elif 'insoles' in str(rel_to_root):
				output_dir = output_base_dir / 'insoles'
			else:
				# Fallback: use original structure
				output_dir = output_base_dir / rel_to_root.parent
			out_base = (output_dir / rel_to_root.name).with_suffix('')
			out_base.parent.mkdir(parents=True, exist_ok=True)
			fmt = output_format
			if fmt is None or fmt == 'auto':
				fmt = choose_output_format_interactive()
			if fmt == 'npy':
				# points only
				out_path = out_base.with_suffix('.npy')
				np.save(out_path, points_final)
				logger.info(f"Saved points (npy): {out_path} | shape={points_final.shape}")
			elif fmt == 'npz':
				out_path = out_base.with_suffix('.npz')
				np.savez_compressed(
					out_path,
					points=points_final,
					normals=normals_final,
					centroid=centroid,
					scale=np.array([scale], dtype=np.float32),
					R=R,
					var_ratio=var_ratio,
					params=np.array(
						[
							('num_points', num_points),
							('candidate_multiplier', candidate_multiplier),
							('z_low_percent', z_low_percent if z_low_percent is not None else 'None'),
							('z_high_percent', z_high_percent if z_high_percent is not None else 'None'),
							('seed', seed if seed is not None else 'None'),
						],
						dtype=object,
					),
				)
				logger.info(
					f"Saved bundle (npz): {out_path} | points={points_final.shape} normals={normals_final.shape}"
				)
			else:
				logger.warning(f"Unknown output format '{fmt}', skip saving.")

		logger.info("Completed successfully.")

		# Return category and final arrays for optional global aggregation
		category = 'insoles' if 'insoles' in str(rel) else ('feet' if 'feet' in str(rel) else 'other')
		return category, points_final, normals_final

	except Exception as e:
		logger.exception(f"Failed to process {stl_path}: {e}")
		return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Convert STL meshes to preprocessed point clouds (sampling + normals + "
			"centering/normalization + PCA alignment + uniform count) and write per-file logs."
		)
	)
	parser.add_argument(
		"--input-dirs",
		nargs='+',
		default=[
			str(Path('data') / 'raw' / 'feet'),
			str(Path('data') / 'raw' / 'insoles'),
		],
		help="One or more directories (or STL files) to process recursively.",
	)
	parser.add_argument(
		"--root-dir",
		default=str(Path('data') / 'raw'),
		help="Root directory used to mirror relative paths inside the logs directory.",
	)
	parser.add_argument(
		"--logs-dir",
		default=str(Path('Log') / 'preprocess'),
		help="Directory to store per-file logs (relative structure preserved under this root).",
	)
	parser.add_argument(
		"--num-points",
		type=int,
		default=4096,
		help="Target number of points per point cloud (e.g., 2048~4096).",
	)
	parser.add_argument(
		"--candidate-multiplier",
		type=int,
		default=4,
		help=(
			"Multiplier of target points for initial surface sampling before FPS. "
			"Higher values improve evenness at a cost of speed/memory."
		),
	)
	parser.add_argument(
		"--z-low-percent",
		type=float,
		default=None,
		help="Optional: remove points below this z-height percentile (e.g., 0.5 for bottom outliers).",
	)
	parser.add_argument(
		"--z-high-percent",
		type=float,
		default=None,
		help="Optional: remove points above this z-height percentile (e.g., 99.5 for tall spikes).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional: random seed for reproducible sampling/FPS start point.",
	)
	parser.add_argument(
		"--save-output",
		action='store_true',
		help="When set, save processed point clouds to the output directory (interactive if not specified).",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Base directory to write npy/npz outputs (interactive if not specified).",
	)
	parser.add_argument(
		"--format",
		choices=['npy', 'npz', 'auto'],
		default='auto',
		help="Output format: npy (points only), npz (bundle), or auto (interactive choice).",
	)
	parser.add_argument(
		"--generate-global-insole-template",
		action='store_true',
		help=(
			"Aggregate all processed insoles to build one global flat template (z=0; normals set to +Z). "
			"Template is sampled to --num-points and saved under <output>/templates/insole_template.(npy|npz)."
		),
	)
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	input_dirs = [Path(p).resolve() for p in args.input_dirs]
	root_dir = Path(args.root_dir).resolve()
	logs_root = Path(args.logs_dir).resolve()
	logs_root.mkdir(parents=True, exist_ok=True)
	
	# Interactive save parameter selection if not fully specified
	save_output = args.save_output
	output_base_dir = args.output_dir
	output_format = args.format
	
	if not args.save_output or args.output_dir is None or args.format == 'auto':
		print("\n=== 交互式参数选择 ===")
		save_output, output_base_dir, output_format = choose_save_parameters_interactive()
		print("=== 参数选择完成 ===\n")
	
	output_base_dir = Path(output_base_dir).resolve()
	if save_output:
		output_base_dir.mkdir(parents=True, exist_ok=True)

	# Set seed if provided
	if args.seed is not None:
		try:
			np.random.seed(args.seed)
		except Exception:
			pass

	# Discover files
	stl_files = list_stl_files(input_dirs)
	if not stl_files:
		print("No STL files found in input directories.")
		return 1

	print(f"Found {len(stl_files)} STL files. Writing logs to: {logs_root}")
	# Accumulators for global insole template
	insole_points_accum: list[np.ndarray] = []
	insole_normals_accum: list[np.ndarray] = []

	for idx, stl in enumerate(stl_files, start=1):
		print(f"[{idx}/{len(stl_files)}] Processing: {stl}")
		result = process_single_stl(
			stl_path=stl,
			root_dir=root_dir,
			logs_root=logs_root,
			num_points=args.num_points,
			candidate_multiplier=args.candidate_multiplier,
			z_low_percent=args.z_low_percent,
			z_high_percent=args.z_high_percent,
			seed=args.seed,
			save_output=save_output,
			output_base_dir=output_base_dir,
			output_format=output_format,
		)
		if result is not None and args.generate_global_insole_template:
			category, pts, nrm = result
			if category == 'insoles':
				insole_points_accum.append(pts)
				insole_normals_accum.append(nrm)

	# After all files, build a single global insole template if requested
	if args.generate_global_insole_template and insole_points_accum:
		print("Building global insole template from all processed insoles...")
		all_pts = np.vstack(insole_points_accum)
		all_nrm = np.vstack(insole_normals_accum)
		# Flatten z to 0 and normals to +Z
		all_pts[:, 2] = 0.0
		all_nrm[:, :] = 0.0
		all_nrm[:, 2] = 1.0
		# FPS down to target count
		keep_idx = farthest_point_sampling(all_pts, args.num_points)
		template_pts = all_pts[keep_idx]
		template_nrm = all_nrm[keep_idx]
		# Save under templates
		templates_dir = output_base_dir / 'templates'
		templates_dir.mkdir(parents=True, exist_ok=True)
		if output_format == 'npy':
			out_path = templates_dir / 'insole_template.npy'
			np.save(out_path, template_pts)
			print(f"Saved global insole template (npy): {out_path} | shape={template_pts.shape}")
		else:
			# use npz to preserve normals
			out_path = templates_dir / 'insole_template.npz'
			np.savez_compressed(out_path, points=template_pts, normals=template_nrm)
			print(f"Saved global insole template (npz): {out_path} | points={template_pts.shape} normals={template_nrm.shape}")

	print("Done.")
	return 0


if __name__ == "__main__":
	sys.exit(main())


