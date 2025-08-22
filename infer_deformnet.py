import os
import sys
from pathlib import Path
import argparse
import logging

import numpy as np
import torch

# Project imports
from datasets.foot_insole_dataset import _load_npz_points
from utils.pc_normalize import normalize_pointcloud, denormalize_pointcloud
from Models.dgcnn_encoder import DGCNNEncoder
from Models.deformation_net import DeformationNet
from utils.logger import setup_logger


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	return tuple(int(p) for p in parts) if parts else (256, 256, 128)


def _load_template(path: Path) -> np.ndarray:
	with np.load(str(path)) as data:
		pts = data['points'] if 'points' in data.files else data[data.files[0]]
	pts = np.asarray(pts, dtype=np.float32)
	if pts.ndim != 2 or pts.shape[1] < 3:
		raise ValueError(f"模板点云形状无效: {path} -> {pts.shape}")
	return pts[:, :3].astype(np.float32)


def _maybe_subsample(arr: np.ndarray, num_points: int) -> np.ndarray:
	N = arr.shape[0]
	if N == num_points:
		return arr
	if N > num_points:
		idx = np.random.choice(N, num_points, replace=False)
		return arr[idx]
	# pad by repeating
	repeat = num_points // N + 1
	idx = np.arange(N).repeat(repeat)[: num_points]
	return arr[idx]


def _build_models_from_args(args_ns) -> tuple[torch.nn.Module, torch.nn.Module]:
	encoder = DGCNNEncoder(
		input_dims=(6 if args_ns.get('use_normals', False) else 3),
		k=int(args_ns.get('dgcnn_k', 20)),
		feat_dim=int(args_ns.get('dgcnn_feat_dim', 256)),
		dropout_p=float(args_ns.get('dgcnn_dropout', 0.3)),
		multi_scale_ks=tuple(int(k) for k in str(args_ns.get('dgcnn_multi_scale_ks', '10,20,30')).split(',') if str(k).strip()),
	)
	regressor = DeformationNet(
		global_feat_dim=int(args_ns.get('dgcnn_feat_dim', 256)),
		hidden_dims=_parse_hidden_dims(args_ns.get('hidden_dims', '256,256,128')),
		dropout_p=float(args_ns.get('mlp_dropout', 0.4)),
	)
	return encoder, regressor


def _select_checkpoint(side: str, ckpt_L: Path | None, ckpt_R: Path | None, ckpt_single: Path | None) -> Path:
	side_u = side.upper()
	if side_u == 'L' and ckpt_L and ckpt_L.exists():
		return ckpt_L
	if side_u == 'R' and ckpt_R and ckpt_R.exists():
		return ckpt_R
	if ckpt_single and ckpt_single.exists():
		return ckpt_single
	raise FileNotFoundError(f"未找到可用的权重文件用于侧别 {side}。请检查 --checkpoint-L/--checkpoint-R 或 --checkpoint")


def infer_one_file(file_path: Path,
					out_dir: Path,
					ckpt_L: Path | None,
					ckpt_R: Path | None,
					ckpt_single: Path | None,
					tpl_L: np.ndarray | None,
					tpl_R: np.ndarray | None,
					device: torch.device,
					logger: logging.Logger | None = None) -> None:
	name = file_path.name
	name_lower = name.lower()
	side = 'L' if '_l.' in name_lower else ('R' if '_r.' in name_lower else 'L')
	ckpt_path = _select_checkpoint(side, ckpt_L, ckpt_R, ckpt_single)
	if logger:
		logger.info(f"开始推理: {name} | 侧别: {side} | 使用权重: {ckpt_path}")
	ckpt = torch.load(str(ckpt_path), map_location='cpu')
	train_args = ckpt.get('args', {})
	if isinstance(train_args, dict):
		args_ns = train_args
	else:
		# Fallback in case older checkpoint stores Namespace-like object
		args_ns = dict(train_args)

	# Build model and load weights
	encoder, regressor = _build_models_from_args(args_ns)
	encoder.load_state_dict(ckpt['encoder'])
	regressor.load_state_dict(ckpt['regressor'])
	encoder.to(device).eval()
	regressor.to(device).eval()

	# Data pipeline consistent with training
	use_normals = bool(args_ns.get('use_normals', False))
	num_points = int(args_ns.get('num_points', 4096))
	norm_mode = str(args_ns.get('normalize', 'center')).lower()
	if logger:
		logger.info(f"配置: use_normals={use_normals} | num_points={num_points} | normalize={norm_mode}")

	foot = _load_npz_points(file_path, use_normals=use_normals)
	foot_norm, centroid, scale = normalize_pointcloud(foot, center_only=False, mode=norm_mode)
	tpl_np = None
	if side == 'L' and tpl_L is not None:
		tpl_np = tpl_L
	elif side == 'R' and tpl_R is not None:
		tpl_np = tpl_R
	# Fallback: use single template from L if provided, else from R
	if tpl_np is None:
		tpl_np = tpl_L if tpl_L is not None else tpl_R
	if tpl_np is None:
		raise RuntimeError('未提供模板点云，且默认模板不存在。请通过 --template/--template-L/--template-R 指定。')
	if logger:
		tpl_side = ('L' if tpl_np is tpl_L else ('R' if tpl_np is tpl_R else 'single'))
		logger.info(f"模板选择: {tpl_side} | 点数: foot={foot.shape[0]} tpl={tpl_np.shape[0]}")

	# Normalize template with same centroid/scale
	tpl_norm = tpl_np.copy()
	tpl_norm[:, :3] = (tpl_norm[:, :3] - centroid) / (scale + 1e-9)

	# Subsample to match training num_points
	before_foot, before_tpl = foot_norm.shape[0], tpl_norm.shape[0]
	foot_norm = _maybe_subsample(foot_norm, num_points)
	tpl_norm = _maybe_subsample(tpl_norm, num_points)
	if logger:
		logger.info(f"下采样: foot {before_foot}->{foot_norm.shape[0]} | tpl {before_tpl}->{tpl_norm.shape[0]}")

	# Torch tensors
	with torch.no_grad():
		foot_t = torch.from_numpy(foot_norm.astype(np.float32)).unsqueeze(0).to(device)  # (1,N,C)
		if foot_t.size(-1) == 3 or foot_t.size(-1) == 6:
			foot_in = foot_t.transpose(2, 1).contiguous()  # (1,C,N)
		else:
			raise ValueError(f"输入点云维度非法: {foot_t.shape}")
		tpl_t = torch.from_numpy(tpl_norm[:, :3].astype(np.float32)).unsqueeze(0).to(device)  # (1,N,3)

		global_feat = encoder(foot_in)  # (1,d)
		pred = regressor(tpl_t, global_feat)  # (1,N,3)
		pred_np = pred.squeeze(0).cpu().numpy().astype(np.float32)

	# Denormalize back to original coordinate system
	pred_denorm = denormalize_pointcloud(pred_np, centroid=centroid, scale=scale, center_only=False)

	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / name
	# Save as .npz with key 'points'
	try:
		np.savez_compressed(str(out_path), points=pred_denorm.astype(np.float32))
	except Exception:
		# Fallback to uncompressed to avoid issues on some platforms
		np.savez(str(out_path), points=pred_denorm.astype(np.float32))
	if logger:
		logger.info(f"已保存: {out_path}")


def main(argv=None):
	parser = argparse.ArgumentParser(description='DeformationNet 推理：读取 test/pointcloud 下的 .npz/.npy，输出到 output/pointcloud')
	parser.add_argument('--input-dir', type=str, default=str(Path('test') / 'pointcloud'))
	parser.add_argument('--output-dir', type=str, default=str(Path('output') / 'pointcloud'))
	parser.add_argument('--checkpoint', type=str, default=str(Path('checkpoints') / 'best.pth'), help='单模型权重（若未按左右分别训练）')
	parser.add_argument('--checkpoint-L', type=str, default=str(Path('checkpoints') / 'L' / 'best.pth'))
	parser.add_argument('--checkpoint-R', type=str, default=str(Path('checkpoints') / 'R' / 'best.pth'))
	parser.add_argument('--template-dir', type=str, default=str(Path('Templates') / 'insoles'), help='模板目录（优先从目录中自动匹配 L/R 模板）')
	parser.add_argument('--template', type=str, default=str(Path('Templates/insoles')/'average_insole_template.npz'))
	parser.add_argument('--template-L', type=str, default='')
	parser.add_argument('--template-R', type=str, default='')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	args = parser.parse_args(argv)

	# Logger
	log_dir = Path('Log') / 'infer'
	logger = setup_logger(log_dir, name='infer_deformnet')

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	ckpt_single = Path(args.checkpoint) if args.checkpoint else None
	ckpt_L = Path(args.checkpoint_L) if args.checkpoint_L else None
	ckpt_R = Path(args.checkpoint_R) if args.checkpoint_R else None
	logger.info(f"输入目录: {input_dir} | 输出目录: {output_dir}")
	logger.info(f"权重: single={ckpt_single} | L={ckpt_L} | R={ckpt_R}")

	if not input_dir.exists():
		raise FileNotFoundError(f"输入目录不存在: {input_dir}")

	# Resolve templates
	tpl_L = tpl_R = None
	template_dir = Path(args.template_dir) if getattr(args, 'template_dir', None) else None
	single_tpl_path = Path(args.template) if args.template else None
	# Prefer explicitly provided side-specific templates
	if args.template_L:
		path_L = Path(args.template_L)
		if path_L.exists():
			tpl_L = _load_template(path_L)
			logger.info(f"模板(L): {path_L}")
	if args.template_R:
		path_R = Path(args.template_R)
		if path_R.exists():
			tpl_R = _load_template(path_R)
			logger.info(f"模板(R): {path_R}")
	# Try to auto-detect from template directory if provided and side-specific not already set
	if template_dir and template_dir.exists() and (tpl_L is None or tpl_R is None):
		# Preferred filenames
		pref_names = [
			('average_insoles_template_L.npz', 'average_insoles_template_R.npz'),
			('average_insole_template_L.npz', 'average_insole_template_R.npz'),
		]
		for nameL, nameR in pref_names:
			candL = template_dir / nameL
			candR = template_dir / nameR
			if tpl_L is None and candL.exists():
				tpl_L = _load_template(candL)
				logger.info(f"模板(L)目录匹配: {candL}")
			if tpl_R is None and candR.exists():
				tpl_R = _load_template(candR)
				logger.info(f"模板(R)目录匹配: {candR}")
			if tpl_L is not None and tpl_R is not None:
				break
		# Fallback: any *_L.npz / *_R.npz pair
		if tpl_L is None:
			candsL = sorted([p for p in template_dir.glob('*_L.npz')])
			if candsL:
				tpl_L = _load_template(candsL[0])
				logger.info(f"模板(L)目录自动: {candsL[0]}")
		if tpl_R is None:
			candsR = sorted([p for p in template_dir.glob('*_R.npz')])
			if candsR:
				tpl_R = _load_template(candsR[0])
				logger.info(f"模板(R)目录自动: {candsR[0]}")
		# Fallback single template in directory
		if tpl_L is None or tpl_R is None:
			# Prefer known singles
			for single_name in ('average_insoles_template.npz', 'average_insole_template.npz'):
				sp = template_dir / single_name
				if sp.exists():
					tpl_single = _load_template(sp)
					if tpl_L is None:
						tpl_L = tpl_single
					if tpl_R is None:
						tpl_R = tpl_single
					logger.info(f"模板(单一)目录兜底: {sp}")
					break
			# Or any .npz
			if tpl_L is None or tpl_R is None:
				any_npz = sorted([p for p in template_dir.glob('*.npz')])
				if any_npz:
					tpl_single = _load_template(any_npz[0])
					if tpl_L is None:
						tpl_L = tpl_single
					if tpl_R is None:
						tpl_R = tpl_single
					logger.info(f"模板(单一)目录兜底: {any_npz[0]}")
	# Try to auto-detect side-specific files based on --template basename, regardless of single template presence
	if single_tpl_path:
		cand_L = single_tpl_path.with_name(single_tpl_path.stem + '_L' + single_tpl_path.suffix)
		cand_R = single_tpl_path.with_name(single_tpl_path.stem + '_R' + single_tpl_path.suffix)
		if tpl_L is None and cand_L.exists():
			tpl_L = _load_template(cand_L)
			logger.info(f"模板(L)自动匹配: {cand_L}")
		if tpl_R is None and cand_R.exists():
			tpl_R = _load_template(cand_R)
			logger.info(f"模板(R)自动匹配: {cand_R}")
	# Finally, if any side is still missing, fall back to the single template if it exists
	if single_tpl_path and single_tpl_path.exists():
		tpl_single = _load_template(single_tpl_path)
		if tpl_L is None:
			tpl_L = tpl_single
		if tpl_R is None:
			tpl_R = tpl_single
		logger.info(f"模板(单一)兜底: {single_tpl_path}")

	device = torch.device(args.device)
	logger.info(f"设备: {device}")

	files = [p for p in input_dir.iterdir() if p.suffix.lower() in ('.npz', '.npy')]
	files.sort()
	if len(files) == 0:
		logger.warning(f"未在 {input_dir} 找到 .npz/.npy 文件。")
		return 0

	for idx, fp in enumerate(files, start=1):
		try:
			logger.info(f"[{idx}/{len(files)}] {fp.name}")
			infer_one_file(
				file_path=fp,
				out_dir=output_dir,
				ckpt_L=ckpt_L if (ckpt_L and ckpt_L.exists()) else None,
				ckpt_R=ckpt_R if (ckpt_R and ckpt_R.exists()) else None,
				ckpt_single=ckpt_single if (ckpt_single and ckpt_single.exists()) else None,
				tpl_L=tpl_L,
				tpl_R=tpl_R,
				device=device,
				logger=logger,
			)
		except Exception:
			logger.exception(f"文件失败: {fp}")

	logger.info(f"推理完成，结果已保存到: {output_dir}")
	return 0


if __name__ == '__main__':
	sys.exit(main())


