import os
import sys
from pathlib import Path
import argparse
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False

try:
	from tqdm import tqdm
	_HAS_TQDM = True
except Exception:
	_HAS_TQDM = False

# 项目内模块导入
from datasets.foot_insole_dataset import FootInsoleDataset
from Models.dgcnn_encoder import DGCNNEncoder
from Models.deformation_net import DeformationNet
from losses.chamfer import chamfer_distance, local_chamfer_distance
from losses.emd import emd_loss
from utils.logger import setup_logger


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _format_bytes(num_bytes: int) -> str:
	units = ['B', 'KB', 'MB', 'GB', 'TB']
	size = float(num_bytes)
	u = 0
	while size >= 1024.0 and u < len(units) - 1:
		size /= 1024.0
		u += 1
	return f"{size:.2f}{units[u]}"


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description='训练 Deformation Network（足模 -> 鞋垫）')
	parser.add_argument('--data-root', type=str, default=str(Path('data') / 'pointcloud'), help='数据根目录')
	parser.add_argument('--template', type=str, default=str(Path('Templates') / 'average_insole_template.npz'), help='模板鞋垫点云路径(.npz)。当 --side=LR 时，若未显式提供左右模板，将尝试自动使用 *_L.npz 与 *_R.npz')
	parser.add_argument('--template-L', type=str, default=None, help='左脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--template-R', type=str, default=None, help='右脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--no-interactive', action='store_true', help='关闭交互式确认，直接按参数启动训练')
	parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR', help='训练侧别：L 或 R；LR 表示顺序训练左右脚（先 L 后 R）')
	# 数据集参数
	parser.add_argument('--num-points', type=int, default=4096, help='每个样本点数（会对输入与标签点云做采样/重复以匹配）')
	parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例（0~1）')
	parser.add_argument('--no-shuffle', action='store_true', help='禁用点顺序随机打乱')
	parser.add_argument('--use-normals', action='store_true', help='足模输入是否包含法向(6维)')
	parser.add_argument('--normalize', type=str, choices=['sphere', 'cube', 'center'], default='center', help='点云标准化方式')
	# 模型参数
	parser.add_argument('--dgcnn-k', type=int, default=20, help='DGCNN 邻域点数 k')
	parser.add_argument('--dgcnn-feat-dim', type=int, default=256, help='DGCNN 输出全局特征维度')
	parser.add_argument('--dgcnn-dropout', type=float, default=0.3, help='DGCNN 内部 Dropout 概率')
	parser.add_argument('--dgcnn-multi-scale-ks', type=str, default='10,20,30', help='多尺度 EdgeConv 的 k 列表，如: 10,20,30；留空禁用')
	parser.add_argument('--hidden-dims', type=str, default='256,256,128', help='回归器隐藏层维度（逗号分隔）')
	parser.add_argument('--mlp-dropout', type=float, default=0.4, help='回归器 MLP 的 Dropout 概率')
	# 优化器与调度
	parser.add_argument('--batch-size', type=int, default=12)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-3, help='优化器权重衰减')
	parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau', 'none'], default='plateau')
	parser.add_argument('--step-size', type=int, default=40, help='StepLR 的步长')
	parser.add_argument('--gamma', type=float, default=0.5, help='StepLR 衰减率')
	# warmup 与 Plateau 调度
	parser.add_argument('--warmup-epochs', type=int, default=5, help='学习率 warmup 轮数（0 关闭）')
	parser.add_argument('--warmup-start-factor', type=float, default=0.1, help='warmup 起始因子，相对 base lr')
	parser.add_argument('--plateau-patience', type=int, default=5, help='ReduceLROnPlateau 的耐心轮数')
	parser.add_argument('--plateau-factor', type=float, default=0.5, help='ReduceLROnPlateau 的衰减因子')
	parser.add_argument('--plateau-min-lr', type=float, default=1e-6, help='ReduceLROnPlateau 的最小学习率')
	# 提前停止
	parser.add_argument('--early-stopping', action='store_true', help='启用提前停止机制')
	parser.add_argument('--patience', type=int, default=20, help='提前停止耐心值')
	parser.add_argument('--min-delta', type=float, default=1e-6, help='提前停止最小改善阈值')
	# 损失/评估参数
	parser.add_argument('--cd-chunk', type=int, default=1024, help='Chamfer 距离计算分块大小')
	parser.add_argument('--cd-normalize-loss', action='store_true', help='对 Chamfer 距离按几何尺度做归一化（保持真实尺寸，仅缩放损失数值）')
	parser.add_argument('--global-cd-weight', type=float, default=0.5, help='全局 Chamfer Distance 的损失权重 (α)')
	parser.add_argument('--local-cd-weight', type=float, default=0.1, help='局部 Chamfer Distance 的损失权重 (0 关闭)')
	parser.add_argument('--local-cd-patches', type=int, default=64, help='每样本局部 patch 数量')
	parser.add_argument('--local-cd-radius', type=float, default=0.2, help='局部邻域半径（与坐标同单位）')
	parser.add_argument('--emd-weight', type=float, default=0.5, help='EMD（Sinkhorn 近似）的损失权重 (β)')
	parser.add_argument('--emd-eps', type=float, default=0.02, help='EMD 的 Sinkhorn 熵正则强度 epsilon')
	parser.add_argument('--emd-iters', type=int, default=50, help='EMD 的 Sinkhorn 迭代次数')
	parser.add_argument('--emd-subset', type=int, default=1024, help='EMD 的子采样点数（降低显存/算力消耗）')
	parser.add_argument('--offset-l2-weight', type=float, default=0.0, help='位移向量 L2 正则项权重')
	parser.add_argument('--clip-grad-norm', type=float, default=1.0, help='梯度裁剪的 max_norm（0 或负数关闭）')
	# 其它
	parser.add_argument('--save-dir', type=str, default='checkpoints')
	parser.add_argument('--log-dir', type=str, default=str(Path('Log') / 'train'))
	parser.add_argument('--num-workers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--seed', type=int, default=42)
	return parser.parse_args(argv)


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	try:
		dims = tuple(int(p) for p in parts)
	except Exception as e:
		raise ValueError(f"hidden-dims 解析失败: {s}") from e
	if len(dims) == 0:
		raise ValueError("hidden-dims 不能为空")
	return dims


def _parse_ms_ks(s: str | None) -> tuple[int, ...]:
	if not s:
		return tuple()
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	ks: list[int] = []
	for p in parts:
		try:
			v = int(p)
			if v > 0:
				ks.append(v)
		except Exception:
			continue
	return tuple(sorted(set(ks)))


def _model_num_params(model: torch.nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def make_dataloaders(args, side: str | None = None, template_path: str | None = None):
	side_norm = None if (side is None or str(side).upper() == 'LR') else str(side).upper()
	train_set = FootInsoleDataset(
		data_root=args.data_root,
		split='train',
		use_normals=args.use_normals,
		template_path=template_path or args.template,
		num_points=args.num_points,
		val_ratio=args.val_ratio,
		random_shuffle_points=(not args.no_shuffle),
		side=side_norm,
	)
	train_set.normalize_mode = args.normalize
	if getattr(args, 'aug_enable', False):
		train_set.augment_enable = True
		train_set.augment_multiplier = max(1, int(getattr(args, 'aug_multiplier', 8)))
		def _parse_range_pair(s: str, typ=float):
			parts = [p.strip() for p in str(s).split(',') if p.strip()]
			if len(parts) == 1:
				v = typ(parts[0])
				return (v, v)
			if len(parts) >= 2:
				return (typ(parts[0]), typ(parts[1]))
			return (0.0, 0.0)
		train_set.aug_jitter_sigma_range = _parse_range_pair(getattr(args, 'aug_jitter_sigma', '0.002,0.01'), float)
		train_set.aug_dropout_patches_range = _parse_range_pair(getattr(args, 'aug_dropout_patches', '0,3'), int)
		train_set.aug_dropout_radius_range = _parse_range_pair(getattr(args, 'aug_dropout_radius', '0.05,0.15'), float)
		train_set.aug_normal_shift_range = _parse_range_pair(getattr(args, 'aug_normal_shift', '0.0,0.02'), float)
		train_set.aug_resample_mode = str(getattr(args, 'aug_resample', 'poisson')).lower()
		train_set.aug_uniform_keep_range = _parse_range_pair(getattr(args, 'aug_uniform_keep', '0.6,1.0'), float)
		train_set.aug_poisson_voxel_range = _parse_range_pair(getattr(args, 'aug_poisson_voxel', '0.01,0.04'), float)

	val_set = FootInsoleDataset(
		data_root=args.data_root,
		split='val',
		use_normals=args.use_normals,
		template_path=template_path or args.template,
		num_points=args.num_points,
		val_ratio=args.val_ratio,
		random_shuffle_points=False,
		side=side_norm,
	)
	val_set.normalize_mode = args.normalize
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
	val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
	return train_loader, val_loader


def build_model(args):
	input_dims = 6 if args.use_normals else 3
	ms_ks = _parse_ms_ks(getattr(args, 'dgcnn_multi_scale_ks', '10,20,30'))
	encoder = DGCNNEncoder(
		input_dims=input_dims,
		k=args.dgcnn_k,
		feat_dim=args.dgcnn_feat_dim,
		dropout_p=getattr(args, 'dgcnn_dropout', 0.1),
		multi_scale_ks=ms_ks,
	)
	regressor = DeformationNet(global_feat_dim=args.dgcnn_feat_dim, hidden_dims=_parse_hidden_dims(args.hidden_dims), dropout_p=getattr(args, 'mlp_dropout', 0.2))
	return encoder, regressor


def evaluate(encoder, regressor, loader, device, cd_chunk: int, local_cd_cfg: dict[str, float | int] | None = None, cd_normalize: bool = False, emd_cfg: dict | None = None, weights: dict | None = None):
	encoder.eval()
	regressor.eval()
	total_loss = 0.0
	total_cnt = 0
	with torch.no_grad():
		for batch in loader:
			foot = batch['foot'].to(device)
			template = batch['template'].to(device)
			target = batch['insole'].to(device)

			foot_in = foot.transpose(2, 1).contiguous()
			global_feat = encoder(foot_in)
			pred = regressor(template, global_feat)
			cd = chamfer_distance(pred, target, reduction='mean', chunk=cd_chunk, normalize_loss=cd_normalize)
			loss = cd * (weights.get('cd', 1.0) if weights else 1.0)
			if local_cd_cfg and local_cd_cfg.get('weight', 0.0) > 0:
				lcd = local_chamfer_distance(
					pred, target,
					num_patches=int(local_cd_cfg.get('patches', 64)),
					radius=float(local_cd_cfg.get('radius', 0.2)),
					reduction='mean',
					chunk=cd_chunk,
					normalize_loss=cd_normalize,
				)
				loss = loss + float(local_cd_cfg['weight']) * lcd
			if emd_cfg and (emd_cfg.get('weight', 0.0) > 0):
				e = emd_loss(
					pred, target,
					epsilon=float(emd_cfg.get('eps', 0.02)),
					num_iters=int(emd_cfg.get('iters', 50)),
					subset_size=int(emd_cfg.get('subset', 1024)),
					reduction='mean',
					normalize_loss=cd_normalize,
				)
				loss = loss + float(emd_cfg['weight']) * e
			bs = foot.size(0)
			total_loss += loss.item() * bs
			total_cnt += bs
	avg_loss = total_loss / max(1, total_cnt)
	return avg_loss


def _resolve_templates_for_side(args):
	side = (args.side or 'LR').upper()
	if side != 'LR':
		return side, args.template
	tpl_L = args.template_L
	tpl_R = args.template_R
	if (tpl_L is None) or (tpl_R is None):
		base = Path(args.template)
		if base.suffix.lower() == '.npz':
			stem = base.stem
			parent = base.parent
			tpl_L = tpl_L or str(parent / f"{stem}_L.npz")
			tpl_R = tpl_R or str(parent / f"{stem}_R.npz")
	return side, (tpl_L, tpl_R)


def _interactive_edit(args):
	def _prompt_typed(prompt: str, caster, default_value):
		while True:
			raw = input(f"{prompt} [默认: {default_value}]: ").strip()
			if raw == '':
				return default_value
			try:
				return caster(raw)
			except Exception:
				print("输入无效，请重试。")

	def _cast_sched(s: str) -> str:
		s = s.strip().lower()
		if s not in ('cosine', 'step', 'plateau', 'none'):
			raise ValueError('非法调度器')
		return s

	def _cast_norm(s: str) -> str:
		s = s.strip().lower()
		if s not in ('center', 'sphere', 'cube'):
			raise ValueError('非法标准化方式')
		return s

	print("=== 交互式参数设置 ===")
	while True:
		print("\n可修改的参数列表：")
		param_list = [
			("数据根目录", args.data_root, str),
			("模板路径", args.template, str),
			("设备", args.device, str),
			("批大小", args.batch_size, int),
			("轮数", args.epochs, int),
			("学习率", args.lr, float),
			("权重衰减", args.weight_decay, float),
			("调度器", args.scheduler, _cast_sched),
			("使用法向", args.use_normals, lambda x: x.lower() in ('y','yes','true','1')), 
			("每样本点数", args.num_points, int),
			("验证集比例", args.val_ratio, float),
			("训练时打乱点顺序(否=固定)", not args.no_shuffle, lambda x: x.lower() in ('y','yes','true','1')),
			("标准化方式", args.normalize, _cast_norm),
			("DGCNN 邻域 k", args.dgcnn_k, int),
			("DGCNN 特征维度", args.dgcnn_feat_dim, int),
			("DGCNN Dropout 概率", args.dgcnn_dropout, float),
			("多尺度 k 列表(逗号分隔)", args.dgcnn_multi_scale_ks, str),
			("回归器隐藏层(逗号分隔)", args.hidden_dims, str),
			("MLP Dropout", args.mlp_dropout, float),
			("CD 分块大小", args.cd_chunk, int),
			("CD 归一化(按几何尺度)", args.cd_normalize_loss, lambda x: x.lower() in ('y','yes','true','1')),
			("全局CD 权重α", args.global_cd_weight, float),
			("局部CD 权重", args.local_cd_weight, float),
			("局部CD patch 数", args.local_cd_patches, int),
			("局部CD 半径", args.local_cd_radius, float),
			("EMD 权重β", args.emd_weight, float),
			("EMD eps", args.emd_eps, float),
			("EMD iters", args.emd_iters, int),
			("EMD subset", args.emd_subset, int),
			("位移L2 权重", args.offset_l2_weight, float),
			("梯度裁剪 max_norm", args.clip_grad_norm, float),
			("早停(开启y/关闭n)", args.early_stopping, lambda x: x.lower() in ('y','yes','true','1')),
			("早停耐心", args.patience, int),
			("早停最小改善", args.min_delta, float),
			("日志目录", args.log_dir, str),
			("模型保存目录", args.save_dir, str),
		]
		for i, (label, current, _) in enumerate(param_list, 1):
			print(f"{i:2d}. {label}: 当前={current}")
		choice = input("\n请输入参数序号 (0完成): ").strip()
		if choice == '0':
			break
		try:
			idx = int(choice) - 1
			if idx < 0 or idx >= len(param_list):
				print("无效序号。")
				continue
			label, current, caster = param_list[idx]
			new_value = _prompt_typed(f"请输入新的 {label}", caster, current)
			# 赋值
			if label == "数据根目录": args.data_root = new_value
			elif label == "模板路径": args.template = new_value
			elif label == "设备": args.device = new_value
			elif label == "批大小": args.batch_size = new_value
			elif label == "轮数": args.epochs = new_value
			elif label == "学习率": args.lr = new_value
			elif label == "权重衰减": args.weight_decay = new_value
			elif label == "调度器": args.scheduler = new_value
			elif label == "使用法向": args.use_normals = bool(new_value)
			elif label == "每样本点数": args.num_points = new_value
			elif label == "验证集比例": args.val_ratio = new_value
			elif label == "训练时打乱点顺序(否=固定)": args.no_shuffle = (not bool(new_value))
			elif label == "标准化方式": args.normalize = new_value
			elif label == "DGCNN 邻域 k": args.dgcnn_k = new_value
			elif label == "DGCNN 特征维度": args.dgcnn_feat_dim = new_value
			elif label == "DGCNN Dropout 概率": args.dgcnn_dropout = new_value
			elif label == "多尺度 k 列表(逗号分隔)": args.dgcnn_multi_scale_ks = new_value
			elif label == "回归器隐藏层(逗号分隔)": args.hidden_dims = new_value
			elif label == "MLP Dropout": args.mlp_dropout = new_value
			elif label == "CD 分块大小": args.cd_chunk = new_value
			elif label == "CD 归一化(按几何尺度)": args.cd_normalize_loss = bool(new_value)
			elif label == "全局CD 权重α": args.global_cd_weight = new_value
			elif label == "局部CD 权重": args.local_cd_weight = new_value
			elif label == "局部CD patch 数": args.local_cd_patches = new_value
			elif label == "局部CD 半径": args.local_cd_radius = new_value
			elif label == "EMD 权重β": args.emd_weight = new_value
			elif label == "EMD eps": args.emd_eps = new_value
			elif label == "EMD iters": args.emd_iters = new_value
			elif label == "EMD subset": args.emd_subset = new_value
			elif label == "位移L2 权重": args.offset_l2_weight = new_value
			elif label == "梯度裁剪 max_norm": args.clip_grad_norm = new_value
			elif label == "早停(开启y/关闭n)": args.early_stopping = bool(new_value)
			elif label == "早停耐心": args.patience = new_value
			elif label == "早停最小改善": args.min_delta = new_value
			elif label == "日志目录": args.log_dir = new_value
			elif label == "模型保存目录": args.save_dir = new_value
			print(f"已修改 {label}: {new_value}")
		except Exception:
			print("请输入有效的数字。")


def train():
	args = parse_args()
	if not args.no_interactive:
		_interactive_edit(args)

	set_seed(args.seed)
	device = torch.device(args.device)
	logger = setup_logger(Path(args.log_dir), name='train_deformnet')

	# 详细环境信息
	logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | cudnn: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
	if torch.cuda.is_available():
		props = torch.cuda.get_device_properties(0)
		logger.info(f"CUDA Device: {props.name} | SMs: {props.multi_processor_count} | Mem: {_format_bytes(props.total_memory)}")
	logger.info(f"使用设备: {device}")

	side, tpl = _resolve_templates_for_side(args)
	sides_to_run = ['L', 'R'] if side == 'LR' else [side]

	for side_run in sides_to_run:
		if isinstance(tpl, tuple):
			tpl_L, tpl_R = tpl
			template_path = tpl_L if side_run == 'L' else tpl_R
		else:
			template_path = args.template

		sub_save = Path(args.save_dir) / side_run
		sub_log = Path(args.log_dir) / side_run
		sub_save.mkdir(parents=True, exist_ok=True)
		sub_log.mkdir(parents=True, exist_ok=True)

		# Data
		train_loader, val_loader = make_dataloaders(args, side=side_run, template_path=template_path)
		logger.info(f"数据集: 训练批次={len(train_loader)} | 验证批次={len(val_loader)} | 每批={args.batch_size}")

		# Model
		encoder, regressor = build_model(args)
		encoder.to(device)
		regressor.to(device)
		logger.info(f"DGCNN: k={args.dgcnn_k}, feat_dim={args.dgcnn_feat_dim}, dropout={args.dgcnn_dropout}, ms_ks={args.dgcnn_multi_scale_ks}")
		logger.info(f"Regressor: hidden={_parse_hidden_dims(args.hidden_dims)}, mlp_dropout={args.mlp_dropout}")
		logger.info(f"参数量: encoder={_model_num_params(encoder):,} | regressor={_model_num_params(regressor):,}")

		# Optim
		params = list(encoder.parameters()) + list(regressor.parameters())
		optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
		if args.scheduler == 'cosine':
			scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
		elif args.scheduler == 'step':
			scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
		elif args.scheduler == 'plateau':
			scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.plateau_min_lr)
		else:
			scheduler = None

		logger.info(f"损失: α(CD)={args.global_cd_weight}, β(EMD)={args.emd_weight}, localCD={args.local_cd_weight}, offsetL2={args.offset_l2_weight} | CD归一化={args.cd_normalize_loss}")
		logger.info(f"EMD: eps={args.emd_eps}, iters={args.emd_iters}, subset={args.emd_subset} | 局部CD: patches={args.local_cd_patches}, radius={args.local_cd_radius}")
		logger.info(f"梯度裁剪: max_norm={args.clip_grad_norm}")

		best_val = float('inf')
		patience_counter = 0
		best_val_epoch = 0
		hist_train_loss: list[float] = []
		hist_val_loss: list[float] = []

		for epoch in range(1, args.epochs + 1):
			start_t = time.time()
			encoder.train()
			regressor.train()
			# LR warmup
			if args.warmup_epochs and epoch <= args.warmup_epochs:
				warmup_progress = 0.0 if args.warmup_epochs <= 1 else float(epoch - 1) / float(args.warmup_epochs - 1)
				factor = args.warmup_start_factor + (1.0 - args.warmup_start_factor) * warmup_progress
				for pg in optimizer.param_groups:
					pg['lr'] = args.lr * factor
			running = 0.0
			count = 0
			pbar = tqdm(total=len(train_loader), desc=f"[{side_run}] Epoch {epoch:03d}", leave=False) if _HAS_TQDM else None
			for it, batch in enumerate(train_loader, start=1):
				foot = batch['foot'].to(device)
				template = batch['template'].to(device)
				target = batch['insole'].to(device)

				optimizer.zero_grad(set_to_none=True)
				foot_in = foot.transpose(2, 1).contiguous()
				global_feat = encoder(foot_in)
				pred = regressor(template, global_feat)

				cd = chamfer_distance(
					pred, target,
					reduction='mean',
					chunk=args.cd_chunk,
					normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
				)
				loss = args.global_cd_weight * cd
				lcd_val = torch.tensor(0.0, device=device)
				if args.local_cd_weight > 0:
					lcd_val = local_chamfer_distance(
						pred, target,
						num_patches=args.local_cd_patches,
						radius=args.local_cd_radius,
						reduction='mean',
						chunk=args.cd_chunk,
						normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
					)
					loss = loss + args.local_cd_weight * lcd_val
				e_val = torch.tensor(0.0, device=device)
				if getattr(args, 'emd_weight', 0.0) > 0:
					e_val = emd_loss(
						pred, target,
						epsilon=getattr(args, 'emd_eps', 0.02),
						num_iters=int(getattr(args, 'emd_iters', 50)),
						subset_size=int(getattr(args, 'emd_subset', 1024)),
						reduction='mean',
						normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
					)
					loss = loss + args.emd_weight * e_val
				if args.offset_l2_weight > 0:
					offsets = pred - template
					reg_l2 = torch.mean(torch.sum(offsets * offsets, dim=-1))
					loss = loss + args.offset_l2_weight * reg_l2
				else:
					reg_l2 = torch.tensor(0.0, device=device)

				loss.backward()
				# 梯度裁剪与记录梯度范数
				grad_total_norm = torch.tensor(0.0)
				if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
					grad_total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.clip_grad_norm))
				optimizer.step()

				bs = foot.size(0)
				running += loss.item() * bs
				count += bs

				if it % 10 == 0:
					current_lr = optimizer.param_groups[0]['lr']
					msg = (
						f"[{side_run}] Ep{epoch:03d} It{it:04d}/{len(train_loader)} | "
						f"loss={loss.item():.6f} | cd={cd.item():.6f} | lcd={lcd_val.item():.6f} | emd={e_val.item():.6f} | "
						f"regL2={reg_l2.item():.6f} | grad_norm={float(grad_total_norm):.4f} | lr={current_lr:.2e}"
					)
					logger.info(msg)
				if pbar is not None:
					pbar.set_postfix(loss=f"{loss.item():.6f}")
					pbar.update(1)
			if pbar is not None:
				pbar.close()

			train_loss = running / max(1, count)
			local_cfg = None
			if args.local_cd_weight > 0:
				local_cfg = {'weight': args.local_cd_weight, 'patches': args.local_cd_patches, 'radius': args.local_cd_radius}
			emd_cfg = None
			if getattr(args, 'emd_weight', 0.0) > 0:
				emd_cfg = {'weight': args.emd_weight, 'eps': args.emd_eps, 'iters': args.emd_iters, 'subset': args.emd_subset}
			weights = {'cd': args.global_cd_weight}
			val_loss = evaluate(encoder, regressor, val_loader, device, cd_chunk=args.cd_chunk, local_cd_cfg=local_cfg, cd_normalize=bool(getattr(args, 'cd_normalize_loss', False)), emd_cfg=emd_cfg, weights=weights)
			hist_train_loss.append(train_loss)
			hist_val_loss.append(val_loss)

			# 调度器
			if scheduler is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
				if isinstance(scheduler, ReduceLROnPlateau):
					scheduler.step(val_loss)
				else:
					scheduler.step()

			# 保存最佳模型与早停
			if val_loss < best_val - args.min_delta:
				best_val = val_loss
				best_val_epoch = epoch
				patience_counter = 0
				ckpt = {
					'epoch': epoch,
					'encoder': encoder.state_dict(),
					'regressor': regressor.state_dict(),
					'optimizer': optimizer.state_dict(),
					'best_val': best_val,
					'args': vars(args),
					'side': side_run,
				}
				torch.save(ckpt, sub_save / 'best.pth')
				logger.info(f"[{side_run}] 保存最佳模型 (val={best_val:.6f}) -> {sub_save / 'best.pth'}")
			else:
				patience_counter += 1
				logger.info(f"[{side_run}] Ep{epoch:03d} 训练/验证: {train_loss:.6f}/{val_loss:.6f} | 耗时: {time.time()-start_t:.1f}s | 耐心: {patience_counter}/{args.patience}")

			if args.early_stopping and patience_counter >= args.patience:
				logger.info(f"[{side_run}] 提前停止触发！最佳验证损失: {best_val:.6f} (第 {best_val_epoch} 轮)")
				break

		# 保存最终权重
		final_ckpt = {
			'epoch': len(hist_train_loss),
			'encoder': encoder.state_dict(),
			'regressor': regressor.state_dict(),
			'args': vars(args),
			'side': side_run,
		}
		torch.save(final_ckpt, sub_save / 'final.pth')
		logger.info(f"[{side_run}] 训练结束，最终模型已保存到 {sub_save / 'final.pth'} | 最佳 val={best_val:.6f} @ epoch {best_val_epoch}")

		# 保存曲线与 CSV
		try:
			log_dir_path = sub_log
			log_dir_path.mkdir(parents=True, exist_ok=True)
			csv_path = log_dir_path / 'metrics.csv'
			with open(csv_path, 'w', encoding='utf-8') as f:
				f.write('epoch,train_loss,val_loss\n')
				for i, (tl, vl) in enumerate(zip(hist_train_loss, hist_val_loss), start=1):
					f.write(f"{i},{tl:.8f},{vl:.8f}\n")
			logger.info(f"[{side_run}] 指标已保存到: {csv_path}")
			if _HAS_MPL:
				plt.figure(figsize=(8, 5))
				epochs_range = range(1, len(hist_train_loss) + 1)
				plt.plot(epochs_range, hist_train_loss, label='train_loss')
				plt.plot(epochs_range, hist_val_loss, label='val_loss')
				if best_val_epoch > 0:
					plt.axvline(x=best_val_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_val_epoch})')
				plt.xlabel('epoch')
				plt.ylabel('loss')
				plt.title(f'Loss Curve ({side_run})')
				plt.legend()
				plt.tight_layout()
				loss_png = log_dir_path / 'loss_curve.png'
				plt.savefig(loss_png, dpi=150)
				plt.close()
				logger.info(f"[{side_run}] Loss 曲线已保存: {loss_png}")
			else:
				logger.info(f"[{side_run}] matplotlib 不可用，已保存 CSV，可用外部工具绘图。")
		except Exception as e:
			logger.warning(f"[{side_run}] 保存曲线失败: {e}")


if __name__ == '__main__':
	raise SystemExit(train())

import os
import sys
from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	_HAS_MPL = True
except Exception:
	_HAS_MPL = False

try:
	from tqdm import tqdm
	_HAS_TQDM = True
except Exception:
	_HAS_TQDM = False

# 项目内模块导入
from datasets.foot_insole_dataset import FootInsoleDataset
from Models.dgcnn_encoder import DGCNNEncoder
from Models.deformation_net import DeformationNet
from losses.chamfer import chamfer_distance, local_chamfer_distance
from losses.emd import emd_loss
from utils.logger import setup_logger


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description='训练 Deformation Network（足模 -> 鞋垫）')
	parser.add_argument('--data-root', type=str, default=str(Path('data') / 'pointcloud'), help='数据根目录')
	parser.add_argument('--template', type=str, default=str(Path('Templates') / 'average_insole_template.npz'), help='模板鞋垫点云路径(.npz)。当 --side=LR 时，若未显式提供左右模板，将尝试自动使用 *_L.npz 与 *_R.npz')
	parser.add_argument('--template-L', type=str, default=None, help='左脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--template-R', type=str, default=None, help='右脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--no-interactive', action='store_true', help='关闭交互式确认，直接按参数启动训练')
	parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR', help='训练侧别：L 或 R；LR 表示顺序训练左右脚（先 L 后 R）')
	# 数据集参数
	parser.add_argument('--num-points', type=int, default=4096, help='每个样本点数')
	parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例（0~1）')
	parser.add_argument('--no-shuffle', action='store_true', help='禁用点顺序随机打乱')
	parser.add_argument('--use-normals', action='store_true', help='足模输入是否包含法向(6维)')
	parser.add_argument('--normalize', type=str, choices=['sphere', 'cube', 'center'], default='center', help='点云标准化方式')
	parser.add_argument('--local-cd-weight', type=float, default=0.0, help='局部 Chamfer Distance 的损失权重 (0 关闭)')
	parser.add_argument('--local-cd-patches', type=int, default=64, help='每样本局部 patch 数量')
	parser.add_argument('--local-cd-radius', type=float, default=0.2, help='局部邻域半径（与坐标同单位）')
	parser.add_argument('--batch-size', type=int, default=12)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-3, help='优化器权重衰减')
	parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau', 'none'], default='plateau')
	parser.add_argument('--step-size', type=int, default=40, help='StepLR 的步长')
	parser.add_argument('--gamma', type=float, default=0.5, help='StepLR 衰减率')
	# warmup 与 Plateau 调度
	parser.add_argument('--warmup-epochs', type=int, default=5, help='学习率 warmup 轮数（0 关闭）')
	parser.add_argument('--warmup-start-factor', type=float, default=0.1, help='warmup 起始因子，相对 base lr')
	parser.add_argument('--plateau-patience', type=int, default=5, help='ReduceLROnPlateau 的耐心轮数')
	parser.add_argument('--plateau-factor', type=float, default=0.5, help='ReduceLROnPlateau 的衰减因子')
	parser.add_argument('--plateau-min-lr', type=float, default=1e-6, help='ReduceLROnPlateau 的最小学习率')
	# 提前停止
	parser.add_argument('--early-stopping', action='store_true', help='启用提前停止机制')
	parser.add_argument('--patience', type=int, default=20, help='提前停止耐心值')
	parser.add_argument('--min-delta', type=float, default=1e-6, help='提前停止最小改善阈值')
	# 模型参数
	parser.add_argument('--dgcnn-k', type=int, default=20, help='DGCNN 邻域点数 k')
	parser.add_argument('--dgcnn-feat-dim', type=int, default=256, help='DGCNN 输出全局特征维度')
	parser.add_argument('--dgcnn-dropout', type=float, default=0.3, help='DGCNN 内部 Dropout 概率')
	parser.add_argument('--dgcnn-multi-scale-ks', type=str, default='10,20,30', help='多尺度 EdgeConv 的 k 列表，如: 10,20,30；留空禁用')
	parser.add_argument('--hidden-dims', type=str, default='256,256,128', help='回归器隐藏层维度（逗号分隔）')
	parser.add_argument('--mlp-dropout', type=float, default=0.4, help='回归器 MLP 的 Dropout 概率')
	# 损失/评估参数
	parser.add_argument('--cd-chunk', type=int, default=1024, help='Chamfer 距离计算分块大小')
	parser.add_argument('--cd-normalize-loss', action='store_true', help='对 Chamfer 距离按几何尺度做归一化（保持真实尺寸，仅缩放损失数值）')
	parser.add_argument('--global-cd-weight', type=float, default=0.5, help='全局 Chamfer Distance 的损失权重 (α)')
	parser.add_argument('--emd-weight', type=float, default=0.5, help='EMD（Sinkhorn 近似）的损失权重 (β)')
	parser.add_argument('--emd-eps', type=float, default=0.02, help='EMD 的 Sinkhorn 熵正则强度 epsilon')
	parser.add_argument('--emd-iters', type=int, default=50, help='EMD 的 Sinkhorn 迭代次数')
	parser.add_argument('--emd-subset', type=int, default=1024, help='EMD 的子采样点数（降低显存/算力消耗）')
	parser.add_argument('--local-cd-weight', type=float, default=0.1, help='局部 Chamfer Distance 的损失权重 (0 关闭)')
	parser.add_argument('--offset-l2-weight', type=float, default=0.0, help='位移向量 L2 正则项权重')
	parser.add_argument('--clip-grad-norm', type=float, default=1.0, help='梯度裁剪的 max_norm（0 或负数关闭）')
	# 数据增强参数
	parser.add_argument('--aug-enable', action='store_true', help='启用训练阶段点云增强')
	parser.add_argument('--aug-multiplier', type=int, default=8, help='每个原样本生成的增强版本数量（仅训练集）')
	parser.add_argument('--aug-jitter-sigma', type=str, default='0.002,0.01', help='点抖动噪声的σ范围')
	parser.add_argument('--aug-dropout-patches', type=str, default='0,3', help='局部 dropout 的 patch 数量范围')
	parser.add_argument('--aug-dropout-radius', type=str, default='0.05,0.15', help='局部 dropout 的半径范围')
	parser.add_argument('--aug-normal-shift', type=str, default='0.0,0.02', help='沿法向的小偏移幅度范围（仅当输入包含法向时生效）')
	parser.add_argument('--aug-resample', type=str, choices=['none', 'uniform', 'poisson'], default='poisson', help='改变点密度的重采样方式')
	parser.add_argument('--aug-uniform-keep', type=str, default='0.6,1.0', help='uniform 重采样时保留比例范围')
	parser.add_argument('--aug-poisson-voxel', type=str, default='0.01,0.04', help='poisson 近似(体素)采样的体素尺寸')
	parser.add_argument('--save-dir', type=str, default='checkpoints')
	parser.add_argument('--log-dir', type=str, default=str(Path('Log') / 'train'))
	parser.add_argument('--num-workers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--seed', type=int, default=42)
	return parser.parse_args(argv)


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	try:
		dims = tuple(int(p) for p in parts)
	except Exception as e:
		raise ValueError(f"hidden-dims 解析失败: {s}") from e
	if len(dims) == 0:
		raise ValueError("hidden-dims 不能为空")
	return dims


def _clean_old_logs(log_dir: Path, name_prefix: str) -> None:
	try:
		if not log_dir.exists():
			return
		for p in log_dir.glob(f"{name_prefix}_*.log"):
			try:
				p.unlink(missing_ok=True)
			except Exception:
				pass
	except Exception:
		pass


def make_dataloaders(args, side: str | None = None, template_path: str | None = None):
	side_norm = None if (side is None or str(side).upper() == 'LR') else str(side).upper()
	train_set = FootInsoleDataset(
		data_root=args.data_root,
		split='train',
		use_normals=args.use_normals,
		template_path=template_path or args.template,
		num_points=args.num_points,
		val_ratio=args.val_ratio,
		random_shuffle_points=(not args.no_shuffle),
		side=side_norm,
	)
	train_set.normalize_mode = args.normalize
	if getattr(args, 'aug_enable', False):
		train_set.augment_enable = True
		train_set.augment_multiplier = max(1, int(getattr(args, 'aug_multiplier', 8)))
		def _parse_range_pair(s: str, typ=float):
			parts = [p.strip() for p in str(s).split(',') if p.strip()]
			if len(parts) == 1:
				v = typ(parts[0])
				return (v, v)
			if len(parts) >= 2:
				return (typ(parts[0]), typ(parts[1]))
			return (0.0, 0.0)
		train_set.aug_jitter_sigma_range = _parse_range_pair(getattr(args, 'aug_jitter_sigma', '0.002,0.01'), float)
		train_set.aug_dropout_patches_range = _parse_range_pair(getattr(args, 'aug_dropout_patches', '0,3'), int)
		train_set.aug_dropout_radius_range = _parse_range_pair(getattr(args, 'aug_dropout_radius', '0.05,0.15'), float)
		train_set.aug_normal_shift_range = _parse_range_pair(getattr(args, 'aug_normal_shift', '0.0,0.02'), float)
		train_set.aug_resample_mode = str(getattr(args, 'aug_resample', 'poisson')).lower()
		train_set.aug_uniform_keep_range = _parse_range_pair(getattr(args, 'aug_uniform_keep', '0.6,1.0'), float)
		train_set.aug_poisson_voxel_range = _parse_range_pair(getattr(args, 'aug_poisson_voxel', '0.01,0.04'), float)

	val_set = FootInsoleDataset(
		data_root=args.data_root,
		split='val',
		use_normals=args.use_normals,
		template_path=template_path or args.template,
		num_points=args.num_points,
		val_ratio=args.val_ratio,
		random_shuffle_points=False,
		side=side_norm,
	)
	val_set.normalize_mode = args.normalize
	train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
	val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
	return train_loader, val_loader


def _parse_ms_ks(s: str | None) -> tuple[int, ...]:
	if not s:
		return tuple()
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	ks: list[int] = []
	for p in parts:
		try:
			v = int(p)
			if v > 0:
				ks.append(v)
		except Exception:
			continue
	return tuple(sorted(set(ks)))


def build_model(args):
	input_dims = 6 if args.use_normals else 3
	ms_ks = _parse_ms_ks(getattr(args, 'dgcnn_multi_scale_ks', '10,20,30'))
	encoder = DGCNNEncoder(
		input_dims=input_dims,
		k=args.dgcnn_k,
		feat_dim=args.dgcnn_feat_dim,
		dropout_p=getattr(args, 'dgcnn_dropout', 0.1),
		multi_scale_ks=ms_ks,
	)
	regressor = DeformationNet(global_feat_dim=args.dgcnn_feat_dim, hidden_dims=_parse_hidden_dims(args.hidden_dims), dropout_p=getattr(args, 'mlp_dropout', 0.2))
	return encoder, regressor


def evaluate(encoder, regressor, loader, device, cd_chunk: int, local_cd_cfg: dict[str, float | int] | None = None, cd_normalize: bool = False, emd_cfg: dict | None = None, weights: dict | None = None):
	encoder.eval()
	regressor.eval()
	total_loss = 0.0
	total_cnt = 0
	with torch.no_grad():
		for batch in loader:
			foot = batch['foot'].to(device)
			template = batch['template'].to(device)
			target = batch['insole'].to(device)

			foot_in = foot.transpose(2, 1).contiguous()
			global_feat = encoder(foot_in)
			pred = regressor(template, global_feat)
			cd = chamfer_distance(pred, target, reduction='mean', chunk=cd_chunk, normalize_loss=cd_normalize)
			loss = cd * (weights.get('cd', 1.0) if weights else 1.0)
			if local_cd_cfg and local_cd_cfg.get('weight', 0.0) > 0:
				lcd = local_chamfer_distance(
					pred, target,
					num_patches=int(local_cd_cfg.get('patches', 64)),
					radius=float(local_cd_cfg.get('radius', 0.2)),
					reduction='mean',
					chunk=cd_chunk,
					normalize_loss=cd_normalize,
				)
				loss = loss + float(local_cd_cfg['weight']) * lcd
			if emd_cfg and (emd_cfg.get('weight', 0.0) > 0):
				e = emd_loss(
					pred, target,
					epsilon=float(emd_cfg.get('eps', 0.02)),
					num_iters=int(emd_cfg.get('iters', 50)),
					subset_size=int(emd_cfg.get('subset', 1024)),
					reduction='mean',
					normalize_loss=cd_normalize,
				)
				loss = loss + float(emd_cfg['weight']) * e
			bs = foot.size(0)
			total_loss += loss.item() * bs
			total_cnt += bs
	avg_loss = total_loss / max(1, total_cnt)
	return avg_loss


def _resolve_templates_for_side(args):
	side = (args.side or 'LR').upper()
	if side != 'LR':
		return side, args.template
	tpl_L = args.template_L
	tpl_R = args.template_R
	if (tpl_L is None) or (tpl_R is None):
		base = Path(args.template)
		if base.suffix.lower() == '.npz':
			stem = base.stem
			parent = base.parent
			tpl_L = tpl_L or str(parent / f"{stem}_L.npz")
			tpl_R = tpl_R or str(parent / f"{stem}_R.npz")
	return side, (tpl_L, tpl_R)


def train():
	args = parse_args()
	set_seed(args.seed)
	device = torch.device(args.device)
	log_dir_path = Path(args.log_dir)
	try:
		for p in log_dir_path.glob("train_deformnet_*.log"):
			p.unlink(missing_ok=True)
	except Exception:
		pass
	logger = setup_logger(log_dir_path, name='train_deformnet')
	logger.info(f"使用设备: {device}")

	side, tpl = _resolve_templates_for_side(args)
	sides_to_run = ['L', 'R'] if side == 'LR' else [side]

	for side_run in sides_to_run:
		if isinstance(tpl, tuple):
			tpl_L, tpl_R = tpl
			template_path = tpl_L if side_run == 'L' else tpl_R
		else:
			template_path = args.template

		sub_save = Path(args.save_dir) / side_run
		sub_log = Path(args.log_dir) / side_run
		sub_save.mkdir(parents=True, exist_ok=True)
		sub_log.mkdir(parents=True, exist_ok=True)

		logger.info(f"开始训练侧别: {side_run} | 模板: {template_path}")

		train_loader, val_loader = make_dataloaders(args, side=side_run, template_path=template_path)
		encoder, regressor = build_model(args)
		encoder.to(device)
		regressor.to(device)

		params = list(encoder.parameters()) + list(regressor.parameters())
		optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
		if args.scheduler == 'cosine':
			scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
		elif args.scheduler == 'step':
			scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
		elif args.scheduler == 'plateau':
			scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.plateau_min_lr)
		else:
			scheduler = None

		best_val = float('inf')
		patience_counter = 0
		best_val_epoch = 0
		hist_train_loss: list[float] = []
		hist_val_loss: list[float] = []

		for epoch in range(1, args.epochs + 1):
			encoder.train()
			regressor.train()
			if args.warmup_epochs and epoch <= args.warmup_epochs:
				warmup_progress = 0.0 if args.warmup_epochs <= 1 else float(epoch - 1) / float(args.warmup_epochs - 1)
				factor = args.warmup_start_factor + (1.0 - args.warmup_start_factor) * warmup_progress
				for pg in optimizer.param_groups:
					pg['lr'] = args.lr * factor
			running = 0.0
			count = 0
			pbar = tqdm(total=len(train_loader), desc=f"[{side_run}] Epoch {epoch:03d}", leave=False) if _HAS_TQDM else None
			for it, batch in enumerate(train_loader, start=1):
				foot = batch['foot'].to(device)
				template = batch['template'].to(device)
				target = batch['insole'].to(device)

				optimizer.zero_grad(set_to_none=True)
				foot_in = foot.transpose(2, 1).contiguous()
				global_feat = encoder(foot_in)
				pred = regressor(template, global_feat)

				cd = chamfer_distance(
					pred, target,
					reduction='mean',
					chunk=args.cd_chunk,
					normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
				)
				loss = args.global_cd_weight * cd
				if args.local_cd_weight > 0:
					lcd = local_chamfer_distance(
						pred, target,
						num_patches=args.local_cd_patches,
						radius=args.local_cd_radius,
						reduction='mean',
						chunk=args.cd_chunk,
						normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
					)
					loss = loss + args.local_cd_weight * lcd
				if getattr(args, 'emd_weight', 0.0) > 0:
					e = emd_loss(
						pred, target,
						epsilon=getattr(args, 'emd_eps', 0.02),
						num_iters=int(getattr(args, 'emd_iters', 50)),
						subset_size=int(getattr(args, 'emd_subset', 1024)),
						reduction='mean',
						normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
					)
					loss = loss + args.emd_weight * e
				if args.offset_l2_weight > 0:
					offsets = pred - template
					reg_l2 = torch.mean(torch.sum(offsets * offsets, dim=-1))
					loss = loss + args.offset_l2_weight * reg_l2
				loss.backward()
				if getattr(args, 'clip_grad_norm', 0.0) and args.clip_grad_norm > 0:
					torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.clip_grad_norm))
				optimizer.step()

				bs = foot.size(0)
				running += loss.item() * bs
				count += bs

				if it % 10 == 0:
					logger.info(f"[{side_run}] Epoch {epoch:03d} | Iter {it:04d}/{len(train_loader)} | loss={loss.item():.6f}")
				if pbar is not None:
					pbar.set_postfix(loss=f"{loss.item():.6f}")
					pbar.update(1)
			if pbar is not None:
				pbar.close()

			train_loss = running / max(1, count)
			local_cfg = None
			if args.local_cd_weight > 0:
				local_cfg = {'weight': args.local_cd_weight, 'patches': args.local_cd_patches, 'radius': args.local_cd_radius}
			emd_cfg = None
			if getattr(args, 'emd_weight', 0.0) > 0:
				emd_cfg = {'weight': args.emd_weight, 'eps': args.emd_eps, 'iters': args.emd_iters, 'subset': args.emd_subset}
			weights = {'cd': args.global_cd_weight}
			val_loss = evaluate(encoder, regressor, val_loader, device, cd_chunk=args.cd_chunk, local_cd_cfg=local_cfg, cd_normalize=bool(getattr(args, 'cd_normalize_loss', False)), emd_cfg=emd_cfg, weights=weights)
			logger.info(f"[{side_run}] [Epoch {epoch:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

			if scheduler is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
				if isinstance(scheduler, ReduceLROnPlateau):
					scheduler.step(val_loss)
				else:
					scheduler.step()

			if val_loss < best_val - args.min_delta:
				best_val = val_loss
				best_val_epoch = epoch
				patience_counter = 0
				ckpt = {
					'epoch': epoch,
					'encoder': encoder.state_dict(),
					'regressor': regressor.state_dict(),
					'optimizer': optimizer.state_dict(),
					'best_val': best_val,
					'args': vars(args),
					'side': side_run,
				}
				torch.save(ckpt, sub_save / 'best.pth')
				logger.info(f"[{side_run}] 保存最佳模型 (val={best_val:.6f}) -> {sub_save / 'best.pth'}")
			else:
				patience_counter += 1
				logger.info(f"[{side_run}] 验证损失未改善，耐心计数: {patience_counter}/{args.patience}")

			if args.early_stopping and patience_counter >= args.patience:
				logger.info(f"[{side_run}] 提前停止触发！最佳验证损失: {best_val:.6f} (第 {best_val_epoch} 轮)")
				break

		actual_epochs = epoch
		final_ckpt = {
			'epoch': actual_epochs,
			'encoder': encoder.state_dict(),
			'regressor': regressor.state_dict(),
			'args': vars(args),
			'side': side_run,
		}
		torch.save(final_ckpt, sub_save / 'final.pth')

		try:
			log_dir_path = sub_log
			log_dir_path.mkdir(parents=True, exist_ok=True)
			csv_path = log_dir_path / 'metrics.csv'
			with open(csv_path, 'w', encoding='utf-8') as f:
				f.write('epoch,train_loss,val_loss\n')
				for i, (tl, vl) in enumerate(zip(hist_train_loss, hist_val_loss), start=1):
					f.write(f"{i},{tl:.8f},{vl:.8f}\n")
			logger.info(f"[{side_run}] 指标已保存到: {csv_path}")
			if _HAS_MPL:
				plt.figure(figsize=(8, 5))
				epochs_range = range(1, len(hist_train_loss) + 1)
				plt.plot(epochs_range, hist_train_loss, label='train_loss')
				plt.plot(epochs_range, hist_val_loss, label='val_loss')
				if args.early_stopping and best_val_epoch > 0:
					plt.axvline(x=best_val_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_val_epoch})')
				plt.xlabel('epoch')
				plt.ylabel('loss')
				plt.title(f'Loss Curve ({side_run})')
				plt.legend()
				plt.tight_layout()
				loss_png = log_dir_path / 'loss_curve.png'
				plt.savefig(loss_png, dpi=150)
				plt.close()
				logger.info(f"[{side_run}] Loss 曲线已保存: {loss_png}")
			else:
				logger.info(f"[{side_run}] matplotlib 不可用，已保存 CSV，可用外部工具绘图。")
		except Exception as e:
			logger.warning(f"[{side_run}] 保存曲线失败: {e}")


if __name__ == '__main__':
	raise SystemExit(train())

