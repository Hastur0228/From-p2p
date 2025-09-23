import os
import sys
from pathlib import Path
import argparse
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
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
from losses.chamfer import chamfer_distance
from losses.emd import emd_loss
from losses.fscore import fscore, precision_recall_fscore
from losses.normal_consistency import normal_consistency_loss
from losses.feature_loss import feature_matching_loss, chamfer_feature_loss
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
	parser.add_argument('--template', type=str, default=str(Path('Templates/insoles') / 'average_insoles_template.npz'), help='模板鞋垫点云路径(.npz)。当 --side=LR 时，若未显式提供左右模板，将尝试自动使用 *_L.npz 与 *_R.npz')
	parser.add_argument('--template-L', type=str, default=None, help='左脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--template-R', type=str, default=None, help='右脚模板路径(.npz)，仅在 --side=LR 时使用')
	parser.add_argument('--no-interactive', action='store_true', help='关闭交互式确认，直接按参数启动训练')
	parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR', help='训练侧别：L 或 R；LR 表示顺序训练左右脚（先 L 后 R）')
	# 数据集参数
	parser.add_argument('--num-points', type=int, default=4096, help='每个样本点数（会对输入与标签点云做采样/重复以匹配）')
	parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例（0~1）')
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
	parser.add_argument('--batch-size', type=int, default=16)
	parser.add_argument('--epochs', type=int, default=200)
	parser.add_argument('--lr', type=float, default=5e-4)
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='优化器权重衰减')
	parser.add_argument('--log-interval', type=int, default=10, help='每多少 iter 打印一次训练日志')
	parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau', 'none'], default='plateau')
	parser.add_argument('--step-size', type=int, default=40, help='StepLR 的步长')
	parser.add_argument('--gamma', type=float, default=0.5, help='StepLR 衰减率')
	parser.add_argument('--ema-alpha', type=float, default=0.2, help='EMA 平滑验证损失的系数 (0~1)，0 表示关闭')
	# warmup 与 Plateau 调度
	parser.add_argument('--warmup-epochs', type=int, default=3, help='学习率 warmup 轮数（0 关闭）')
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
	parser.add_argument('--cd-normalize-loss', action='store_true', default=False, help='对 Chamfer 距离按几何尺度做归一化（保持真实尺寸，仅缩放损失数值，默认关闭）')
	parser.add_argument('--normalize-loss-by-mean', action='store_true', default=True, help='按训练过程中的平均CD/EMD值进行归一化（默认启用）')
	parser.add_argument('--global-cd-weight', type=float, default=0.5, help='全局 Chamfer Distance 的损失权重 (α)')
	parser.add_argument('--emd-weight', type=float, default=0.5, help='EMD（Sinkhorn 近似）的损失权重 (β)')
	parser.add_argument('--emd-eps', type=float, default=0.02, help='EMD 的 Sinkhorn 熵正则强度 epsilon')
	parser.add_argument('--emd-iters', type=int, default=50, help='EMD 的 Sinkhorn 迭代次数')
	parser.add_argument('--emd-subset', type=int, default=2048, help='EMD 的子采样点数（降低显存/算力消耗）')
	parser.add_argument('--offset-l2-weight', type=float, default=0.0, help='位移向量 L2 正则项权重')
	parser.add_argument('--clip-grad-norm', type=float, default=0.5, help='梯度裁剪的 max_norm（0 或负数关闭）')
	# F-score阈值自动计算为1.5×平均CD距离，无需手动设置
	parser.add_argument('--normal-consistency-weight', type=float, default=0.0, help='法向一致性损失权重 (0 关闭)')
	parser.add_argument('--normal-consistency-k', type=int, default=20, help='法向一致性计算的KNN邻域点数')
	parser.add_argument('--feature-loss-weight', type=float, default=0.0, help='特征匹配损失权重 (0 关闭)')
	parser.add_argument('--feature-loss-type', type=str, choices=['matching', 'chamfer_feature'], default='matching', help='特征损失类型：matching 或 chamfer_feature')
	# 数据增强（训练阶段）
	parser.add_argument('--aug-enable', action='store_true', default=True, help='启用训练阶段点云增强')
	parser.add_argument('--aug-backend', type=str, choices=['train_loop', 'dataset'], default='train_loop', help='增强实现位置：训练循环内或数据集内（避免重复）')
	parser.add_argument('--aug-jitter-sigma', type=float, default=0.005, help='点抖动高斯噪声σ')
	parser.add_argument('--aug-scale-range', type=str, default='0.98,1.02', help='缩放范围，如 0.98,1.02')
	parser.add_argument('--aug-dropout-prob', type=float, default=0.1, help='点丢失概率')
	parser.add_argument('--aug-upsample-factor', type=float, default=1.05, help='点补充倍数（>1 会重复保留点）')
	# 其它
	parser.add_argument('--save-dir', type=str, default='checkpoints')
	parser.add_argument('--log-dir', type=str, default=str(Path('Log') / 'train'))
	parser.add_argument('--num-workers', type=int, default=4)
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--gui', action='store_true', help='使用图形化界面配置参数并启动训练')
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

	def _cast_aug_backend(s: str) -> str:
		s = s.strip().lower()
		if s not in ('train_loop', 'dataset'):
			raise ValueError('非法增强后端')
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
			("StepLR 步长", args.step_size, int),
			("StepLR 衰减gamma", args.gamma, float),
			("warmup 轮数", args.warmup_epochs, int),
			("warmup 起始因子", args.warmup_start_factor, float),
			("Plateau 耐心轮数", args.plateau_patience, int),
			("Plateau 衰减因子", args.plateau_factor, float),
			("Plateau 最小学习率", args.plateau_min_lr, float),
			("EMA 平滑系数", args.ema_alpha, float),
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
			("平均值归一化", getattr(args, 'normalize_loss_by_mean', True), lambda x: x.lower() in ('y','yes','true','1')),
			("全局CD 权重α", args.global_cd_weight, float),
			("EMD 权重β", args.emd_weight, float),
			("EMD eps", args.emd_eps, float),
			("EMD iters", args.emd_iters, int),
			("EMD subset", args.emd_subset, int),
			("位移L2 权重", args.offset_l2_weight, float),
			("梯度裁剪 max_norm", args.clip_grad_norm, float),
			# F-score阈值现在自动计算为1.5×平均CD距离
			("法向一致性权重", getattr(args, 'normal_consistency_weight', 0.0), float),
			("法向一致性 KNN k", getattr(args, 'normal_consistency_k', 20), int),
			("特征损失权重", getattr(args, 'feature_loss_weight', 0.0), float),
			("特征损失类型", getattr(args, 'feature_loss_type', 'matching'), str),
			# 数据增强（新增）
			("数据增强启用", getattr(args, 'aug_enable', False), lambda x: x.lower() in ('y','yes','true','1')),
			("增强后端(train_loop/dataset)", getattr(args, 'aug_backend', 'train_loop'), _cast_aug_backend),
			("抖动噪声σ", getattr(args, 'aug_jitter_sigma', 0.005), float),
			("缩放范围(逗号分隔)", getattr(args, 'aug_scale_range', '0.98,1.02'), str),
			("点丢失概率", getattr(args, 'aug_dropout_prob', 0.1), float),
			("点补充倍数", getattr(args, 'aug_upsample_factor', 1.05), float),
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
			elif label == "StepLR 步长": args.step_size = new_value
			elif label == "StepLR 衰减gamma": args.gamma = new_value
			elif label == "warmup 轮数": args.warmup_epochs = new_value
			elif label == "warmup 起始因子": args.warmup_start_factor = new_value
			elif label == "Plateau 耐心轮数": args.plateau_patience = new_value
			elif label == "Plateau 衰减因子": args.plateau_factor = new_value
			elif label == "Plateau 最小学习率": args.plateau_min_lr = new_value
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
			elif label == "平均值归一化": args.normalize_loss_by_mean = bool(new_value)
			elif label == "全局CD 权重α": args.global_cd_weight = new_value
			elif label == "EMD 权重β": args.emd_weight = new_value
			elif label == "EMD eps": args.emd_eps = new_value
			elif label == "EMD iters": args.emd_iters = new_value
			elif label == "EMD subset": args.emd_subset = new_value
			elif label == "位移L2 权重": args.offset_l2_weight = new_value
			elif label == "梯度裁剪 max_norm": args.clip_grad_norm = new_value
			# F-score阈值自动计算，无需手动设置
			elif label == "法向一致性权重": args.normal_consistency_weight = new_value
			elif label == "法向一致性 KNN k": args.normal_consistency_k = new_value
			elif label == "特征损失权重": args.feature_loss_weight = new_value
			elif label == "特征损失类型": args.feature_loss_type = new_value
			elif label == "数据增强启用": args.aug_enable = bool(new_value)
			elif label == "增强后端(train_loop/dataset)": args.aug_backend = new_value
			elif label == "抖动噪声σ": args.aug_jitter_sigma = new_value
			elif label == "缩放范围(逗号分隔)": args.aug_scale_range = new_value
			elif label == "点丢失概率": args.aug_dropout_prob = new_value
			elif label == "点补充倍数": args.aug_upsample_factor = new_value
			print(f"已修改 {label}: {new_value}")
		except Exception:
			print("请输入有效的数字。")


def apply_data_augmentation(points: torch.Tensor, args) -> torch.Tensor:
	"""
	对输入点云进行轻量数据增强（训练阶段）：
	- 高斯噪声抖动（σ）
	- 全局缩放（±2%）
	- 点丢失+补点（保持点数不变）

	支持 BxNx3 或 Nx3，返回形状与输入一致。
	"""
	if not getattr(args, 'aug_enable', False):
		return points

	original_shape = points.shape
	if points.dim() == 2:  # Nx3 -> 1xNx3
		points = points.unsqueeze(0)

	# B, N, 3
	B, N, _ = points.shape

	# 高斯噪声
	jit_sigma = float(getattr(args, 'aug_jitter_sigma', 0.0))
	if jit_sigma > 0:
		noise = torch.randn_like(points) * jit_sigma
		points = points + noise

	# 全局缩放（每个样本一个缩放因子）
	scale_spec = str(getattr(args, 'aug_scale_range', ''))
	if scale_spec:
		try:
			lo, hi = [float(x) for x in scale_spec.split(',')]
			scales = torch.rand(B, 1, 1, device=points.device, dtype=points.dtype) * (hi - lo) + lo
			points = points * scales
		except Exception:
			pass

	# 点丢失 + 补点（重复保留的点以恢复至 N）
	drop_p = float(getattr(args, 'aug_dropout_prob', 0.0))
	up_factor = float(getattr(args, 'aug_upsample_factor', 1.0))
	if drop_p > 0 or up_factor > 1.0:
		keep_ratio = (1.0 - max(0.0, min(1.0, drop_p))) / max(1e-6, up_factor)
		keep_ratio = float(max(0.05, min(1.0, keep_ratio)))
		new_points = points.clone()
		for b in range(B):
			num_keep = max(1, int(N * keep_ratio))
			perm = torch.randperm(N, device=points.device)
			keep_idx = perm[:num_keep]
			kept = points[b, keep_idx, :]
			if num_keep < N:
				dup_idx = torch.randint(0, num_keep, (N - num_keep,), device=points.device)
				dup = kept[dup_idx, :]
				combined = torch.cat([kept, dup], dim=0)
			else:
				combined = kept[:N, :]
				new_points[b] = combined
		points = new_points

	if original_shape != points.shape:
		points = points.squeeze(0)
	return points


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
	if getattr(args, 'aug_enable', False) and getattr(args, 'aug_backend', 'train_loop') == 'dataset':
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


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
	parts = [p.strip() for p in str(s).split(',') if p.strip()]
	try:
		dims = tuple(int(p) for p in parts)
	except Exception as e:
		raise ValueError(f"hidden-dims 解析失败: {s}") from e
	if len(dims) == 0:
		raise ValueError("hidden-dims 不能为空")
	return dims


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


def evaluate(encoder, regressor, loader, device, cd_chunk: int, cd_normalize: bool = False, emd_cfg: dict | None = None, weights: dict | None = None, fscore_threshold: float | None = None, normalize_loss_by_mean: bool = True, global_cd_mean: float | None = None, global_emd_mean: float | None = None):
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

			# 与训练loss保持一致的归一化算法
			if normalize_loss_by_mean:
				if global_cd_mean is not None:
					cd_for_loss = cd / (global_cd_mean + 1e-8)
				else:
					cd_for_loss = cd / (cd + 1e-8)  # 使用当前值作为基准
			else:
				cd_for_loss = cd

			loss = cd_for_loss * (weights.get('cd', 1.0) if weights else 1.0)

			if emd_cfg and (emd_cfg.get('weight', 0.0) > 0):
				e = emd_loss(
					pred, target,
					epsilon=float(emd_cfg.get('eps', 0.02)),
					num_iters=int(emd_cfg.get('iters', 50)),
					subset_size=int(emd_cfg.get('subset', 1024)),
					reduction='mean',
					normalize_loss=cd_normalize,
				)

				# 对EMD也应用同样的归一化
				if normalize_loss_by_mean:
					if global_emd_mean is not None:
						emd_for_loss = e / (global_emd_mean + 1e-8)
					else:
						emd_for_loss = e / (e + 1e-8)  # 使用当前值作为基准
				else:
					emd_for_loss = e

				loss = loss + float(emd_cfg['weight']) * emd_for_loss
			bs = foot.size(0)
			total_loss += loss.item() * bs
			total_cnt += bs
	avg_loss = total_loss / max(1, total_cnt)

	# 计算F-score
	f_score_val = 0.0
	if hasattr(torch, 'tensor') and len(loader.dataset) > 0:
		# 获取第一个batch的数据来计算F-score
		for batch in loader:
			foot = batch['foot'].to(device)
			template = batch['template'].to(device)
			target = batch['insole'].to(device)

			foot_in = foot.transpose(2, 1).contiguous()
			global_feat = encoder(foot_in)
			pred = regressor(template, global_feat)

			# 如果没有提供阈值，则使用默认值0.01
			threshold = fscore_threshold if fscore_threshold is not None else 0.01
			f_score_val = fscore(pred, target, threshold=threshold, reduction='mean').item()
			break

	return avg_loss, f_score_val


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
		if getattr(args, 'aug_enable', False):
			logger.info(f"数据增强: 启用 | backend={getattr(args,'aug_backend','train_loop')} | σ={getattr(args,'aug_jitter_sigma',0.0)} | scale={getattr(args,'aug_scale_range','')} | dropout={getattr(args,'aug_dropout_prob',0.0)} | upsample={getattr(args,'aug_upsample_factor',1.0)}")
		else:
			logger.info("数据增强: 禁用")

		# Model
		encoder, regressor = build_model(args)
		encoder.to(device)
		regressor.to(device)
		logger.info(f"DGCNN: k={args.dgcnn_k}, feat_dim={args.dgcnn_feat_dim}, dropout={args.dgcnn_dropout}, ms_ks={args.dgcnn_multi_scale_ks}")
		logger.info(f"Regressor: hidden={_parse_hidden_dims(args.hidden_dims)}, mlp_dropout={args.mlp_dropout}")
		logger.info(f"参数量: encoder={_model_num_params(encoder):,} | regressor={_model_num_params(regressor):,}")

		# Optim
		params = list(encoder.parameters()) + list(regressor.parameters())
		optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
		if args.scheduler == 'cosine':
			scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
		elif args.scheduler == 'step':
			scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
		elif args.scheduler == 'plateau':
			scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.plateau_min_lr)
		else:
			scheduler = None

		logger.info(f"损失: α(CD)={args.global_cd_weight}, β(EMD)={args.emd_weight}, γ(法向一致性)={getattr(args, 'normal_consistency_weight', 0.0)}, δ(特征)={getattr(args, 'feature_loss_weight', 0.0)}, offsetL2={args.offset_l2_weight}")
		logger.info(f"归一化: 几何尺度归一化={args.cd_normalize_loss}, 平均值归一化={getattr(args, 'normalize_loss_by_mean', True)}")
		logger.info(f"EMD: eps={args.emd_eps}, iters={args.emd_iters}, subset={args.emd_subset}")
		logger.info(f"法向一致性: 权重={getattr(args, 'normal_consistency_weight', 0.0)}, KNN k={getattr(args, 'normal_consistency_k', 20)}")
		logger.info(f"特征损失: 权重={getattr(args, 'feature_loss_weight', 0.0)}, 类型={getattr(args, 'feature_loss_type', 'matching')}")
		logger.info(f"F-score: 动态阈值=1.5×平均CD距离")
		logger.info(f"梯度裁剪: max_norm={args.clip_grad_norm}")

		# 显示完整的参数配置
		logger.info("=== 训练参数配置 ===")
		logger.info(f"数据配置: 批大小={args.batch_size}, 轮数={args.epochs}, 学习率={args.lr:.2e}")
		logger.info(f"优化器: 权重衰减={args.weight_decay}, 调度器={args.scheduler}")
		if args.scheduler == 'step':
			logger.info(f"StepLR: 步长={args.step_size}, 衰减率={args.gamma}")
		elif args.scheduler == 'plateau':
			logger.info(f"Plateau: 耐心={args.plateau_patience}, 因子={args.plateau_factor}, 最小lr={args.plateau_min_lr}")
		logger.info(f"数据增强: 启用={getattr(args, 'aug_enable', False)}")
		if getattr(args, 'aug_enable', False):
			logger.info(f"增强参数: backend={getattr(args,'aug_backend','train_loop')}, jitter={getattr(args,'aug_jitter_sigma',0.005)}, scale={getattr(args,'aug_scale_range','0.98,1.02')}")
		logger.info(f"模型配置: DGCNN_k={args.dgcnn_k}, 特征维度={args.dgcnn_feat_dim}, 隐藏层={_parse_hidden_dims(args.hidden_dims)}")
		logger.info(f"损失权重: CD={args.global_cd_weight}, EMD={args.emd_weight}, 法向一致性={getattr(args, 'normal_consistency_weight', 0.0)}, 特征={getattr(args, 'feature_loss_weight', 0.0)}, L2正则={args.offset_l2_weight}")
		logger.info(f"EMD配置: eps={args.emd_eps}, iters={args.emd_iters}, subset={args.emd_subset}")
		logger.info(f"法向一致性: KNN_k={getattr(args, 'normal_consistency_k', 20)}")
		logger.info(f"特征损失: 类型={getattr(args, 'feature_loss_type', 'matching')}")
		logger.info(f"其他配置: CD分块={args.cd_chunk}, 点数={args.num_points}, 验证比例={args.val_ratio}")
		logger.info(f"早停配置: 启用={args.early_stopping}, 耐心={args.patience}, 最小改善={args.min_delta}")
		logger.info("==================")

		best_val = float('inf')
		patience_counter = 0
		best_val_epoch = 0
		hist_train_loss: list[float] = []
		hist_train_cd: list[float] = []
		hist_train_emd: list[float] = []
		hist_val_loss: list[float] = []
		val_ema: float | None = None

		# 用于新的归一化方式
		global_cd_mean = None
		global_emd_mean = None

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
			cd_running = 0.0
			emd_running = 0.0
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

				# 新的归一化方式：按训练平均值归一化
				if getattr(args, 'normalize_loss_by_mean', True):
					if global_cd_mean is not None:
						cd_norm = cd / (global_cd_mean + 1e-8)  # 使用全局平均值归一化
					else:
						cd_norm = cd / (cd + 1e-8)  # 使用当前值作为基准，归一化到1
				else:
					cd_norm = cd  # 不启用归一化

				loss = args.global_cd_weight * cd_norm

				e_val = torch.tensor(0.0, device=device)
				emd_norm = torch.tensor(0.0, device=device)
				if getattr(args, 'emd_weight', 0.0) > 0:
					e_val = emd_loss(
						pred, target,
						epsilon=getattr(args, 'emd_eps', 0.02),
						num_iters=int(getattr(args, 'emd_iters', 50)),
						subset_size=int(getattr(args, 'emd_subset', 1024)),
						reduction='mean',
						normalize_loss=bool(getattr(args, 'cd_normalize_loss', False)),
					)

					# 对EMD也应用新的归一化方式
					if getattr(args, 'normalize_loss_by_mean', True):
						if global_emd_mean is not None:
							emd_norm = e_val / (global_emd_mean + 1e-8)  # 使用全局平均值归一化
						else:
							emd_norm = e_val / (e_val + 1e-8)  # 使用当前值作为基准，归一化到1
					else:
						emd_norm = e_val  # 不启用归一化

					loss = loss + args.emd_weight * emd_norm
				# 法向一致性损失
				norm_cons_loss = torch.tensor(0.0, device=device)
				if getattr(args, 'normal_consistency_weight', 0.0) > 0:
					norm_cons_loss = normal_consistency_loss(
						pred, target,
						k=getattr(args, 'normal_consistency_k', 20),
						reduction='mean'
					)
					loss = loss + args.normal_consistency_weight * norm_cons_loss

				# 特征匹配损失
				feat_loss = torch.tensor(0.0, device=device)
				if getattr(args, 'feature_loss_weight', 0.0) > 0:
					# 提取特征用于特征损失
					with torch.no_grad():
						pred_feat = encoder(pred.transpose(2, 1).contiguous())
						target_feat = encoder(target.transpose(2, 1).contiguous())

					if getattr(args, 'feature_loss_type', 'matching') == 'matching':
						feat_loss = feature_matching_loss(
							pred_feat, target_feat, pred, target, reduction='mean'
						)
					else:  # chamfer_feature
						feat_loss = chamfer_feature_loss(pred, target, encoder, reduction='mean')

					loss = loss + args.feature_loss_weight * feat_loss

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
				cd_running += cd.item() * bs
				emd_running += e_val.item() * bs
				count += bs

				log_every = max(1, int(getattr(args, 'log_interval', 10)))
				if (it % log_every == 0) or (it == 1) or (it == len(train_loader)):
					current_lr = optimizer.param_groups[0]['lr']
					# 获取内存使用信息
					mem_info = ""
					if torch.cuda.is_available():
						mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
						mem_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
						mem_info = f" | GPU_mem={mem_allocated:.2f}/{mem_reserved:.2f}GB"

					# 计算平均值
					cd_mean = cd_running / count if count > 0 else 0.0
					emd_mean = emd_running / count if count > 0 else 0.0

					# 显示归一化后的值
					cd_display = cd_norm.item()
					emd_display = emd_norm.item()

					msg = (
						f"[{side_run}] Ep{epoch:03d} It{it:04d}/{len(train_loader)} | "
						f"loss={loss.item():.6f} | cd_norm={cd_display:.6f}({cd_mean:.4f}) | emd_norm={emd_display:.6f}({emd_mean:.4f}) | "
						f"norm_cons={norm_cons_loss.item():.6f} | feat={feat_loss.item():.6f} | "
						f"regL2={reg_l2.item():.6f} | grad_norm={float(grad_total_norm):.4f} | lr={current_lr:.2e}{mem_info}"
					)

					# 添加详细的参数信息（每10个epoch显示一次）
					if it == 1 and epoch % 10 == 1:
						detail_msg = (
							f"[{side_run}] Ep{epoch:03d} 详细参数: "
							f"batch_size={bs} | num_points={args.num_points} | "
							f"CD_weight={args.global_cd_weight} | EMD_weight={args.emd_weight} | "
							f"norm_cons_weight={getattr(args, 'normal_consistency_weight', 0.0)} | "
							f"feat_weight={getattr(args, 'feature_loss_weight', 0.0)} | "
							f"L2_weight={args.offset_l2_weight}"
						)
						logger.info(detail_msg)
						if getattr(args, 'normalize_loss_by_mean', True):
							# 安全地处理可能为None的变量
							cd_mean_str = f"{global_cd_mean:.6f}" if global_cd_mean is not None else "N/A"
							emd_mean_str = f"{global_emd_mean:.6f}" if global_emd_mean is not None else "N/A"

							mean_msg = (
								f"[{side_run}] Ep{epoch:03d} 当前归一化基准: "
								f"CD_mean={cd_mean_str} | EMD_mean={emd_mean_str} | "
								f"归一化启用: CD_norm = CD / CD_mean, EMD_norm = EMD / EMD_mean"
							)
							logger.info(mean_msg)
					logger.info(msg)
				if pbar is not None:
					pbar.set_postfix(loss=f"{loss.item():.6f}")
					pbar.update(1)
			if pbar is not None:
				pbar.close()

			train_loss = running / max(1, count)
			train_cd = cd_running / max(1, count)
			train_emd = emd_running / max(1, count)
			emd_cfg = None
			if getattr(args, 'emd_weight', 0.0) > 0:
				emd_cfg = {'weight': args.emd_weight, 'eps': args.emd_eps, 'iters': args.emd_iters, 'subset': args.emd_subset}
			weights = {'cd': args.global_cd_weight}
			# 动态计算F-score阈值：1.5倍平均CD距离
			dynamic_fscore_threshold = 1.5 * train_cd if train_cd > 0 else getattr(args, 'fscore_threshold', 0.01)

			val_loss, val_fscore = evaluate(
				encoder, regressor, val_loader, device,
				cd_chunk=args.cd_chunk,
				cd_normalize=bool(getattr(args, 'cd_normalize_loss', False)),
				emd_cfg=emd_cfg,
				weights=weights,
				fscore_threshold=dynamic_fscore_threshold,
				normalize_loss_by_mean=getattr(args, 'normalize_loss_by_mean', True),
				global_cd_mean=global_cd_mean,
				global_emd_mean=global_emd_mean
			)

			# EMA 平滑验证损失（可选）
			alpha = float(getattr(args, 'ema_alpha', 0.0))
			if alpha > 0.0:
				val_ema = (alpha * float(val_loss)) + ((1.0 - alpha) * (val_ema if val_ema is not None else float(val_loss)))
			# EMA 平滑验证损失
			alpha = float(getattr(args, 'ema_alpha', 0.0))
			if alpha > 0.0:
				val_ema = (alpha * float(val_loss)) + ((1.0 - alpha) * (val_ema if val_ema is not None else float(val_loss)))
			hist_train_loss.append(train_loss)
			hist_train_cd.append(train_cd)
			hist_train_emd.append(train_emd)
			hist_val_loss.append(val_loss)

			# 更新全局平均值用于新的归一化方式
			if getattr(args, 'normalize_loss_by_mean', True):
				if global_cd_mean is None:
					global_cd_mean = train_cd
				else:
					global_cd_mean = 0.9 * global_cd_mean + 0.1 * train_cd  # 指数移动平均

				if global_emd_mean is None:
					global_emd_mean = train_emd
				else:
					global_emd_mean = 0.9 * global_emd_mean + 0.1 * train_emd  # 指数移动平均

			# 调度器
			if scheduler is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
				prev_lr = optimizer.param_groups[0]['lr']
				if isinstance(scheduler, ReduceLROnPlateau):
					scheduler.step(val_ema if (val_ema is not None) else val_loss)
				else:
					scheduler.step()
				new_lr = optimizer.param_groups[0]['lr']
				if abs(new_lr - prev_lr) > 1e-12:
					logger.info(f"[{side_run}] Ep{epoch:03d} 调度器更新: lr {prev_lr:.2e} -> {new_lr:.2e}")

			# 每个 epoch 均输出一次训练/验证损失（调度器之后，打印最新学习率）
			elapsed = time.time() - start_t
			current_lr = optimizer.param_groups[0]['lr']

			# 获取更多系统信息
			mem_info = ""
			if torch.cuda.is_available():
				mem_allocated = torch.cuda.memory_allocated() / 1024**3
				mem_reserved = torch.cuda.memory_reserved() / 1024**3
				mem_info = f" | GPU_mem={mem_allocated:.2f}/{mem_reserved:.2f}GB"

			# 计算训练速度（samples/second）
			train_speed = len(train_loader) * args.batch_size / elapsed if elapsed > 0 else 0

			# 计算归一化后的训练CD和EMD
			if getattr(args, 'normalize_loss_by_mean', True):
				if global_cd_mean is not None and global_emd_mean is not None:
					train_cd_norm = train_cd / (global_cd_mean + 1e-8)
					train_emd_norm = train_emd / (global_emd_mean + 1e-8)
				else:
					# 如果全局平均值不存在，使用训练平均值作为基准
					train_cd_norm = train_cd / (train_cd + 1e-8) if train_cd > 0 else 0.0
					train_emd_norm = train_emd / (train_emd + 1e-8) if train_emd > 0 else 0.0
			else:
				train_cd_norm = train_cd
				train_emd_norm = train_emd

			if val_ema is not None:
				logger.info(f"[{side_run}] Ep{epoch:03d} 训练/验证: {train_loss:.6f}/{val_loss:.6f} (EMA={val_ema:.6f}) | CD_norm={train_cd_norm:.4f} | EMD_norm={train_emd_norm:.4f} | F-score={val_fscore:.4f}(阈值={dynamic_fscore_threshold:.4f}) | lr={current_lr:.2e} | speed={train_speed:.1f}samples/s | time={elapsed:.1f}s{mem_info}")
			else:
				logger.info(f"[{side_run}] Ep{epoch:03d} 训练/验证: {train_loss:.6f}/{val_loss:.6f} | CD_norm={train_cd_norm:.4f} | EMD_norm={train_emd_norm:.4f} | F-score={val_fscore:.4f}(阈值={dynamic_fscore_threshold:.4f}) | lr={current_lr:.2e} | speed={train_speed:.1f}samples/s | time={elapsed:.1f}s{mem_info}")

			# 每5个epoch显示一次详细的训练统计信息
			if epoch % 5 == 0:
				logger.info(f"[{side_run}] Ep{epoch:03d} 统计: 最佳验证={best_val:.6f}@{best_val_epoch} | 耐心={patience_counter}/{args.patience} | 数据增强={getattr(args, 'aug_enable', False)}")

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
				logger.info(f"[{side_run}] Ep{epoch:03d} | 耗时: {time.time()-start_t:.1f}s | 耐心: {patience_counter}/{args.patience}")

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
		# 计算训练统计信息
		total_epochs = len(hist_train_loss)
		total_time = time.time() - train_start_time if 'train_start_time' in locals() else 0
		avg_epoch_time = total_time / total_epochs if total_epochs > 0 else 0
		total_samples = total_epochs * len(train_loader) * args.batch_size
		avg_speed = total_samples / total_time if total_time > 0 else 0

		logger.info(f"[{side_run}] 训练结束，最终模型已保存到 {sub_save / 'final.pth'} | 最佳 val={best_val:.6f} @ epoch {best_val_epoch}")
		logger.info(f"[{side_run}] 训练统计: 总轮数={total_epochs} | 总时间={total_time:.1f}s | 平均每轮={avg_epoch_time:.1f}s | 平均速度={avg_speed:.1f}samples/s")
		# 计算最终的归一化值
		if getattr(args, 'normalize_loss_by_mean', True):
			if global_cd_mean is not None and global_emd_mean is not None:
				final_cd_norm = hist_train_cd[-1] / (global_cd_mean + 1e-8)
				final_emd_norm = hist_train_emd[-1] / (global_emd_mean + 1e-8)
			else:
				# 如果全局平均值不存在，使用历史平均值作为基准
				final_cd_norm = hist_train_cd[-1] / (hist_train_cd[-1] + 1e-8) if hist_train_cd[-1] > 0 else 0.0
				final_emd_norm = hist_train_emd[-1] / (hist_train_emd[-1] + 1e-8) if hist_train_emd[-1] > 0 else 0.0
		else:
			final_cd_norm = hist_train_cd[-1]
			final_emd_norm = hist_train_emd[-1]

		logger.info(f"[{side_run}] 最终性能: 训练loss={hist_train_loss[-1]:.6f} | 验证loss={hist_val_loss[-1]:.6f} | CD_norm={final_cd_norm:.4f} | EMD_norm={final_emd_norm:.4f} | F-score={val_fscore:.4f}")
		logger.info(f"[{side_run}] 最佳性能: 验证loss={best_val:.6f} @ epoch {best_val_epoch} | 改善次数={patience_counter}")
		logger.info(f"[{side_run}] 最终归一化基准: CD_mean={global_cd_mean:.6f if global_cd_mean is not None else 'N/A'} | EMD_mean={global_emd_mean:.6f if global_emd_mean is not None else 'N/A'}")

		# 保存曲线与 CSV
		try:
			log_dir_path = sub_log
			log_dir_path.mkdir(parents=True, exist_ok=True)
			csv_path = log_dir_path / 'metrics.csv'
			with open(csv_path, 'w', encoding='utf-8') as f:
				f.write('epoch,train_loss,train_cd,train_emd,train_cd_norm,train_emd_norm,val_loss\n')
				for i, (tl, tcd, temd, vl) in enumerate(zip(hist_train_loss, hist_train_cd, hist_train_emd, hist_val_loss), start=1):
					# 计算归一化值
					if getattr(args, 'normalize_loss_by_mean', True):
						if global_cd_mean is not None and global_emd_mean is not None:
							cd_norm_val = tcd / (global_cd_mean + 1e-8)
							emd_norm_val = temd / (global_emd_mean + 1e-8)
						else:
							# 如果全局平均值不存在，使用历史值作为基准
							cd_norm_val = tcd / (tcd + 1e-8) if tcd > 0 else 0.0
							emd_norm_val = temd / (temd + 1e-8) if temd > 0 else 0.0
					else:
						cd_norm_val = tcd
						emd_norm_val = temd
					f.write(f"{i},{tl:.8f},{tcd:.6f},{temd:.6f},{cd_norm_val:.6f},{emd_norm_val:.6f},{vl:.8f}\n")
			logger.info(f"[{side_run}] 指标已保存到: {csv_path}")
			if _HAS_MPL:
				# Loss曲线
				plt.figure(figsize=(12, 8))

				# 子图1: Loss曲线
				plt.subplot(2, 2, 1)
				epochs_range = range(1, len(hist_train_loss) + 1)
				plt.plot(epochs_range, hist_train_loss, label='train_loss')
				plt.plot(epochs_range, hist_val_loss, label='val_loss')
				if best_val_epoch > 0:
					plt.axvline(x=best_val_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_val_epoch})')
				plt.xlabel('epoch')
				plt.ylabel('loss')
				plt.title(f'Loss Curve ({side_run})')
				plt.legend()
				plt.grid(True, alpha=0.3)

				# 子图2: CD曲线 (原始 + 归一化)
				plt.subplot(2, 2, 2)
				plt.plot(epochs_range, hist_train_cd, label='train_cd (raw)', color='orange', alpha=0.7)
				# 计算并绘制归一化后的CD
				cd_norm_history = []
				for cd_val in hist_train_cd:
					if getattr(args, 'normalize_loss_by_mean', True):
						if global_cd_mean is not None:
							cd_norm_val = cd_val / (global_cd_mean + 1e-8)
						else:
							cd_norm_val = cd_val / (cd_val + 1e-8) if cd_val > 0 else 0.0
					else:
						cd_norm_val = cd_val
					cd_norm_history.append(cd_norm_val)
				plt.plot(epochs_range, cd_norm_history, label='train_cd_norm', color='orange', linewidth=2)
				plt.xlabel('epoch')
				plt.ylabel('CD')
				plt.title(f'CD Curve ({side_run})')
				plt.legend()
				plt.grid(True, alpha=0.3)

				# 子图3: EMD曲线 (原始 + 归一化)
				plt.subplot(2, 2, 3)
				plt.plot(epochs_range, hist_train_emd, label='train_emd (raw)', color='green', alpha=0.7)
				# 计算并绘制归一化后的EMD
				emd_norm_history = []
				for emd_val in hist_train_emd:
					if getattr(args, 'normalize_loss_by_mean', True):
						if global_emd_mean is not None:
							emd_norm_val = emd_val / (global_emd_mean + 1e-8)
						else:
							emd_norm_val = emd_val / (emd_val + 1e-8) if emd_val > 0 else 0.0
					else:
						emd_norm_val = emd_val
					emd_norm_history.append(emd_norm_val)
				plt.plot(epochs_range, emd_norm_history, label='train_emd_norm', color='green', linewidth=2)
				plt.xlabel('epoch')
				plt.ylabel('EMD')
				plt.title(f'EMD Curve ({side_run})')
				plt.legend()
				plt.grid(True, alpha=0.3)

				# 子图4: Loss vs CD/EMD对比 (使用归一化值)
				plt.subplot(2, 2, 4)
				plt.plot(epochs_range, hist_train_loss, label='total_loss', color='blue')
				plt.plot(epochs_range, cd_norm_history, label='cd_norm', color='orange', alpha=0.7)
				plt.plot(epochs_range, emd_norm_history, label='emd_norm', color='green', alpha=0.7)
				plt.xlabel('epoch')
				plt.ylabel('normalized value')
				plt.title(f'Normalized Loss Components ({side_run})')
				plt.legend()
				plt.grid(True, alpha=0.3)

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
