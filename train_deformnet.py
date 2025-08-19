import os
import sys
from pathlib import Path
import argparse
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
try:
	import matplotlib
	matplotlib.use('Agg')  # 无显示环境下保存图片
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
	parser.add_argument('--side', type=str, choices=['L', 'R', 'LR', 'l', 'r', 'lr'], default='LR',
						help='训练侧别：L 或 R；LR 表示顺序训练左右脚（先 L 后 R），并分别保存到子目录。默认 LR')
	# 数据集参数
	parser.add_argument('--num-points', type=int, default=4096, help='每个样本点数（会对输入与标签点云做采样/重复以匹配）')
	parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例（0~1）')
	parser.add_argument('--no-shuffle', action='store_true', help='禁用点顺序随机打乱')
	parser.add_argument('--use-normals', action='store_true', help='足模输入是否包含法向(6维)')
	parser.add_argument('--normalize', type=str, choices=['sphere', 'cube', 'center'], default='center',
						help='点云标准化方式：sphere=单位球(最大范数)，cube=单位立方(最大绝对值)，center=仅中心化')
	parser.add_argument('--local-cd-weight', type=float, default=0.0, help='局部 Chamfer Distance 的损失权重 (0 关闭)')
	parser.add_argument('--local-cd-patches', type=int, default=64, help='每样本局部 patch 数量')
	parser.add_argument('--local-cd-radius', type=float, default=0.2, help='局部邻域半径（与坐标同单位）')
	parser.add_argument('--batch-size', type=int, default=12)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='优化器权重衰减')
	parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'plateau', 'none'], default='cosine')
	parser.add_argument('--step-size', type=int, default=40, help='StepLR 的步长')
	parser.add_argument('--gamma', type=float, default=0.5, help='StepLR 衰减率')
	# 学习率 warmup 与 Plateau 调度
	parser.add_argument('--warmup-epochs', type=int, default=0, help='学习率 warmup 轮数（0 关闭）')
	parser.add_argument('--warmup-start-factor', type=float, default=0.1, help='warmup 起始因子，相对 base lr')
	parser.add_argument('--plateau-patience', type=int, default=10, help='ReduceLROnPlateau 的耐心轮数')
	parser.add_argument('--plateau-factor', type=float, default=0.5, help='ReduceLROnPlateau 的衰减因子')
	parser.add_argument('--plateau-min-lr', type=float, default=1e-6, help='ReduceLROnPlateau 的最小学习率')
	# 提前停止参数
	parser.add_argument('--early-stopping', action='store_true', help='启用提前停止机制')
	parser.add_argument('--patience', type=int, default=20, help='提前停止耐心值（验证损失连续不改善的轮数）')
	parser.add_argument('--min-delta', type=float, default=1e-6, help='提前停止最小改善阈值')
	# 模型参数
	parser.add_argument('--dgcnn-k', type=int, default=20, help='DGCNN 邻域点数 k')
	parser.add_argument('--dgcnn-feat-dim', type=int, default=512, help='DGCNN 输出全局特征维度')
	parser.add_argument('--dgcnn-dropout', type=float, default=0.1, help='DGCNN 内部 Dropout 概率')
	parser.add_argument('--dgcnn-multi-scale-ks', type=str, default='10,20,30', help='多尺度 EdgeConv 的 k 列表，如: 10,20,30；留空禁用')
	parser.add_argument('--hidden-dims', type=str, default='256,256,128', help='回归器隐藏层维度（逗号分隔）')
	parser.add_argument('--mlp-dropout', type=float, default=0.2, help='回归器 MLP 的 Dropout 概率')
	# 损失/评估参数
	parser.add_argument('--cd-chunk', type=int, default=1024, help='Chamfer 距离计算分块大小')
	parser.add_argument('--global-cd-weight', type=float, default=1.0, help='全局 Chamfer Distance 的损失权重')
	parser.add_argument('--offset-l2-weight', type=float, default=0.0, help='位移向量 L2 正则项权重')
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
	# 设置标准化模式
	train_set.normalize_mode = args.normalize

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
	# 去重排序
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


# accuracy 指标已移除


def evaluate(encoder, regressor, loader, device, cd_chunk: int, local_cd_cfg: dict[str, float | int] | None = None):
	encoder.eval()
	regressor.eval()
	total_loss = 0.0
	total_cnt = 0
	with torch.no_grad():
		for batch in loader:
			foot = batch['foot'].to(device)  # (B,N,C)
			template = batch['template'].to(device)  # (B,N,3)
			target = batch['insole'].to(device)  # (B,N,3)

			# 编码足模 -> 全局特征
			foot_in = foot.transpose(2, 1).contiguous()  # (B,C,N)
			global_feat = encoder(foot_in)  # (B,d)
			pred = regressor(template, global_feat)  # (B,N,3)
			loss = chamfer_distance(pred, target, reduction='mean', chunk=cd_chunk)
			if local_cd_cfg and local_cd_cfg.get('weight', 0.0) > 0:
				lcd = local_chamfer_distance(
					pred, target,
					num_patches=int(local_cd_cfg.get('patches', 64)),
					radius=float(local_cd_cfg.get('radius', 0.2)),
					reduction='mean',
					chunk=cd_chunk,
				)
				loss = loss + float(local_cd_cfg['weight']) * lcd
			bs = foot.size(0)
			total_loss += loss.item() * bs
			total_cnt += bs
	avg_loss = total_loss / max(1, total_cnt)
	return avg_loss


def _resolve_templates_for_side(args):
	side = (args.side or 'LR').upper()
	if side != 'LR':
		return side, args.template
	# LR 模式：确定 L/R 两个模板
	tpl_L = args.template_L
	tpl_R = args.template_R
	if (tpl_L is None) or (tpl_R is None):
		# 尝试从 --template 推断 *_L/_R
		base = Path(args.template)
		if base.suffix.lower() == '.npz':
			stem = base.stem
			parent = base.parent
			tpl_L = tpl_L or str(parent / f"{stem}_L.npz")
			tpl_R = tpl_R or str(parent / f"{stem}_R.npz")
	return side, (tpl_L, tpl_R)


def train():
	args = parse_args()
	# 交互式参数确认（默认开启，可通过 --no-interactive 关闭）
	if not args.no_interactive:
		# 提取默认参数，用于展示对比，并逐项询问是否更改
		_defaults = parse_args([])

		def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
			hint = 'Y/n' if default_yes else 'y/N'
			while True:
				ans = input(f"{prompt} ({hint}): ").strip().lower()
				if not ans:
					return default_yes
				if ans in ('y', 'yes'):
					return True
				if ans in ('n', 'no'):
					return False
				print("请输入 y 或 n。")

		def _prompt_typed(prompt: str, caster, default_value):
			while True:
				raw = input(f"{prompt} [默认: {default_value}]: ").strip()
				if raw == '':
					return default_value
				try:
					return caster(raw)
				except Exception:
					print("输入无效，请重试。")

		def _maybe_change(label: str, cur, dft, caster):
			print(f"{label}: 当前={cur} | 默认={dft}")
			if _prompt_yes_no("是否更改此参数?", default_yes=False):
				return _prompt_typed(f"请输入新的 {label}", caster, cur)
			return cur

		print("=== 交互式参数设置（逐项询问是否更改）===")
		args.data_root = _maybe_change("数据根目录", args.data_root, _defaults.data_root, str)
		args.template = _maybe_change("模板路径", args.template, _defaults.template, str)
		args.device = _maybe_change("设备", args.device, _defaults.device, str)
		args.batch_size = _maybe_change("批大小", args.batch_size, _defaults.batch_size, int)
		args.epochs = _maybe_change("轮数", args.epochs, _defaults.epochs, int)
		args.lr = _maybe_change("学习率", args.lr, _defaults.lr, float)
		args.weight_decay = _maybe_change("权重衰减", args.weight_decay, _defaults.weight_decay, float)

		# scheduler 选择
		print(f"调度器: 当前={args.scheduler} | 可选=['cosine','step','plateau','none']")
		if _prompt_yes_no("是否更改调度器?", default_yes=False):
			def _cast_sched(s: str) -> str:
				s = s.strip().lower()
				if s not in ('cosine', 'step', 'plateau', 'none'):
					raise ValueError('非法调度器')
				return s
			args.scheduler = _prompt_typed("请输入调度器 (cosine|step|plateau|none)", _cast_sched, args.scheduler)
		if args.scheduler == 'step':
			args.step_size = _maybe_change("StepLR 步长", args.step_size, _defaults.step_size, int)
			args.gamma = _maybe_change("StepLR 衰减率", args.gamma, _defaults.gamma, float)
		if args.scheduler == 'plateau':
			args.plateau_patience = _maybe_change("Plateau 耐心", args.plateau_patience, _defaults.plateau_patience, int)
			args.plateau_factor = _maybe_change("Plateau 衰减因子", args.plateau_factor, _defaults.plateau_factor, float)
			args.plateau_min_lr = _maybe_change("Plateau 最小 lr", args.plateau_min_lr, _defaults.plateau_min_lr, float)

		# 数据集相关
		args.use_normals = _prompt_yes_no(
			f"使用法向? (当前: {'是' if args.use_normals else '否'})",
			default_yes=args.use_normals,
		)
		args.num_points = _maybe_change("每样本点数", args.num_points, _defaults.num_points, int)
		args.val_ratio = _maybe_change("验证集比例", args.val_ratio, _defaults.val_ratio, float)
		# 打乱：内部参数为 no_shuffle，交互使用正向语义
		shuffle_now = not args.no_shuffle
		shuffle_now = _prompt_yes_no(f"训练时打乱点顺序? (当前: {'是' if shuffle_now else '否'})", default_yes=shuffle_now)
		args.no_shuffle = (not shuffle_now)

		# 标准化方式
		print(f"标准化方式: 当前={args.normalize} | 可选=['center','sphere','cube']")
		if _prompt_yes_no("是否更改标准化方式?", default_yes=False):
			def _cast_norm(s: str) -> str:
				s = s.strip().lower()
				if s not in ('center', 'sphere', 'cube'):
					raise ValueError('非法标准化方式')
				return s
			args.normalize = _prompt_typed("请输入标准化方式 (center|sphere|cube)", _cast_norm, args.normalize)

		# 模型结构
		args.dgcnn_k = _maybe_change("DGCNN 邻域 k", args.dgcnn_k, _defaults.dgcnn_k, int)
		args.dgcnn_feat_dim = _maybe_change("DGCNN 特征维度", args.dgcnn_feat_dim, _defaults.dgcnn_feat_dim, int)
		args.dgcnn_dropout = _maybe_change("DGCNN Dropout 概率", args.dgcnn_dropout, _defaults.dgcnn_dropout, float)
		args.dgcnn_multi_scale_ks = _maybe_change("多尺度 k 列表(逗号分隔)", args.dgcnn_multi_scale_ks, _defaults.dgcnn_multi_scale_ks, str)
		args.hidden_dims = _maybe_change("回归器隐藏层(逗号分隔)", args.hidden_dims, _defaults.hidden_dims, str)

		# 损失/评估
		args.cd_chunk = _maybe_change("Chamfer 分块大小", args.cd_chunk, _defaults.cd_chunk, int)
		# 局部 CD 开关与参数
		enable_local_cd = _prompt_yes_no(
			f"启用局部CD? (当前: {'是' if args.local_cd_weight > 0 else '否'})",
			default_yes=(args.local_cd_weight > 0),
		)
		if enable_local_cd:
			args.local_cd_weight = _maybe_change("局部CD 权重", args.local_cd_weight, _defaults.local_cd_weight, float)
			args.local_cd_patches = _maybe_change("局部CD patch 数", args.local_cd_patches, _defaults.local_cd_patches, int)
			args.local_cd_radius = _maybe_change("局部CD 半径", args.local_cd_radius, _defaults.local_cd_radius, float)
		else:
			args.local_cd_weight = 0.0

		# 提前停止参数
		args.early_stopping = _prompt_yes_no(
			f"启用提前停止? (当前: {'是' if args.early_stopping else '否'})",
			default_yes=args.early_stopping,
		)
		if args.early_stopping:
			args.patience = _maybe_change("提前停止耐心值", args.patience, _defaults.patience, int)
			args.min_delta = _maybe_change("提前停止最小改善阈值", args.min_delta, _defaults.min_delta, float)

		# 其余
		args.num_workers = _maybe_change("数据加载进程数", args.num_workers, _defaults.num_workers, int)
		args.seed = _maybe_change("随机种子", args.seed, _defaults.seed, int)
		args.log_dir = _maybe_change("日志目录", args.log_dir, _defaults.log_dir, str)
		args.save_dir = _maybe_change("模型保存目录", args.save_dir, _defaults.save_dir, str)

		# 最终汇总与确认
		print("\n=== 配置预览 ===")
		print(f"数据根目录: {args.data_root}")
		print(f"模板路径: {args.template}")
		print(f"设备: {args.device}")
		print(f"批大小: {args.batch_size} | 轮数: {args.epochs}")
		print(f"学习率: {args.lr} | 权重衰减: {args.weight_decay}")
		print(f"调度器: {args.scheduler}")
		if args.scheduler == 'step':
			print(f"  StepLR: step_size={args.step_size}, gamma={args.gamma}")
		if args.scheduler == 'plateau':
			print(f"  Plateau: patience={args.plateau_patience}, factor={args.plateau_factor}, min_lr={args.plateau_min_lr}")
		print(f"Warmup: epochs={args.warmup_epochs}, start_factor={args.warmup_start_factor}")
		print(f"使用法向: {'是' if args.use_normals else '否'}")
		print(f"点数: {args.num_points} | 验证集比例: {args.val_ratio} | 打乱: {'是' if not args.no_shuffle else '否'}")
		print(f"标准化: {args.normalize}")
		print(f"DGCNN: k={args.dgcnn_k}, feat_dim={args.dgcnn_feat_dim}, dropout={args.dgcnn_dropout}, ms_ks={args.dgcnn_multi_scale_ks}")
		print(f"回归器隐藏层: {_parse_hidden_dims(args.hidden_dims)} | MLP Dropout={args.mlp_dropout}")
		print(f"CD 分块: {args.cd_chunk}")
		print(f"局部CD: weight={args.local_cd_weight} | patches={args.local_cd_patches} | radius={args.local_cd_radius}")
		print(f"损失权重: global_cd={args.global_cd_weight}, offset_l2={args.offset_l2_weight}")
		print(f"提前停止: {'是' if args.early_stopping else '否'} | 耐心值: {args.patience} | 最小改善阈值: {args.min_delta}")
		print(f"数据加载进程数: {args.num_workers} | 随机种子: {args.seed}")
		print(f"日志目录: {args.log_dir} | 模型保存目录: {args.save_dir}")

		if not _prompt_yes_no("确认开始训练?", default_yes=True):
			print("已取消。")
			return
	set_seed(args.seed)
	device = torch.device(args.device)
	# 仅保留最新日志：创建前清理旧日志
	log_dir_path = Path(args.log_dir)
	_clean_old_logs(log_dir_path, name_prefix='train_deformnet')
	logger = setup_logger(log_dir_path, name='train_deformnet')
	logger.info(f"使用设备: {device}")

	side, tpl = _resolve_templates_for_side(args)
	sides_to_run = ['L', 'R'] if side == 'LR' else [side]

	for side_run in sides_to_run:
		# 针对当前侧别解析模板与输出目录
		if isinstance(tpl, tuple):
			tpl_L, tpl_R = tpl
			template_path = tpl_L if side_run == 'L' else tpl_R
		else:
			template_path = args.template

		# 日志与保存子目录
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
		
		# 提前停止相关变量
		patience_counter = 0
		best_val_epoch = 0

		# 历史记录
		hist_train_loss: list[float] = []
		hist_val_loss: list[float] = []
		# 不再记录 accuracy

		for epoch in range(1, args.epochs + 1):
			encoder.train()
			regressor.train()
			# 学习率 warmup（按 epoch 线性上升）
			if args.warmup_epochs and epoch <= args.warmup_epochs:
				warmup_progress = 0.0 if args.warmup_epochs <= 1 else float(epoch - 1) / float(args.warmup_epochs - 1)
				factor = args.warmup_start_factor + (1.0 - args.warmup_start_factor) * warmup_progress
				for pg in optimizer.param_groups:
					pg['lr'] = args.lr * factor
			running = 0.0
			count = 0
			# epoch 进度条
			pbar = tqdm(total=len(train_loader), desc=f"[{side_run}] Epoch {epoch:03d}", leave=False) if _HAS_TQDM else None
			for it, batch in enumerate(train_loader, start=1):
				foot = batch['foot'].to(device)  # (B,N,C)
				template = batch['template'].to(device)  # (B,N,3)
				target = batch['insole'].to(device)  # (B,N,3)

				optimizer.zero_grad(set_to_none=True)

				# 足模编码 -> 全局特征
				foot_in = foot.transpose(2, 1).contiguous()  # (B,C,N)
				global_feat = encoder(foot_in)  # (B,512)

				# 点级回归 -> 预测鞋垫
				pred = regressor(template, global_feat)  # (B,N,3)

				# Chamfer Distance 监督（可加权）
				loss = args.global_cd_weight * chamfer_distance(pred, target, reduction='mean', chunk=args.cd_chunk)
				if args.local_cd_weight > 0:
					lcd = local_chamfer_distance(
						pred, target,
						num_patches=args.local_cd_patches,
						radius=args.local_cd_radius,
						reduction='mean',
						chunk=args.cd_chunk,
					)
					loss = loss + args.local_cd_weight * lcd
				# 位移 L2 正则
				if args.offset_l2_weight > 0:
					offsets = pred - template
					reg_l2 = torch.mean(torch.sum(offsets * offsets, dim=-1))
					loss = loss + args.offset_l2_weight * reg_l2
				loss.backward()
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
			val_loss = evaluate(encoder, regressor, val_loader, device, cd_chunk=args.cd_chunk, local_cd_cfg=local_cfg)
			logger.info(f"[{side_run}] [Epoch {epoch:03d}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

			# 记录
			hist_train_loss.append(train_loss)
			hist_val_loss.append(val_loss)
			

			# 调度器更新：warmup 阶段不更新主调度器
			if scheduler is not None and (not args.warmup_epochs or epoch > args.warmup_epochs):
				if isinstance(scheduler, ReduceLROnPlateau):
					scheduler.step(val_loss)
				else:
					scheduler.step()

			# 保存最佳模型和提前停止检查
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

			# 提前停止检查
			if args.early_stopping and patience_counter >= args.patience:
				logger.info(f"[{side_run}] 提前停止触发！最佳验证损失: {best_val:.6f} (第 {best_val_epoch} 轮)")
				break

		# 训练结束保存最终权重
		actual_epochs = epoch  # 实际训练的轮数
		final_ckpt = {
			'epoch': actual_epochs,
			'encoder': encoder.state_dict(),
			'regressor': regressor.state_dict(),
			'args': vars(args),
			'side': side_run,
		}
		torch.save(final_ckpt, sub_save / 'final.pth')
		
		if args.early_stopping and patience_counter >= args.patience:
			logger.info(f"[{side_run}] 训练提前停止！实际训练 {actual_epochs} 轮，最佳验证损失: {best_val:.6f} (第 {best_val_epoch} 轮)")
		else:
			logger.info(f"[{side_run}] 训练完成！共训练 {actual_epochs} 轮，最终模型保存在 {sub_save / 'final.pth'}")

		# 保存曲线图与指标 CSV 到该侧别日志目录
		try:
			log_dir_path = sub_log
			log_dir_path.mkdir(parents=True, exist_ok=True)

			# 保存 CSV
			csv_path = log_dir_path / 'metrics.csv'
			with open(csv_path, 'w', encoding='utf-8') as f:
				f.write('epoch,train_loss,val_loss\n')
				for i, (tl, vl) in enumerate(zip(hist_train_loss, hist_val_loss), start=1):
					f.write(f"{i},{tl:.8f},{vl:.8f}\n")
			logger.info(f"[{side_run}] 指标已保存到: {csv_path}")

			if _HAS_MPL:
				# Loss 曲线
				plt.figure(figsize=(8, 5))
				epochs_range = range(1, len(hist_train_loss) + 1)
				plt.plot(epochs_range, hist_train_loss, label='train_loss')
				plt.plot(epochs_range, hist_val_loss, label='val_loss')
				
				# 标记最佳验证损失点
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

				# 不再绘制 Accuracy 曲线
			else:
				logger.info(f"[{side_run}] matplotlib 不可用，已保存 CSV，可用外部工具绘图。")
		except Exception as e:
			logger.warning(f"[{side_run}] 保存曲线失败: {e}")


if __name__ == '__main__':
	raise SystemExit(train())