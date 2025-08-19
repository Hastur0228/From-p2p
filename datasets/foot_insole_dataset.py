from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.pc_normalize import normalize_pointcloud


def _load_npz_points(path: Path, use_normals: bool) -> np.ndarray:
    """
    加载 .npz/.npy 文件，返回 (N, C) 数组。
    - 若为 .npz，优先读取键 'points' 和可选 'normals'。
    - 若 use_normals=True 且存在法向，则拼接为 (N,6)，否则为 (N,3)。
    """
    suffix = path.suffix.lower()
    if suffix == '.npz':
        with np.load(str(path)) as data:
            if 'points' in data.files:
                points = data['points']
            else:
                # 兼容极端情况：取第一个数组
                key0 = data.files[0]
                points = data[key0]
            normals = data['normals'] if ('normals' in data.files and use_normals) else None
    elif suffix == '.npy':
        points = np.load(str(path))
        normals = None
    else:
        raise ValueError(f"不支持的文件类型: {path}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"无效点云形状: {path} -> {points.shape}")
    points = points[:, :3].astype(np.float32)
    if use_normals and normals is not None and normals.ndim == 2 and normals.shape[1] >= 3:
        normals = normals[:, :3].astype(np.float32)
        arr = np.concatenate([points, normals], axis=1)
    else:
        arr = points
    return arr


def _pair_files(feet_dir: Path, insoles_dir: Path) -> List[Tuple[Path, Path]]:
    """
    依据文件名中的编号与侧别（L/R）进行配对。
    期望文件名示例：001_foot_L.npz 对应 001_insole_L.npz
    返回已配对的 (foot_path, insole_path) 列表。
    """
    feet = {}
    for n in os.listdir(feet_dir):
        nl = n.lower()
        if nl.endswith('.npz') or nl.endswith('.npy'):
            feet[nl] = feet_dir / n

    pairs: List[Tuple[Path, Path]] = []
    for n in os.listdir(insoles_dir):
        nl = n.lower()
        if not (nl.endswith('.npz') or nl.endswith('.npy')):
            continue
        # 尝试将 insole 文件名映射到 foot 文件名
        # 规则：将 "insole" 替换为 "foot"
        cand = nl.replace('insole', 'foot')
        foot_path = feet.get(cand)
        if foot_path is not None:
            pairs.append((foot_path, insoles_dir / n))
    pairs.sort(key=lambda p: str(p[0]))
    return pairs


class FootInsoleDataset(Dataset):
    """
    足模点云 -> 目标鞋垫点云 的监督数据集。
    - 输入足模点云: (N, 3/6)
    - 输出鞋垫点云: (N, 3)
    模板点云由外部提供（通常固定为平均鞋垫 4096 点）。
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        val_ratio: float = 0.1,
        use_normals: bool = False,
        num_points: int = 4096,
        template_path: str | Path | None = None,
        random_shuffle_points: bool = True,
        side: str | None = None,
        normalize_mode: str = 'center',
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.use_normals = use_normals
        self.num_points = num_points
        self.random_shuffle_points = random_shuffle_points
        # 规范化模式：'sphere'（单位球）、'cube'（单位立方）、'center'（仅中心化）。
        self.normalize_mode = str(normalize_mode)
        if side is not None:
            side = side.upper()
            if side not in ("L", "R"):
                raise ValueError("side 只能为 'L' 或 'R' 或 None")
        self.side = side
        self.split = split
        # 增强配置（由外部在构造后设置），默认关闭
        self.augment_enable: bool = False
        self.augment_multiplier: int = 1
        self.aug_jitter_sigma_range: tuple[float, float] = (0.0, 0.0)
        self.aug_dropout_patches_range: tuple[int, int] = (0, 0)
        self.aug_dropout_radius_range: tuple[float, float] = (0.05, 0.15)
        self.aug_normal_shift_range: tuple[float, float] = (0.0, 0.0)
        self.aug_resample_mode: str = 'none'  # ['none','uniform','poisson']
        self.aug_uniform_keep_range: tuple[float, float] = (0.6, 1.0)
        self.aug_poisson_voxel_range: tuple[float, float] = (0.01, 0.04)

        feet_dir = self.data_root / 'feet'
        insoles_dir = self.data_root / 'insoles'
        if not feet_dir.exists() or not insoles_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {feet_dir} | {insoles_dir}")

        pairs = _pair_files(feet_dir, insoles_dir)
        # 可选：按左右脚进行过滤
        if self.side is not None:
            suffix_L = '_l.'
            suffix_R = '_r.'
            filtered = []
            for fp, ip in pairs:
                name = ip.name.lower()
                if self.side == 'L' and suffix_L in name:
                    filtered.append((fp, ip))
                elif self.side == 'R' and suffix_R in name:
                    filtered.append((fp, ip))
            pairs = filtered
        if len(pairs) == 0:
            raise RuntimeError("未找到任何可配对的 足模/鞋垫 点云文件")

        # 简单切分 train/val（按顺序 9:1）
        num_total = len(pairs)
        num_val = max(1, int(num_total * val_ratio))
        if split == 'train':
            self.pairs = pairs[:-num_val] if num_total > num_val else pairs
        elif split == 'val':
            self.pairs = pairs[-num_val:]
        else:
            raise ValueError("split 只能为 'train' 或 'val'")

        # 读取模板点云（仅 xyz）
        self.template: np.ndarray | None = None
        if template_path is not None:
            tpath = Path(template_path)
            with np.load(str(tpath)) as data:
                tpl = data['points'] if 'points' in data.files else data[data.files[0]]
            if tpl.ndim != 2 or tpl.shape[1] < 3:
                raise ValueError(f"模板点云形状无效: {tpath} -> {tpl.shape}")
            self.template = tpl[:, :3].astype(np.float32)

        # 预先归一化并缓存每个样本，避免在 __getitem__ 中重复计算
        # 每个样本缓存: dict(foot, insole, template?, centroid, scale)
        # 注意：训练集支持在 __getitem__ 中按需做数据增强，不在此处膨胀缓存尺寸
        self._cache: list[dict] = []
        eps = 1e-9
        for fp, ip in self.pairs:
            foot_raw = _load_npz_points(fp, use_normals=self.use_normals)
            with np.load(str(ip)) as data:
                if 'points' in data.files:
                    insole_raw = data['points']
                else:
                    insole_raw = data[data.files[0]]
            insole_raw = insole_raw[:, :3].astype(np.float32)

            # 以足模为基准做标准化（center/sphere/cube），得到统一坐标
            foot_norm, centroid, scale = normalize_pointcloud(foot_raw, center_only=False, mode=self.normalize_mode)
            insole_norm = insole_raw.copy()
            insole_norm[:, :3] = (insole_norm[:, :3] - centroid) / (scale + eps)

            item = {
                'foot': foot_norm.astype(np.float32),
                'insole': insole_norm.astype(np.float32),
                'centroid': centroid.astype(np.float32).reshape(1, 3),
                'scale': np.float32(scale),
            }
            if self.template is not None:
                tpl = self.template
                tpl_norm = tpl.copy()
                tpl_norm[:, :3] = (tpl_norm[:, :3] - centroid) / (scale + eps)
                item['template'] = tpl_norm.astype(np.float32)
            self._cache.append(item)

    def __len__(self) -> int:
        base = len(self.pairs)
        if self.split == 'train' and self.augment_enable and self.augment_multiplier > 0:
            return base * (1 + int(self.augment_multiplier))
        return base

    def _maybe_subsample(self, arr: np.ndarray) -> np.ndarray:
        # 保证点数为 num_points，如不足则重复采样，如超出则随机下采样
        N = arr.shape[0]
        if N == self.num_points:
            return arr
        if N > self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            # 重复采样以补齐
            repeat = self.num_points // N + 1
            idx = np.arange(N).repeat(repeat)[: self.num_points]
        return arr[idx]

    # -------------------- 数据增强工具 --------------------
    def _random_in_range(self, a: float, b: float) -> float:
        lo = float(min(a, b))
        hi = float(max(a, b))
        return np.random.uniform(lo, hi)

    def _random_int_in_range(self, a: int, b: int) -> int:
        lo = int(min(a, b))
        hi = int(max(a, b))
        return int(np.random.randint(lo, hi + 1))

    def _apply_point_jitter(self, xyz: np.ndarray) -> None:
        sigma = self._random_in_range(*self.aug_jitter_sigma_range)
        if sigma <= 0:
            return
        noise = np.random.normal(loc=0.0, scale=sigma, size=xyz.shape).astype(np.float32)
        xyz += noise

    def _apply_local_dropout(self, xyz: np.ndarray) -> np.ndarray:
        num_patches = self._random_int_in_range(*self.aug_dropout_patches_range)
        if num_patches <= 0:
            return np.arange(xyz.shape[0], dtype=np.int64)
        N = xyz.shape[0]
        keep_mask = np.ones(N, dtype=bool)
        for _ in range(num_patches):
            center_idx = np.random.randint(0, N)
            center = xyz[center_idx]
            radius = self._random_in_range(*self.aug_dropout_radius_range)
            if radius <= 0:
                continue
            d2 = np.sum((xyz - center) ** 2, axis=1)
            keep_mask &= (d2 > (radius * radius))
        # 若全部被删除，退化为保留全部
        if not np.any(keep_mask):
            keep_mask[:] = True
        return np.nonzero(keep_mask)[0].astype(np.int64)

    def _apply_normal_shift(self, arr: np.ndarray) -> None:
        # arr: (N, 6) [xyz|normal]
        if arr.shape[1] < 6:
            return
        mag = self._random_in_range(*self.aug_normal_shift_range)
        if mag <= 0:
            return
        normals = arr[:, 3:6]
        arr[:, 0:3] += normals * mag

    def _apply_resample(self, xyz: np.ndarray) -> np.ndarray:
        mode = (self.aug_resample_mode or 'none').lower()
        if mode == 'none':
            return np.arange(xyz.shape[0], dtype=np.int64)
        N = xyz.shape[0]
        if mode == 'uniform':
            keep_ratio = self._random_in_range(*self.aug_uniform_keep_range)
            keep_ratio = float(np.clip(keep_ratio, 0.1, 1.0))
            M = max(1, int(round(N * keep_ratio)))
            idx = np.random.choice(N, M, replace=False)
            return np.sort(idx.astype(np.int64))
        if mode == 'poisson':
            # 以体素网格近似泊松盘采样（快速）
            voxel = self._random_in_range(*self.aug_poisson_voxel_range)
            voxel = max(1e-6, float(voxel))
            mins = xyz.min(axis=0)
            keys = np.floor((xyz - mins) / voxel).astype(np.int64)
            # 哈希到字典，保留每个体素一个代表点
            h = {}
            for i, k in enumerate(map(tuple, keys)):
                if k not in h:
                    h[k] = i
            sel = np.array(list(h.values()), dtype=np.int64)
            return np.sort(sel)
        return np.arange(xyz.shape[0], dtype=np.int64)

    def _apply_augmentations(self, foot: np.ndarray, insole: np.ndarray, template: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        # 深拷贝以免污染缓存
        foot_aug = foot.copy()
        insole_aug = insole.copy()
        tpl_aug = None if template is None else template.copy()

        # 1) 沿法向小偏移（仅 foot，若包含法向）
        self._apply_normal_shift(foot_aug)

        # 2) 点抖动（对 xyz 应用）
        self._apply_point_jitter(foot_aug[:, 0:3])
        self._apply_point_jitter(insole_aug[:, 0:3])
        if tpl_aug is not None:
            self._apply_point_jitter(tpl_aug[:, 0:3])

        # 3) 局部 dropout（对 foot / insole / 模板独立执行，以引入多样性）
        foot_keep = self._apply_local_dropout(foot_aug[:, 0:3])
        insole_keep = self._apply_local_dropout(insole_aug[:, 0:3])
        tpl_keep = None if tpl_aug is None else self._apply_local_dropout(tpl_aug[:, 0:3])

        foot_aug = foot_aug[foot_keep]
        insole_aug = insole_aug[insole_keep]
        if tpl_aug is not None and tpl_keep is not None:
            tpl_aug = tpl_aug[tpl_keep]

        # 4) 重采样以改变点密度（近似 Poisson / Uniform）
        foot_idx = self._apply_resample(foot_aug[:, 0:3])
        insole_idx = self._apply_resample(insole_aug[:, 0:3])
        foot_aug = foot_aug[foot_idx]
        insole_aug = insole_aug[insole_idx]
        if tpl_aug is not None:
            tpl_idx = self._apply_resample(tpl_aug[:, 0:3])
            tpl_aug = tpl_aug[tpl_idx]

        return foot_aug, insole_aug, tpl_aug

    def __getitem__(self, idx: int):
        base_len = len(self.pairs)
        if self.split == 'train' and self.augment_enable and self.augment_multiplier > 0:
            base_idx = idx % base_len
            do_augment = (idx >= base_len)
        else:
            base_idx = idx
            do_augment = False
        cache = self._cache[base_idx]
        foot = cache['foot']
        insole = cache['insole']
        # 可选：随机打乱点顺序（不修改缓存，先复制）
        if self.random_shuffle_points:
            foot = foot.copy()
            insole = insole.copy()
            np.random.shuffle(foot)
            np.random.shuffle(insole)

        tpl = cache.get('template', None)
        if tpl is not None and self.random_shuffle_points:
            tpl = tpl.copy()

        # 训练集增强：仅在扩展样本（idx>=base_len）时应用，原样本保留未增强版本
        if do_augment:
            foot, insole, tpl = self._apply_augmentations(foot, insole, tpl)

        # 下采样/补齐至固定点数
        foot = self._maybe_subsample(foot)
        insole = self._maybe_subsample(insole)
        if tpl is not None:
            tpl = self._maybe_subsample(tpl)

        item = {
            'foot': torch.from_numpy(foot),  # (N,C)
            'insole': torch.from_numpy(insole),  # (N,3)
            'centroid': torch.from_numpy(cache['centroid']).view(1, 3),  # (1,3)
            'scale': torch.tensor(float(cache['scale']), dtype=torch.float32),  # 标量
        }
        if tpl is not None:
            item['template'] = torch.from_numpy(tpl)  # (N,3)
        return item


