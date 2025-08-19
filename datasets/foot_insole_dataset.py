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
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.use_normals = use_normals
        self.num_points = num_points
        self.random_shuffle_points = random_shuffle_points
        # 规范化模式：'sphere'（单位球）、'cube'（单位立方）、'center'（仅中心化）。默认使用 'center'，避免单位化缩放
        self.normalize_mode = 'center'
        if side is not None:
            side = side.upper()
            if side not in ("L", "R"):
                raise ValueError("side 只能为 'L' 或 'R' 或 None")
        self.side = side

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
        return len(self.pairs)

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

    def __getitem__(self, idx: int):
        cache = self._cache[idx]
        foot = cache['foot']
        insole = cache['insole']
        # 可选：随机打乱点顺序（不修改缓存，先复制）
        if self.random_shuffle_points:
            foot = foot.copy()
            insole = insole.copy()
            np.random.shuffle(foot)
            np.random.shuffle(insole)

        # 下采样/补齐至固定点数
        foot = self._maybe_subsample(foot)
        insole = self._maybe_subsample(insole)

        item = {
            'foot': torch.from_numpy(foot),  # (N,C)
            'insole': torch.from_numpy(insole),  # (N,3)
            'centroid': torch.from_numpy(cache['centroid']).view(1, 3),  # (1,3)
            'scale': torch.tensor(float(cache['scale']), dtype=torch.float32),  # 标量
        }
        if 'template' in cache and cache['template'] is not None:
            tpl = cache['template']
            if self.random_shuffle_points:
                tpl = tpl.copy()
            tpl = self._maybe_subsample(tpl)
            item['template'] = torch.from_numpy(tpl)  # (N,3)
        return item


