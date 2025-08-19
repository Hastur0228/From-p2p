import numpy as np
import torch


def normalize_pointcloud(points: np.ndarray, center_only: bool = False, mode: str | None = None):
    """
    标准化点云（以 xyz 的质心为中心），支持三种模式：
    - center: 仅中心化（减质心），不缩放
    - sphere: 等比缩放到单位球（除以最大范数）
    - cube: 等比缩放到单位立方（除以最大绝对坐标值）
    输入: points (N, C) -> C>=3，允许包含法向(追加在后)
    返回: normed_points (N, C), centroid (1,3), scale (标量 float)
    说明:
    - 仅对前 3 维的 xyz 进行中心化/缩放；法向等附加属性保持不变。
    - scale 为标量；反标准化时先乘以 scale，再加回 centroid。
    兼容旧参数 center_only：当 mode 未指定(None)时，center_only=False 等价于 sphere，True 等价于 center。
    """
    assert points.ndim == 2 and points.shape[1] >= 3, f"invalid shape: {points.shape}"
    xyz = points[:, :3].astype(np.float32)
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz_centered = xyz - centroid

    eps = 1e-9
    # 兼容旧参数
    if mode is None:
        mode = 'center' if center_only else 'sphere'

    mode = str(mode).lower()
    if mode == 'center':
        scale = 1.0
        xyz_norm = xyz_centered
    elif mode == 'sphere':
        # 单位球：除以最大范数，保持比例
        scale = float(np.max(np.linalg.norm(xyz_centered, axis=1)) + eps)
        xyz_norm = xyz_centered / scale
    elif mode == 'cube':
        # 单位立方：除以最大绝对坐标值，使得坐标范围落在 [-1, 1]
        scale = float(np.max(np.abs(xyz_centered)) + eps)
        xyz_norm = xyz_centered / scale
    else:
        raise ValueError(f"Unknown normalize mode: {mode}. Expected one of ['center','sphere','cube'].")

    normed_points = points.copy()
    normed_points[:, :3] = xyz_norm
    return normed_points.astype(np.float32), centroid.astype(np.float32), float(scale)


def denormalize_pointcloud(points: np.ndarray | torch.Tensor,
                           centroid: np.ndarray | torch.Tensor,
                           scale: float | np.ndarray | torch.Tensor,
                           center_only: bool = False):
    """
    反标准化点云，仅作用于 xyz 三维。
    - points: (N, C)
    - centroid: (1,3) 或 (3,)
    - scale: 标量
    - 若 center_only=True，则不进行缩放，仅加回质心。
    支持 numpy 与 torch 张量。
    """
    is_torch = isinstance(points, torch.Tensor)
    if is_torch:
        xyz = points[..., :3]
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=xyz.dtype, device=xyz.device)
        if not isinstance(centroid, torch.Tensor):
            centroid = torch.as_tensor(centroid, dtype=xyz.dtype, device=xyz.device)
        centroid = centroid.view(1, 3)
        if not center_only:
            xyz = xyz * scale
        xyz = xyz + centroid
        out = points.clone()
        out[..., :3] = xyz
        return out
    else:
        xyz = points[:, :3]
        centroid_np = np.asarray(centroid, dtype=xyz.dtype).reshape(1, 3)
        scale_val = float(scale)
        if not center_only:
            xyz = xyz * scale_val
        xyz = xyz + centroid_np
        out = points.copy()
        out[:, :3] = xyz
        return out


