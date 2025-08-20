import torch


def emd_loss(
    pcd1: torch.Tensor,
    pcd2: torch.Tensor,
    epsilon: float = 0.02,
    num_iters: int = 50,
    reduction: str = 'mean',
    subset_size: int | None = 1024,
    normalize_loss: bool = False,
    scale_eps: float = 1e-6,
) -> torch.Tensor:
    """
    近似 Earth Mover's Distance (EMD) via Sinkhorn (entropic OT) for batched point clouds.
    - pcd1, pcd2: (B, N, 3), (B, M, 3)
    - epsilon: Entropic regularization strength
    - num_iters: Sinkhorn iterations
    - subset_size: Optional subsampling size to bound memory/time. If None, use all points (may be heavy)
    - normalize_loss: If True, divide distances by geometric scale (bbox diagonal) per sample
    - reduction: 'mean' or 'sum'
    返回: 标量损失 (tensor)
    """
    if pcd1.dim() != 3 or pcd2.dim() != 3:
        raise ValueError("pcd1/pcd2 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B, N, _ = pcd1.shape
    _, M, _ = pcd2.shape
    device = pcd1.device

    # 统一子采样点数
    if subset_size is not None and subset_size > 0:
        K = min(subset_size, N, M)
        if K <= 0:
            return torch.zeros((), device=device, dtype=pcd1.dtype)
        idx1 = torch.randperm(N, device=device)[:K]
        idx2 = torch.randperm(M, device=device)[:K]
        x = pcd1[:, idx1, :]
        y = pcd2[:, idx2, :]
    else:
        K = min(N, M)
        # 若 N != M，则简单截断到 K
        x = pcd1[:, :K, :]
        y = pcd2[:, :K, :]

    # 几何尺度归一化（仅对损失数值缩放，不改变坐标）
    scale = None
    if normalize_loss:
        with torch.no_grad():
            bb1 = (pcd1.max(dim=1).values - pcd1.min(dim=1).values)  # (B,3)
            bb2 = (pcd2.max(dim=1).values - pcd2.min(dim=1).values)  # (B,3)
            diag1 = torch.linalg.norm(bb1, dim=-1)  # (B,)
            diag2 = torch.linalg.norm(bb2, dim=-1)  # (B,)
            scale = 0.5 * (diag1 + diag2)
            scale = torch.clamp(scale, min=scale_eps)  # (B,)

    # 代价矩阵（欧氏距离）与核矩阵
    # C: (B, K, K)
    C = torch.cdist(x, y, p=2)  # 不平方，保留可微分
    if normalize_loss:
        C = C / scale.view(-1, 1, 1)
    Kmat = torch.exp(-C / max(epsilon, 1e-8))  # (B, K, K)

    # 统一质量分布（均匀）
    a = torch.full((B, K), 1.0 / K, device=device, dtype=pcd1.dtype)
    b = torch.full((B, K), 1.0 / K, device=device, dtype=pcd1.dtype)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn 迭代
    for _ in range(int(num_iters)):
        Kv = torch.bmm(Kmat, v.unsqueeze(-1)).squeeze(-1) + 1e-12
        u = a / Kv
        KTu = torch.bmm(Kmat.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-12
        v = b / KTu

    # 运输计划
    T = u.unsqueeze(-1) * Kmat * v.unsqueeze(-2)  # (B, K, K)
    cost = torch.sum(T * C, dim=(1, 2))  # (B,)

    if reduction == 'mean':
        return cost.mean()
    elif reduction == 'sum':
        return cost.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")

