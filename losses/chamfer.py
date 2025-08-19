import torch


def _min_dists_chunked(A: torch.Tensor, B: torch.Tensor, chunk: int = 1024, squared: bool = True) -> torch.Tensor:
    """
    计算 A 中每个点到 B 的最近距离（欧氏），返回 (BATCH, N)；按 N 维分块以减小内存。
    A: (BATCH, N, 3), B: (BATCH, M, 3)
    """
    BATCH, N, _ = A.shape
    device = A.device
    mins = []
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        a = A[:, start:end, :]  # (BATCH, C, 3)
        # cdist: (BATCH, C, M)
        d = torch.cdist(a, B, p=2)
        if squared:
            d = d.pow(2)
        m, _ = d.min(dim=2)  # (BATCH, C)
        mins.append(m)
    return torch.cat(mins, dim=1)


def chamfer_distance(
    pcd1: torch.Tensor,
    pcd2: torch.Tensor,
    reduction: str = 'mean',
    chunk: int = 1024,
    squared: bool = True,
) -> torch.Tensor:
    """
    计算 Chamfer Distance (CD) between two point clouds，带分块避免内存占用过大。
    pcd1, pcd2: (B, N, 3) 与 (B, M, 3)
    reduction: 'mean' or 'sum'
    chunk: 分块大小（按 N/M 维）。4096 点建议 512~1024 视显存而定。
    """
    if pcd1.dim() != 3 or pcd2.dim() != 3:
        raise ValueError("pcd1/pcd2 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    # min over pcd2 for each point in pcd1
    min1 = _min_dists_chunked(pcd1, pcd2, chunk=chunk, squared=squared)  # (B, N)
    # min over pcd1 for each point in pcd2
    min2 = _min_dists_chunked(pcd2, pcd1, chunk=chunk, squared=squared)  # (B, M)

    if reduction == 'mean':
        loss = min1.mean() + min2.mean()
    elif reduction == 'sum':
        loss = min1.sum() + min2.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")
    return loss


def local_chamfer_distance(
    pcd1: torch.Tensor,
    pcd2: torch.Tensor,
    num_patches: int = 64,
    radius: float = 0.2,
    reduction: str = 'mean',
    chunk: int = 1024,
    squared: bool = True,
) -> torch.Tensor:
    """
    局部 Chamfer Distance：围绕 pcd1 的局部 patch（球邻域）计算 CD，然后在所有 patch 上取均值。
    - pcd1, pcd2: (B, N, 3), (B, M, 3)
    - num_patches: 每个 batch 样本抽取的 patch 数量（从 pcd1 中随机采样作为 patch 中心）
    - radius: 邻域半径（单位与数据坐标一致）
    - reduction: 目前仅支持 'mean'（返回所有样本与 patch 的均值）
    - chunk: 传递给分块 cdist 的参数，避免内存过大

    注意：若某个 patch 在任一云中无点，则对非空一侧与另一整云计算单向最近距离作为退化补偿；
          若两侧都为空则跳过该 patch。
    """
    if pcd1.dim() != 3 or pcd2.dim() != 3:
        raise ValueError("pcd1/pcd2 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B, N, _ = pcd1.shape
    _, M, _ = pcd2.shape
    device = pcd1.device

    total = 0.0
    count = 0

    # 对每个 batch 独立处理
    for b in range(B):
        x = pcd1[b]  # (N,3)
        y = pcd2[b]  # (M,3)
        if N == 0 or M == 0:
            continue

        # 随机选择 patch 中心（从 x 中采样）
        if num_patches >= N:
            centers_idx = torch.arange(N, device=device)
        else:
            centers_idx = torch.randperm(N, device=device)[:num_patches]
        centers = x[centers_idx]  # (P,3)

        # 计算到中心的距离，得到掩码
        # dist_x: (P,N), dist_y: (P,M)
        dist_x = torch.cdist(centers.unsqueeze(0), x.unsqueeze(0), p=2).squeeze(0)
        dist_y = torch.cdist(centers.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0)

        for p in range(centers.shape[0]):
            mask_x = dist_x[p] <= radius
            mask_y = dist_y[p] <= radius

            if mask_x.any() and mask_y.any():
                x_patch = x[mask_x].unsqueeze(0)
                y_patch = y[mask_y].unsqueeze(0)
                # 局部 CD（双向）
                loss_patch = chamfer_distance(x_patch, y_patch, reduction='mean', chunk=chunk, squared=squared)
                total = total + loss_patch
                count += 1
            elif mask_x.any() and (not mask_y.any()):
                # y 在该邻域为空：对 x_patch 到 y 全局做单向最近距离
                x_patch = x[mask_x].unsqueeze(0)
                min1 = _min_dists_chunked(x_patch, y.unsqueeze(0), chunk=chunk, squared=squared).mean()
                total = total + min1
                count += 1
            elif mask_y.any() and (not mask_x.any()):
                # x 在该邻域为空：对 y_patch 到 x 全局做单向最近距离
                y_patch = y[mask_y].unsqueeze(0)
                min2 = _min_dists_chunked(y_patch, x.unsqueeze(0), chunk=chunk, squared=squared).mean()
                total = total + min2
                count += 1
            else:
                # 两侧都为空：跳过
                continue

    if count == 0:
        # 退化：返回 0，避免除零
        return torch.zeros((), device=device, dtype=pcd1.dtype)

    if reduction == 'mean':
        return total / count
    elif reduction == 'sum':
        return total  # 理论上不常用
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")

