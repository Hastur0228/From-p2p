import torch
import torch.nn.functional as F


def normal_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_normals: torch.Tensor = None,
    target_normals: torch.Tensor = None,
    k: int = 20,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    计算法向一致性损失，保证曲面平滑性。

    通过比较预测点云和目标点云的局部法向来确保曲面的一致性。
    如果没有提供法向，则基于邻域点估算。

    参数：
    - pred: 预测点云 (B, N, 3)
    - target: 目标点云 (B, M, 3)
    - pred_normals: 预测点云的法向 (B, N, 3)，可选
    - target_normals: 目标点云的法向 (B, M, 3)，可选
    - k: KNN 邻域点数，用于估算法向
    - reduction: 'mean' 或 'sum'

    返回：
    - 法向一致性损失 (tensor)
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred/target 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B, N, _ = pred.shape
    _, M, _ = target.shape
    device = pred.device

    # 如果没有提供法向，则基于邻域估算
    if pred_normals is None:
        pred_normals = _estimate_normals(pred, k=k)  # (B, N, 3)

    if target_normals is None:
        target_normals = _estimate_normals(target, k=k)  # (B, M, 3)

    # 对于每个预测点，找到最近的目标点及其法向
    losses = []
    for b in range(B):
        pred_b = pred[b]  # (N, 3)
        target_b = target[b]  # (M, 3)
        pred_normals_b = pred_normals[b]  # (N, 3)
        target_normals_b = target_normals[b]  # (M, 3)

        # 计算 pred 到 target 的最近距离和对应索引
        dists = torch.cdist(pred_b, target_b, p=2)  # (N, M)
        _, nearest_indices = dists.min(dim=1)  # (N,)

        # 获取对应的目标法向
        corresponding_target_normals = target_normals_b[nearest_indices]  # (N, 3)

        # 计算法向夹角的余弦相似度
        cos_sim = F.cosine_similarity(
            pred_normals_b, corresponding_target_normals, dim=-1
        )  # (N,)

        # 将相似度转换为损失 (1 - cos_sim)，范围 [0, 2]
        # 完美对齐时 cos_sim = 1，损失 = 0
        # 完全相反时 cos_sim = -1，损失 = 2
        loss_b = 1 - cos_sim  # (N,)

        losses.append(loss_b.mean())  # 每个batch的平均损失

    losses = torch.stack(losses)  # (B,)

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")


def _estimate_normals(points: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    基于KNN邻域估算点云法向。

    参数：
    - points: 点云 (B, N, 3)
    - k: KNN 邻域点数

    返回：
    - 估算的法向 (B, N, 3)
    """
    B, N, _ = points.shape
    device = points.device

    normals = []
    for b in range(B):
        points_b = points[b]  # (N, 3)

        # 计算距离矩阵
        dists = torch.cdist(points_b, points_b, p=2)  # (N, N)

        # 找到每个点的K个最近邻（排除自身）
        _, indices = dists.topk(k + 1, dim=1, largest=False)  # (N, k+1)
        indices = indices[:, 1:]  # 排除自身，(N, k)

        # 获取邻域点
        neighbors = points_b[indices]  # (N, k, 3)

        # 计算每个点的质心
        centroids = neighbors.mean(dim=1)  # (N, 3)

        # 计算协方差矩阵并估算法向
        normals_b = []
        for i in range(N):
            # 邻域点去中心化
            centered = neighbors[i] - centroids[i]  # (k, 3)

            # 计算协方差矩阵
            cov = torch.matmul(centered.T, centered) / k  # (3, 3)

            # 计算特征值和特征向量
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            # 最小的特征值对应的特征向量作为法向
            normal = eigenvectors[:, 0]  # (3,)

            # 确保法向朝向一致（简单的启发式方法）
            # 这里使用质心到点的向量与法向的点积来判断方向
            view_dir = points_b[i] - centroids[i]  # (3,)
            if torch.dot(normal, view_dir) < 0:
                normal = -normal

            normals_b.append(normal)

        normals_b = torch.stack(normals_b)  # (N, 3)
        normals.append(normals_b)

    return torch.stack(normals)  # (B, N, 3)
