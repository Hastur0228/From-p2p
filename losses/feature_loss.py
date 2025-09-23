import torch
import torch.nn.functional as F


def feature_matching_loss(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    基于特征的匹配损失，让模型在感知空间更一致。

    通过比较预测点云和目标点云的特征表示来确保感知一致性。
    使用最近邻匹配的方式将预测特征与目标特征对应起来。

    参数：
    - pred_features: 预测点云的特征 (B, N, F)
    - target_features: 目标点云的特征 (B, M, F)
    - pred_points: 预测点云坐标 (B, N, 3)
    - target_points: 目标点云坐标 (B, M, 3)
    - reduction: 'mean' 或 'sum'

    返回：
    - 特征匹配损失 (tensor)
    """
    if pred_features.dim() != 3 or target_features.dim() != 3:
        raise ValueError("features 应为 3D 张量，形如 (B,N,F)/(B,M,F)")
    if pred_points.dim() != 3 or target_points.dim() != 3:
        raise ValueError("points 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B, N, F = pred_features.shape
    _, M, _ = target_features.shape
    device = pred_features.device

    losses = []

    for b in range(B):
        pred_feat_b = pred_features[b]  # (N, F)
        target_feat_b = target_features[b]  # (M, F)
        pred_points_b = pred_points[b]  # (N, 3)
        target_points_b = target_points[b]  # (M, 3)

        # 方案1: 基于几何距离的特征匹配
        # 计算 pred 到 target 的最近距离，找到几何上对应的目标特征
        dists = torch.cdist(pred_points_b, target_points_b, p=2)  # (N, M)
        _, nearest_indices = dists.min(dim=1)  # (N,)

        # 获取对应的目标特征
        corresponding_target_feat = target_feat_b[nearest_indices]  # (N, F)

        # 计算特征之间的L2距离
        feat_dist = torch.norm(pred_feat_b - corresponding_target_feat, dim=-1)  # (N,)

        losses.append(feat_dist.mean())

    losses = torch.stack(losses)  # (B,)

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")


def perceptual_loss(
    pred_features: torch.Tensor,
    target_features: torch.Tensor,
    layers: list[int] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    感知损失，比较多层特征表示。

    参数：
    - pred_features: 预测的多层特征列表 [layer1, layer2, ...] 每个 (B, N, F)
    - target_features: 目标的多层特征列表 [layer1, layer2, ...] 每个 (B, M, F)
    - layers: 要使用的层索引列表，默认使用所有层
    - reduction: 'mean' 或 'sum'

    返回：
    - 感知损失 (tensor)
    """
    if not isinstance(pred_features, list) or not isinstance(target_features, list):
        raise ValueError("pred_features 和 target_features 应为特征列表")

    if len(pred_features) != len(target_features):
        raise ValueError("pred_features 和 target_features 的层数应相同")

    if layers is None:
        layers = list(range(len(pred_features)))

    total_loss = 0.0
    num_layers = len(layers)

    for layer_idx in layers:
        if layer_idx >= len(pred_features):
            continue

        pred_feat = pred_features[layer_idx]
        target_feat = target_features[layer_idx]

        # 使用特征匹配损失
        layer_loss = feature_matching_loss(
            pred_feat.unsqueeze(0) if pred_feat.dim() == 2 else pred_feat,
            target_feat.unsqueeze(0) if target_feat.dim() == 2 else target_feat,
            torch.zeros_like(pred_feat[..., :3]),  # 占位符，不使用几何匹配
            torch.zeros_like(target_feat[..., :3]),  # 占位符，不使用几何匹配
            reduction=reduction
        )
        total_loss = total_loss + layer_loss

    return total_loss / max(1, num_layers)


def chamfer_feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    encoder,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Chamfer特征损失：结合几何和特征的匹配。

    先对点云提取特征，然后基于特征相似度计算Chamfer距离。

    参数：
    - pred: 预测点云 (B, N, 3)
    - target: 目标点云 (B, M, 3)
    - encoder: 特征提取器
    - reduction: 'mean' 或 'sum'

    返回：
    - Chamfer特征损失 (tensor)
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred/target 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    # 提取特征
    with torch.no_grad():
        pred_feat = encoder(pred.transpose(2, 1).contiguous())  # (B, N, F)
        target_feat = encoder(target.transpose(2, 1).contiguous())  # (B, M, F)

    B, N, F = pred_feat.shape
    _, M, _ = target_feat.shape

    # 计算特征相似度矩阵
    pred_feat_norm = F.normalize(pred_feat, dim=-1)  # (B, N, F)
    target_feat_norm = F.normalize(target_feat, dim=-1)  # (B, M, F)

    # 特征相似度 (余弦相似度)
    sim_matrix = torch.bmm(pred_feat_norm, target_feat_norm.transpose(1, 2))  # (B, N, M)

    # 转换为距离 (1 - similarity)
    dist_matrix = 1 - sim_matrix  # (B, N, M)

    # 计算Chamfer距离
    # pred到target的距离
    min_dist_pred_to_target = dist_matrix.min(dim=2)[0]  # (B, N)
    # target到pred的距离
    min_dist_target_to_pred = dist_matrix.min(dim=1)[0]  # (B, M)

    if reduction == 'mean':
        loss = min_dist_pred_to_target.mean() + min_dist_target_to_pred.mean()
    elif reduction == 'sum':
        loss = min_dist_pred_to_target.sum() + min_dist_target_to_pred.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")

    return loss
