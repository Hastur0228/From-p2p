import torch
import numpy as np


def fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.01,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    计算 F-score (F-measure)，评价点云重建的覆盖率和精度。

    F-score = 2 * (precision * recall) / (precision + recall)

    其中：
    - precision = TP / (TP + FP) = 正确预测的点 / 所有预测的点
    - recall = TP / (TP + FN) = 正确预测的点 / 所有目标点

    参数：
    - pred: 预测点云 (B, N, 3)
    - target: 目标点云 (B, M, 3)
    - threshold: 距离阈值，用于判断点是否匹配
    - reduction: 'mean' 或 'sum'

    返回：
    - F-score 值 (tensor)
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred/target 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B = pred.shape[0]
    device = pred.device
    f_scores = []

    for b in range(B):
        pred_b = pred[b]  # (N, 3)
        target_b = target[b]  # (M, 3)

        # 计算 pred 到 target 的最近距离
        dist_pred_to_target = torch.cdist(pred_b, target_b, p=2).min(dim=1)[0]  # (N,)
        # 计算 target 到 pred 的最近距离
        dist_target_to_pred = torch.cdist(target_b, pred_b, p=2).min(dim=1)[0]  # (M,)

        # 计算 precision: pred 中距离 target < threshold 的点数 / pred 总数
        precision = (dist_pred_to_target < threshold).float().mean()

        # 计算 recall: target 中距离 pred < threshold 的点数 / target 总数
        recall = (dist_target_to_pred < threshold).float().mean()

        # 计算 F-score
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = torch.tensor(0.0, device=device)

        f_scores.append(f_score)

    f_scores = torch.stack(f_scores)  # (B,)

    if reduction == 'mean':
        return f_scores.mean()
    elif reduction == 'sum':
        return f_scores.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")


def precision_recall_fscore(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.01,
    reduction: str = 'mean',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 precision, recall 和 F-score。

    返回：
    - (precision, recall, fscore)
    """
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred/target 应为 3D 张量，形如 (B,N,3)/(B,M,3)")

    B = pred.shape[0]
    device = pred.device
    precisions, recalls, f_scores = [], [], []

    for b in range(B):
        pred_b = pred[b]  # (N, 3)
        target_b = target[b]  # (M, 3)

        # 计算 pred 到 target 的最近距离
        dist_pred_to_target = torch.cdist(pred_b, target_b, p=2).min(dim=1)[0]  # (N,)
        # 计算 target 到 pred 的最近距离
        dist_target_to_pred = torch.cdist(target_b, pred_b, p=2).min(dim=1)[0]  # (M,)

        # 计算 precision 和 recall
        precision = (dist_pred_to_target < threshold).float().mean()
        recall = (dist_target_to_pred < threshold).float().mean()

        # 计算 F-score
        if precision + recall > 0:
            f_score = 2 * (precision * recall) / (precision + recall)
        else:
            f_score = torch.tensor(0.0, device=device)

        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)

    precisions = torch.stack(precisions)  # (B,)
    recalls = torch.stack(recalls)  # (B,)
    f_scores = torch.stack(f_scores)  # (B,)

    if reduction == 'mean':
        return precisions.mean(), recalls.mean(), f_scores.mean()
    elif reduction == 'sum':
        return precisions.sum(), recalls.sum(), f_scores.sum()
    else:
        raise ValueError("reduction 必须是 'mean' 或 'sum'")
