import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseRegressor(nn.Module):
    """
    点级别的 MLP 回归器：输入为 [模板点(3), 全局足模特征(d)]，输出为位移向量(3)。
    采用共享 MLP（逐点线性层），对每个模板点独立回归。
    """

    def __init__(self, global_feat_dim: int, hidden_dims: list[int] | tuple[int, ...] = (256, 256, 128), dropout_p: float = 0.1):
        super().__init__()
        in_dim = 3 + global_feat_dim
        dims = [in_dim, *hidden_dims, 3]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, template_points: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        template_points: (B, N, 3)
        global_feat: (B, d)
        返回: offsets (B, N, 3)
        """
        B, N, _ = template_points.shape
        # 广播全局特征到每个点
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)  # (B,N,d)
        x = torch.cat([template_points, global_expanded], dim=-1)  # (B,N,3+d)
        x = x.view(B * N, -1)
        offsets = self.mlp(x).view(B, N, 3)
        return offsets


class DeformationNet(nn.Module):
    """
    变形网络：模板点 + 全局足模特征 -> 预测鞋垫点云（采用残差：pred = template + offset）。
    编码器（如 DGCNN）需在外部传入。
    """

    def __init__(self, global_feat_dim: int, hidden_dims: list[int] | tuple[int, ...] = (256, 256, 128), dropout_p: float = 0.1):
        super().__init__()
        self.regressor = PointwiseRegressor(global_feat_dim=global_feat_dim, hidden_dims=hidden_dims, dropout_p=dropout_p)

    def forward(self, template_points: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        offsets = self.regressor(template_points, global_feat)
        return template_points + offsets


