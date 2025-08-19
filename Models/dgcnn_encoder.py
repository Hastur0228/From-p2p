import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    以批为单位计算 k 近邻（基于欧氏距离）。
    输入: x 形状 (B, C, N)
    返回: 邻接索引 idx 形状 (B, N, k)
    """
    # x -> (B, N, C)
    x = x.transpose(2, 1).contiguous()
    # 使用 cdist 计算两两距离，避免显式构造 (x_i - x_j)^2
    # cdist 返回欧氏距离，我们只需 topk 最小（取负后 topk 最大）
    dist = torch.cdist(x, x, p=2)  # (B, N, N)
    # 为了与常见实现一致，这里取最小的 k 个邻居（排除自身不会影响稳定性）
    idx = torch.topk(-dist, k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    构造 EdgeConv 的图特征: [x_j - x_i, x_i]
    输入: x (B, C, N)
    输出: (B, 2C, N, k)
    """
    B, C, N = x.size()
    idx = knn(x, k=k)  # (B, N, k)

    # 根据 idx 索引邻居特征
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # (B,1,1)
    idx = (idx + idx_base).view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(B * N, C)[idx, :]  # (B*N*k, C)
    feature = feature.view(B, N, -1, C)  # (B, N, k, C)
    x_i = x.view(B, N, 1, C).repeat(1, 1, feature.size(2), 1)  # (B, N, k, C)
    feature = torch.cat((feature - x_i, x_i), dim=3)  # (B, N, k, 2C)
    feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)
    return feature


class EdgeConvBlock(nn.Module):
    """DGCNN 中的基本 EdgeConv 模块: [x_j - x_i, x_i] -> Conv2d -> BN -> LeakyReLU -> max_k"""

    def __init__(self, in_channels: int, out_channels: int, k: int = 20, dropout_p: float = 0.1):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)
        features = get_graph_feature(x, k=self.k)  # (B, 2C, N, k)
        features = self.conv(features)  # (B, out, N, k)
        features, _ = torch.max(features, dim=-1, keepdim=False)  # (B, out, N)
        return features


class DGCNNEncoder(nn.Module):
    """
    DGCNN 编码器，将点云 (B, C_in, N) 编码为全局特征向量 (B, feat_dim)。
    - C_in 可为 3（xyz）或 6（xyz+normals）。
    - 默认为 4 个 EdgeConv 层，拼接后再经 MLP 聚合为全局特征。
    """

    def __init__(self, input_dims: int = 3, k: int = 20, feat_dim: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.k = k

        self.ec1 = EdgeConvBlock(input_dims, 64, k, dropout_p=dropout_p)
        self.ec2 = EdgeConvBlock(64, 64, k, dropout_p=dropout_p)
        self.ec3 = EdgeConvBlock(64, 128, k, dropout_p=dropout_p)
        self.ec4 = EdgeConvBlock(128, 256, k, dropout_p=dropout_p)

        # 拼接四层输出: 64 + 64 + 128 + 256 = 512
        concat_channels = 64 + 64 + 128 + 256
        self.mlp = nn.Sequential(
            nn.Conv1d(concat_channels, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(1024, feat_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C_in) 或 (B, C_in, N)
        返回: 全局特征 (B, feat_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"期望输入为 3D 张量 (B,N,C) 或 (B,C,N)，得到 {x.shape}")
        # 统一为 (B, C_in, N)
        if x.size(1) != 3 and x.size(1) != 6:
            # 可能是 (B, N, C)
            x = x.transpose(2, 1).contiguous()

        x1 = self.ec1(x)  # (B,64,N)
        x2 = self.ec2(x1)  # (B,64,N)
        x3 = self.ec3(x2)  # (B,128,N)
        x4 = self.ec4(x3)  # (B,256,N)

        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B,512,N)
        feat_map = self.mlp(x_cat)  # (B, feat_dim, N)
        # 全局池化（max + mean 也可，这里用 max）
        global_feat = torch.max(feat_map, dim=2, keepdim=False)[0]  # (B, feat_dim)
        return global_feat


