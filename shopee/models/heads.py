import torch
from torch import nn
from torch.nn import functional as F


class ResNetHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int, use_neck: bool):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features, n_classes)
        self.use_neck = use_neck

    def forward(self, x):
        if not self.use_neck:
            x = self.pooling(x)
            x = torch.flatten(x, start_dim=1)
        x = self.apply_fc_out(x)
        return x

    def apply_fc_out(self, x):
        return self.fc1(x)


class Neck(nn.Module):
    def __init__(self, in_features: int, hidden_dim):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):

        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.bn3(x)

        return x
