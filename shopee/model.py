from torch import nn

from shopee.models.heads import Neck, ResNetHead
from shopee.models.resnet import ResNetBase


def build_model(backbone: str, n_classes: int, **kwargs) -> nn.Module:
    return Model(backbone=backbone, n_classes=n_classes, **kwargs)


class Model(nn.Module):
    def __init__(self, *, backbone: str, n_classes: int, use_neck: bool,):
        super().__init__()

        self.backbone = ResNetBase(backbone)
        self.in_features = self.backbone.out_features
        self.use_neck = use_neck
        if self.use_neck:
            self.hidden_dim = 1024
            self.neck = Neck(self.in_features, 1024)
            self.in_features = self.hidden_dim
        self.head = ResNetHead(self.in_features, n_classes, self.use_neck)

    def forward(self, x):
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x
