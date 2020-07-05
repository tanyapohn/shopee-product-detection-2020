import torch
from torch import nn
from torchvision import models


class ResNetBase(nn.Module):
    def __init__(self, backbone='resnext50_32x4d_ssl'):
        super(ResNetBase, self).__init__()
        if backbone.endswith('l'):
            self.backbone = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models',
                backbone,
            )
        else:
            self.backbone = getattr(models, backbone)(pretrained=True)
        self.out_features = self.backbone.fc.in_features

    def forward(self, x):
        base = self.backbone
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)
        return x
