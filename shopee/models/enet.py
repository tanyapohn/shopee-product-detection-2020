import efficientnet_pytorch as enet
from torch import nn
import torch
import math
from torch.nn import functional as F

sigmoid = torch.nn.Sigmoid()


class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

# ==============================================#


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Mish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        tanh_sp_i = F.softplus(i).tanh()
        return grad_output * (tanh_sp_i + i * sigmoid_i * (1 - tanh_sp_i * tanh_sp_i))


swish = Swish.apply
mish = Mish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


class Mish_module(nn.Module):
    def forward(self, x):
        return mish(x)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class enet_arcface_v2(nn.Module):
    def __init__(self, backbone,  out_dim, act_fn):
        super(enet_arcface_v2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)
        # self.enet.load_state_dict(torch.load(pretrained_dict[backbone]), strict=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

        self.gfc = nn.Linear(self.enet._fc.in_features, 4096)
        self.metric_classify = ArcMarginProduct(4096, out_dim)
        # self.myfc_1 = nn.Linear(4096, out_dim_1)
        self.myfc_2_1 = nn.Linear(4096, 512)
        self.myfc_2_2 = nn.Linear(512, out_dim)
        self.enet._fc = nn.Identity()
        self.act_fn = act_fn

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.act_fn(self.gfc(x))
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc_2_1(dropout(x))
            else:
                out += self.myfc_2_1(dropout(x))
        out /= len(self.dropouts)
        out = self.myfc_2_2(self.act_fn(out))
        metric_output = self.metric_classify(x)
        return out, metric_output