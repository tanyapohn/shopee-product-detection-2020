import numpy as np
import torch
from torch import nn


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    rand_index = torch.randperm(data.size(0))

    shuffled_data = data[rand_index]
    shuffled_target = target[rand_index]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [
        target,
        shuffled_target,
        lam,
    ]
    return data, targets


def cutmix_criterion(preds, targets,):
    target, shuffled_target, lam = (
        targets[0], targets[1], targets[2],
    )
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return (
            lam * criterion(preds, target) + (1 - lam) * criterion(preds, shuffled_target)
    )
