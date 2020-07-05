import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
import numpy as np


class RocMeter(object):
    """
    ROC-AUC metrics for multiclasses
    y_true: true labels
    y_pred: probabilities of all classes
    """

    def __init__(self):
        self.roc = 0
        self._y_true = torch.tensor([], dtype=torch.long).cuda()
        self._y_proba = torch.tensor([], dtype=torch.float).cuda()

    def reset(self):
        self.__init__()

    def update(self, y_true, y_pred, n=1):
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        y_proba = F.softmax(y_pred, dim=1)
        # y_true = y_true.data.cpu()
        self._y_true = torch.cat([self._y_true, y_true])
        self._y_proba = torch.cat([self._y_proba, y_proba])

    def compute(self):
        self._y_true = self._y_true.data.cpu().numpy()
        self._y_proba = self._y_proba.data.cpu().numpy()
        # print(np.unique(self._y_true))
        # print(self._y_true.shape)
        # print(self._y_proba.shape)
        return roc_auc_score(self._y_true, self._y_proba, multi_class='ovo')


class TopKAccuracy(object):
    def __init__(self, k=5):
        self._n_samples = 0
        self._n_corrects = 0
        self._k = k

    def reset(self):
        self.__init__()

    def update(self, y_true, y_pred):
        sorted_idx = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y_true = y_true.view(-1, 1).expand(-1, self._k)
        corrects = torch.sum(torch.eq(sorted_idx, expanded_y_true), dim=1)
        self._n_corrects += torch.sum(corrects).item()
        self._n_samples += corrects.shape[0]

    def compute(self):
        return self._n_corrects / self._n_samples
