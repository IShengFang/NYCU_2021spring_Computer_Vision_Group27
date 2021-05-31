# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


def cosine_similarity(x_test, x_train):
    dot = x_test @ x_train.T
    x_test_norm = torch.norm(x_test, dim=1)
    x_train_norm = torch.norm(x_train, dim=1)
    dot = dot / x_test_norm[:,None]
    dot = dot / x_train_norm
    return 1 - dot


class kNN(nn.Module):
    def __init__(self, k, x_train, y_train, norm=2, use_cosine=False, num_classes=15):
        super().__init__()
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        self.norm = norm
        self.use_cosine = use_cosine
        self.num_classes = num_classes

    def forward(self, x_test):
        dist = torch.cdist(x_test, self.x_train, p=self.norm)
        values, indices = dist.topk(self.k)
        return values, indices

    def classification(self, x_test, y_test):
        if self.use_cosine:
            dist = cosine_similarity(x_test, self.x_train)
        else:
            dist = torch.cdist(x_test, self.x_train, p=self.norm)
        values, indices = dist.topk(self.k)
        y_pred = self.y_train[indices].reshape(x_test.size(0), -1)
        count, y_pred = F.one_hot(y_pred, self.num_classes).sum(1).max(1)
        acc = (y_test==y_pred).float().mean()
        return y_pred, acc
