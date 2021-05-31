# -*- coding: utf-8 -*-
import cv2
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import kNN


def sift(img):
    sift_descriptor = cv2.SIFT_create()
    kp, des = sift_descriptor.detectAndCompute(img, None)
    return kp, des


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls_mode', type=str, default='knn', help='knn, svm, nn')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_size', type=int, default=16)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--norm', type=int, default=2)
    args = parser.parse_args()

    if args.cls_mode == 'knn':
        tf = [
            transforms.Grayscale(),
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ]
        # tf = [
        #     transforms.Grayscale(),
        #     transforms.Resize((args.img_size, args.img_size)),
        #     transforms.ToTensor()
        # ]
        tf = transforms.Compose(tf)

        dataset = ImageFolder('./data/train/', tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_train, y_train = next(iter(dataset))
        x_train = x_train.view(x_train.size(0), -1)

        dataset = ImageFolder('./data/test/', tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_test, y_test = next(iter(dataset))
        x_test = x_test.view(x_test.size(0), -1)

        model = kNN.kNN(args.k, x_train, y_train, args.norm)
        y_pred, acc = model.classification(x_test, y_test)
        print(f'test acc: {acc}')
