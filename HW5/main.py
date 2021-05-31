# -*- coding: utf-8 -*-
import cv2
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import kNN

EPS = 1e-6


def sift(img):
    sift_descriptor = cv2.SIFT_create()
    kp, des = sift_descriptor.detectAndCompute(img, None)
    return kp, des


def bag_of_sift(model, des_per_image, num_clusters):
    feature = np.zeros((len(des_per_image), num_clusters))
    for i in range(len(des_per_image)):
        _, assign_idx = model.assign(des_per_image[i])
        u, counts = np.unique(assign_idx, return_counts=True)
        counts = counts.astype(np.float32)
        feature[i,u] = counts / counts.sum()
        # feature[i,u] = counts
    return torch.tensor(feature)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls_mode', type=str, default='knn', help='knn, svm, nn')
    parser.add_argument('--repr_mode', type=str, default='tiny', help='tiny, sift')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--img_size', type=int, default=16)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--cosine', action='store_true', default=False)
    parser.add_argument('--num_clusters', type=int, default=300)
    args = parser.parse_args()

    # feature
    if args.repr_mode == 'tiny':
        tf = [
            transforms.Grayscale(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
        tf = transforms.Compose(tf)

        dataset = ImageFolder('./data/train/', tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_train, y_train = next(iter(dataset))
        x_train = x_train.view(x_train.size(0), -1)

        dataset = ImageFolder('./data/test/', tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_test, y_test = next(iter(dataset))
        x_test = x_test.view(x_test.size(0), -1)

    elif args.repr_mode == 'sift':
        tf = [transforms.Grayscale()]
        tf = transforms.Compose(tf)

        print('Find descriptors of training images....')
        dataset = ImageFolder('./data/train/', tf)
        des_per_x_train = []
        y_train = []
        for (x, y) in tqdm(dataset, ncols=80):
            kp, des = sift(np.array(x))
            des_per_x_train.append(des)
            y_train.append(y)
        des_vstack = np.vstack(des_per_x_train)

        print('Find descriptors of testing images....')
        dataset = ImageFolder('./data/test/', tf)
        des_per_x_test = []
        y_test = []
        for (x, y) in tqdm(dataset, ncols=80):
            kp, des = sift(np.array(x))
            des_per_x_test.append(des)
            y_test.append(y)

        print('Find centroids with K-means....')
        max_iter = 300
        model = faiss.Kmeans(
                    d=des_vstack.shape[1],
                    k=args.num_clusters,
                    gpu=True, niter=max_iter, nredo=10)
        model.train(des_vstack)

        x_train = bag_of_sift(model, des_per_x_train, args.num_clusters)
        x_test = bag_of_sift(model, des_per_x_test, args.num_clusters)
        y_train = torch.tensor(y_train).type(torch.int64)
        y_test = torch.tensor(y_test).type(torch.int64)

    else:
        raise NotImplementedError

    # method
    if args.cls_mode == 'knn':        
        model = kNN.kNN(args.k, x_train, y_train, args.norm, args.cosine)
        y_pred, acc = model.classification(x_test, y_test)
        print(f'Test accuracy: {acc:.4f}')

    elif args.cls_mode == 'svm':
        svc = SVC()
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        acc = (torch.tensor(y_pred)==y_test).sum() / y_test.size(0)
        print(f'Test accuracy: {acc:.4f}')

    elif args.cls_mode == 'nn':
        raise NotImplementedError

    else:
        raise NotImplementedError
