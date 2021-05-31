# -*- coding: utf-8 -*-
from datetime import time
import os
import cv2
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import kNN


def sift(dataset):
    sift_descriptor = cv2.SIFT_create()
    des_per_x = []
    y = []
    for data in tqdm(dataset, ncols=80):
        kp, des = sift_descriptor.detectAndCompute(np.array(data[0]), None)
        des_per_x.append(des)
        y.append(data[1])
    return des_per_x, y


def quantize(model, des_per_image, num_clusters, normalize=True):
    feature = np.zeros((len(des_per_image), num_clusters))
    for i in range(len(des_per_image)):
        _, assign_idx = model.assign(des_per_image[i])
        u, counts = np.unique(assign_idx, return_counts=True)
        counts = counts.astype(np.float32)
        feature[i,u] = counts
        if normalize:
            feature[i,u] /= counts.sum()
    return torch.tensor(feature)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls_mode', type=str, default='knn', help='knn, svm, nn')
    parser.add_argument('--repr_mode', type=str, default='tiny', help='tiny, sift')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--log_fn', type=str, default='log.json')
    
    # for feature (tiny)
    parser.add_argument('--img_size', type=int, default=16)
    parser.add_argument('--normalize', action='store_true', default=False)

    # for feature (sift)
    parser.add_argument('--num_clusters', type=int, default=300)

    # for model (kNN)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--norm', type=int, default=2)

    # for model (SVM)
    parser.add_argument('--c', type=float, default=1.0)

    args = parser.parse_args()

    log_fn = args.log_fn
    log = json.load(open(log_fn, 'r')) if os.path.exists(log_fn) else []
    json_args = {'time': int(time.time()), **vars(args)}
    json_args.pop('log_fn')
    train_dir = os.path.join(args.data_root, 'train')
    test_dir = os.path.join(args.data_root, 'test')

    # feature
    if args.repr_mode == 'tiny':
        tf = [
            transforms.Grayscale(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
        tf = transforms.Compose(tf)

        dataset = ImageFolder(train_dir, tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_train, y_train = next(iter(dataset))
        x_train = x_train.view(x_train.size(0), -1)
        if args.normalize:
            x_train = (x_train-x_train.mean(1)[:,None]) / (x_train.std(1)[:,None]+1e-6)

        dataset = ImageFolder(test_dir, tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_test, y_test = next(iter(dataset))
        x_test = x_test.view(x_test.size(0), -1)
        if args.normalize:
            x_test = (x_test-x_test.mean(1)[:,None]) / (x_test.std(1)[:,None]+1e-6)

        json_args.pop('num_clusters')

    elif args.repr_mode == 'sift':
        tf = [transforms.Grayscale()]
        tf = transforms.Compose(tf)

        print('Find descriptors of training images....')
        dataset = ImageFolder(train_dir, tf)
        des_per_x_train, y_train = sift(dataset)

        print('Find descriptors of testing images....')
        dataset = ImageFolder(test_dir, tf)
        des_per_x_test, y_test = sift(dataset)

        print('Find centroids with K-means....')
        des_vstack = np.vstack(des_per_x_train)
        model = faiss.Kmeans(
                    d=des_vstack.shape[1],
                    k=args.num_clusters,
                    gpu=True, niter=300, nredo=10)
        model.train(des_vstack)

        print('Vecter quantization....')
        x_train = quantize(model, des_per_x_train, args.num_clusters)
        x_test = quantize(model, des_per_x_test, args.num_clusters)
        y_train = torch.tensor(y_train).type(torch.int64)
        y_test = torch.tensor(y_test).type(torch.int64)

        json_args.pop('img_size')
        json_args.pop('normalize')

    else:
        raise NotImplementedError

    # model
    if args.cls_mode == 'knn':
        model = kNN.kNN(args.k, x_train, y_train, args.norm)
        y_pred, acc = model.predict(x_test, y_test)
        print(f'test acc: {acc:.4f}')
        json_args['acc'] = acc.item()
        json_args.pop('c')

    elif args.cls_mode == 'svm':
        svc = SVC(C=args.c)
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        acc = (torch.tensor(y_pred)==y_test).sum() / y_test.size(0)
        print(f'test acc: {acc:.4f}')
        json_args['acc'] = acc.item()
        json_args.pop('k')
        json_args.pop('norm')

    elif args.cls_mode == 'nn':
        raise NotImplementedError

    else:
        raise NotImplementedError

    # print(json_args)
    log.append(json_args)
    json.dump(log, open(log_fn, 'w'), indent=2)
