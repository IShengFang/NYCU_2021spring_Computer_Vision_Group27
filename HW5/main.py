# -*- coding: utf-8 -*-
import os
import cv2
import json
import time
import faiss
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from libsvm.svmutil import svm_train, svm_predict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

import kNN, model


def sift(dataset):
    sift_descriptor = cv2.SIFT_create(
                            nfeatures=0,
                            nOctaveLayers=5,
                            contrastThreshold=0.01,
                            edgeThreshold=80,
                            sigma=0.6)
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


def generate_nn_exp_name(args):
    if args.pretrained:
        args.nn_exp_name += '_Pretrain'
        if args.fixed_weight:
            args.nn_exp_name += '_fixed'
    else:
        args.nn_exp_name += '_Scratch'

    args.nn_exp_name += '_{}_E{}_lr{}_b1_{}_b2_{}_bs_{}'.format(
        args.nn_model_name, args.epochs, args.lr, args.beta_1, args.beta_2,
        args.batch_size)

    return args.nn_exp_name


def train(net, writer, train_loader, valid_loader,
          optimizer, criterion, epochs, device, cpt_num, args):
    step = 0
    print('start training...')
    exp_pbar = tqdm(range(epochs))
    for epoch in exp_pbar:
        net.train()
        running_loss = 0
        batch_idx = 0
        epoch_pbar = tqdm(train_loader)
        for imgs, labels in epoch_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = net(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('train/loss', loss.item(), step)
            step += 1
            epoch_pbar.set_description(f'epoch: {epoch:>2}/{epochs}, batch: {batch_idx:>3}/{len(train_loader)}, loss: {loss.item():.5f}')

            if step%cpt_num == 0:
                save_path = os.path.join(args.cpt_dir, '{}_E_{}_iter_{}.cpt'.format(args.nn_model_name, epoch, step,))
                print('saving model cpt @ {}'.format(save_path))
                torch.save(net.state_dict(), save_path)

            batch_idx += 1

        running_loss /= batch_idx
        print('----------------------------')
        exp_pbar.set_description(f'epoch: {epoch:>2}/{epochs}, avg. loss: {running_loss:.5f}')
        evaluate(net, valid_loader, train_loader, criterion, writer, device, step)
        print('============================')

        save_path = os.path.join(args.cpt_dir, '{}_E_{}_iter_{}.cpt'.format(
                                                args.nn_model_name, epoch, step,))
        print('saving model cpt @ {}'.format(save_path))
        torch.save(net.state_dict(), save_path)


def evaluate(net, valid_loader, train_loader, criterion, logger, device, step):
    net.eval()
    y_true = []
    y_pred = []
    print('start evaluating...')

    print('valid_set')
    eval_pbar = tqdm(valid_loader)
    correct = 0
    batch_idx = 0
    val_loss = 0
    for imgs, labels in eval_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = net(imgs)
            loss = criterion(output, labels)
            val_loss += loss.item()

        _, pred = output.data.max(1)
        correct += (labels==pred).sum().item()
        y_true += labels.data.cpu()
        y_pred += pred.data.cpu()
        batch_idx += 1
    val_loss = val_loss / batch_idx
    val_acc = correct / len(valid_loader.dataset)

    eval_pbar = tqdm(train_loader)
    correct = 0
    batch_idx = 0
    for imgs, labels in eval_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = net(imgs)
        _, pred = output.data.max(1)
        correct += (labels == pred).sum().item()
        y_true += labels.data.cpu()
        y_pred += pred.data.cpu()
        batch_idx += 1
    train_acc = correct / len(train_loader.dataset)
    logger.add_scalar('val/loss', val_loss, step)
    logger.add_scalar('val/total_acc', val_acc, step)
    logger.add_scalar('train/total_acc', train_acc, step)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cls_mode', type=str, default='knn', help='knn, svm, nn')
    parser.add_argument('--repr_mode', type=str, default='tiny', help='tiny, sift, nn')
    parser.add_argument('--data_root', type=str, default='./data')

    # used to save results of kNN and SVM
    parser.add_argument('--log_fn', type=str, default='log.json')
    
    # for feature (tiny)
    parser.add_argument('--img_size', type=int, default=16)
    parser.add_argument('--normalize', action='store_true', default=False)

    # for feature (sift)
    parser.add_argument('--num_clusters', type=int, default=300)

    # for model (kNN)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--norm', type=int, default=2)

    # for model (NN)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--cpt_num', type=int, default=1000)
    parser.add_argument('--nn_model_name', type=str,
                        default='resnet18',
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, \
                              resnext50_32x4d, resnext101_32x8d, \
                              wide_resnet50_2, wide_resnet101_2, \
                              densenet121, densenet169, densenet161, densenet201, \
                              inception_v3, googlenet, ')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--fixed_weight', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str,
                        default='logs/',
                        help='Directory path for save tensorboard log')
    parser.add_argument('--cpt_dir', type=str,
                        default='cpts/',
                        help='Directory path for save model cpt')
    parser.add_argument('--nn_exp_name', default='NN_scene_classifier',
                        help='Experiment name')
    parser.add_argument('--generate_exp_name', action='store_true', default=False)
    args = parser.parse_args()
    if args.generate_exp_name:
        args.nn_exp_name = generate_nn_exp_name(args)

    log_fn = args.log_fn
    log = json.load(open(log_fn, 'r')) if os.path.exists(log_fn) else []
    json_args = {
        'time': int(time.time()),
        'repr_mode': args.repr_mode,
        'cls_mode': args.cls_mode
    }
    train_dir = os.path.join(args.data_root, 'train')
    test_dir = os.path.join(args.data_root, 'test')

    # feature
    if args.repr_mode == 'tiny':
        print('Use tiny representation')
        tf = [transforms.Grayscale(),
              transforms.Resize((args.img_size, args.img_size)),
              transforms.ToTensor()]
        tf = transforms.Compose(tf)

        dataset = ImageFolder(train_dir, tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_train, y_train = next(iter(dataset))
        x_train = x_train.view(x_train.size(0), -1)

        dataset = ImageFolder(test_dir, tf)
        dataset = DataLoader(dataset, batch_size=len(dataset))
        x_test, y_test = next(iter(dataset))
        x_test = x_test.view(x_test.size(0), -1)

        if args.normalize:
            x_train = (x_train-x_train.mean(1)[:,None]) / (x_train.std(1)[:,None]+1e-6)
            x_test = (x_test-x_test.mean(1)[:,None]) / (x_test.std(1)[:,None]+1e-6)

        json_args['img_size'] = args.img_size
        json_args['normalize'] = args.normalize

    elif args.repr_mode == 'sift':
        print('Use SIFT for representation')
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
        km_model = faiss.Kmeans(
                    d=des_vstack.shape[1],
                    k=args.num_clusters,
                    gpu=True, niter=300, nredo=10, verbose=True)
        km_model.train(des_vstack)

        print('Vecter quantization....')
        x_train = quantize(km_model, des_per_x_train, args.num_clusters)
        x_test = quantize(km_model, des_per_x_test, args.num_clusters)
        y_train = torch.tensor(y_train).type(torch.int64)
        y_test = torch.tensor(y_test).type(torch.int64)

        json_args['num_clusters'] = args.num_clusters

    elif args.repr_mode == 'nn':
        print('Use CNN for representation')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.log_dir = os.path.join(args.log_dir, args.nn_exp_name)
        os.makedirs(args.log_dir, exist_ok=True)
        args.cpt_dir = os.path.join(args.cpt_dir, args.nn_exp_name)
        os.makedirs(args.cpt_dir, exist_ok=True)
        tf = [transforms.RandomHorizontalFlip(p=0.5), 
              transforms.RandomResizedCrop(args.img_size),
              transforms.ToTensor()]
        tf = transforms.Compose(tf)

        train_dataset = ImageFolder(train_dir, tf)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.n_threads)

        tf = [transforms.Resize((args.img_size, args.img_size)),
              transforms.ToTensor()]
        tf = transforms.Compose(tf)

        test_dataset = ImageFolder(test_dir, tf)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                  num_workers=args.n_threads)

    else:
        raise NotImplementedError

    # model
    if args.cls_mode == 'knn':
        print('Use kNN for modeling')
        model = kNN.kNN(args.k, x_train, y_train, args.norm)
        y_pred, acc = model.predict(x_test, y_test)
        print(f'test acc: {acc:.4f}')
        json_args['k'] = args.k
        json_args['norm'] = args.norm
        json_args['acc'] = acc.item()
        log.append(json_args)
        json.dump(log, open(log_fn, 'w'), indent=2)

    elif args.cls_mode == 'svm':
        print('Use SVM for modeling')
        model = svm_train(
                    y_train.numpy(), x_train.numpy(),
                    f'-s 0 -t 0 -c {args.c} -q')
        res = svm_predict(y_test.numpy(), x_test.numpy(), model, '-q')
        acc = res[1][0] / 100
        print(f'test acc: {acc:.4f}')
        json_args['acc'] = acc
        log.append(json_args)
        json.dump(log, open(log_fn, 'w'), indent=2)

    elif args.cls_mode == 'nn':
        print('Use CNN for modeling')
        writer = SummaryWriter(args.log_dir)
        net = model.CNN_Model(args.nn_model_name, args.pretrained)
        criterion = nn.CrossEntropyLoss()
        net = net.to(device)
        if args.fixed_weight:
            optimizer = torch.optim.Adam(net.fc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        train(net, writer, train_loader, test_loader, optimizer, 
              criterion, args.epochs, device, args.cpt_num, args)

    else:
        raise NotImplementedError
