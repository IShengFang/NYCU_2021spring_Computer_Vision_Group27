# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt


def serialize(data):
    res = []
    for d in data:
        res.append(list(d.values()))
    return np.array(res)

if __name__ == '__main__':
    ACC = -1
    C = -2
    K = -3
    NUM_CLUSTERS = 3
    STANDARDIZE = 4

    # tiny_knn
    # time, repr_mode, cls_mode, img_size, standardize, k, norm, acc
    plt.clf()
    all_data = json.load(open('tiny_knn.json'))
    all_data = serialize(all_data)

    data = all_data[all_data[:,STANDARDIZE]=='True']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='w/ standardize', marker='.')

    data = all_data[all_data[:,STANDARDIZE]=='False']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='w/o standardize', marker='x')

    plt.title('Tiny representation + kNN')
    plt.xticks(k)
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid('on')
    plt.savefig('tiny_knn.png', dpi=150)

    # sift_knn
    # time, repr_mode, cls_mode, num_clusters, k, norm, acc
    plt.clf()
    all_data = json.load(open('sift_knn.json'))
    all_data = serialize(all_data)

    data = all_data[all_data[:,NUM_CLUSTERS]=='150']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 150', marker='.')

    data = all_data[all_data[:,NUM_CLUSTERS]=='300']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 300', marker='x')

    data = all_data[all_data[:,NUM_CLUSTERS]=='450']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 450', marker='o')

    data = all_data[all_data[:,NUM_CLUSTERS]=='600']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 600', marker='^')

    plt.title('SIFT representation + kNN')
    plt.xticks(k)
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid('on')
    plt.savefig('sift_knn.png', dpi=150)

    # sift_svm
    # time, repr_mode, cls_mode, num_clusters, acc
    plt.clf()
    all_data = json.load(open('sift_svm.json'))
    all_data = serialize(all_data)

    acc = all_data[:,ACC].astype(np.float32) * 100
    x = np.arange(acc.shape[0])
    plt.plot(x, acc, marker='.')

    plt.title('SIFT representation + SVM')
    plt.xticks(x, all_data[:,NUM_CLUSTERS])
    plt.xlabel('number of clusters')
    plt.ylabel('Accuracy (%)')
    plt.grid('on')
    plt.savefig('sift_svm.png', dpi=150)
