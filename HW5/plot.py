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
    NORMALIZE = 4

    # tiny_knn
    # time, repr_mode, cls_mode, img_size, normalize, k, norm, acc
    plt.clf()
    all_data = json.load(open('tiny_knn.json'))
    all_data = serialize(all_data)

    data = all_data[all_data[:,NORMALIZE]=='True']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='w/ normalize', marker='.')

    data = all_data[all_data[:,NORMALIZE]=='False']
    k = data[:,K].astype(np.int32)
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='w/o normalize', marker='x')

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
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 300', marker='x')

    data = all_data[all_data[:,NUM_CLUSTERS]=='450']
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(k, acc, label='number of clusters: 450', marker='o')

    data = all_data[all_data[:,NUM_CLUSTERS]=='600']
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
    # time, repr_mode, cls_mode, num_clusters, c, acc
    plt.clf()
    all_data = json.load(open('sift_svm.json'))
    all_data = serialize(all_data)

    data = all_data[all_data[:,NUM_CLUSTERS]=='150']
    c = data[:,C].astype(np.float32)
    x = np.arange(c.shape[0])
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(x, acc, label='number of clusters: 150', marker='.')

    data = all_data[all_data[:,NUM_CLUSTERS]=='300']
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(x, acc, label='number of clusters: 300', marker='x')

    data = all_data[all_data[:,NUM_CLUSTERS]=='450']
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(x, acc, label='number of clusters: 450', marker='o')

    data = all_data[all_data[:,NUM_CLUSTERS]=='600']
    acc = data[:,ACC].astype(np.float32) * 100
    plt.plot(x, acc, label='number of clusters: 600', marker='^')

    plt.title('SIFT representation + SVM')
    plt.xticks(x, c)
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid('on')
    plt.savefig('sift_svm.png', dpi=150)
