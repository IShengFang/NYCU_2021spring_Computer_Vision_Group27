#!/bin/bash
# tiny + kNN
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 1
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 1
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 3
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 3
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 5
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 5
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 7
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 7
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 9
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 9
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 11
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 11
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 13
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 13
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 15
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 15
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 17
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 17
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 19
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 19
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 21
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 21
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 23
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 23
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 25
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 25
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 27
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 27
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 29
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 29
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --standardize --k 31
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 31

# SIFT + kNN
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 17
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 19
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 21
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 23
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 25
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 27
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 29
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 150 --k 31
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 17
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 19
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 21
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 23
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 25
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 27
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 29
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 300 --k 31
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 17
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 19
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 21
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 23
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 25
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 27
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 29
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 450 --k 31
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 17
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 19
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 21
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 23
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 25
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 27
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 29
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 600 --k 31

# SIFT + SVM
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 0.001
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 150 --c 10000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 0.001
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 300 --c 10000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 0.001
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 450 --c 10000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 0.001
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 600 --c 10000