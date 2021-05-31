#!/bin/bash
# tiny + kNN
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 1
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 1
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 3
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 3
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 5
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 5
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 7
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 7
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 9
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 9
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 11
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 11
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 13
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 13
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --normalize --k 15
python main.py --cls_mode knn --repr_mode tiny --log_fn tiny_knn.json --k 15

# SIFT + kNN
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 100 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 200 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 400 --k 15
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 1
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 3
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 5
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 7
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 9
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 11
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 13
python main.py --cls_mode knn --repr_mode sift --log_fn sift_knn.json --num_clusters 800 --k 15

# SIFT + SVM
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 100 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 200 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 400 --c 1000
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 0.01
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 0.1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 1
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 10
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 100
python main.py --cls_mode svm --repr_mode sift --log_fn sift_svm.json --num_clusters 800 --c 1000
