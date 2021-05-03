# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
from numpy.linalg import norm, svd, inv
import matplotlib.pyplot as plt


def sift(image):
    sift_descriptor = cv2.SIFT_create()
    kp, des = sift_descriptor.detectAndCompute(image, None)
    return kp, des


def matching(des1, des2, k=1):
    matches = []
    for i in range(des1.shape[0]):
        dis = []
        for j in range(des2.shape[0]):
            dis.append((norm(des1[i]-des2[j]), j, i))
        dis = sorted(dis, key=lambda x: x[0])
        dms = []
        for t in range(k):
            dm = cv2.DMatch(
                    _distance=dis[t][0],
                    _trainIdx=dis[t][1],
                    _queryIdx=dis[t][2])
            dms.append(dm)
        matches.append(dms[0] if k==1 else dms)
    return matches

# def drawMatchesKnn(img1, kp1, img2, kp2, good):

#     # Combine two images
#     result = np.concatenate((img1, img2), axis=1)

#     # Collect descriptors' coordinate
#     src_temp = []
#     dst_temp = []
#     for i in good:
#         src_temp.append(kp1[i[0].queryIdx].pt)
#         dst_temp.append(kp2[i[0].trainIdx].pt)
#     src_pts = np.asarray(src_temp) # (n,2)
#     dst_pts = np.asarray(dst_temp) # (n,2)

#     n, _ = src_pts.shape
#     h1, w1, c1 = img1.shape
    
#     for i in range(n):
#         color = randomcolor()
#         x1 = int(src_pts[i,0])
#         y1 = int(src_pts[i,1])
#         cv2.circle(result, (x1, y1), 7, color, 1)

#         x2 = w1 + int(dst_pts[i,0])
#         y2 = int(dst_pts[i,1])
#         cv2.circle(result, (x2, y2), 7, color, 1)

#         # (y-y1) / (y2-y1) - (x-x1) / (x2-x1)
#         for x in range(x1, x2):
#             y = y1 + ((x-x1)/(x2-x1)*(y2-y1))
#             result[int(y), x] = color
#         #cv2.line(result, (x1,y1), (x2, y2), color, 1)

#     return result


def match_feature(img1, kp1, des1, img2, kp2, des2, ratio):
    if ratio is None:
        matches = matching(des1, des2, k=1)
    else:
        matches_knn = matching(des1, des2, k=2)
        matches = []
        for m1, m2 in matches_knn:
            if m1.distance/m2.distance < ratio:
                matches.append(m1)
    matches = sorted(matches, key=lambda x: x.distance)

    plot_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=0)
    plt.imshow(plot_matches)
    plt.axis('off')
    plt.savefig('2_feature_matching.png', dpi=300)

    src_pts = []
    dest_pts = []
    for m in matches:
        src_pts.append(kp1[m.queryIdx].pt)
        dest_pts.append(kp2[m.trainIdx].pt)

    src_pts = np.array(src_pts)
    dest_pts = np.array(dest_pts)
    return src_pts, dest_pts


def get_homography(src_pts, dest_pts):
    P = np.zeros((src_pts.shape[0]*2, 9))
    for i, (src_pt, dest_pt) in enumerate(zip(src_pts, dest_pts)):
        PT_i = np.array([src_pt[0], src_pt[1], 1])
        P[i*2,:] = [*(-1*PT_i), 0, 0, 0, *(dest_pt[0]*PT_i)]
        P[i*2+1,:] = [0, 0, 0, *(-1*PT_i), *(dest_pt[1]*PT_i)]
    U, D, VT = svd(P, full_matrices=False)
    h = VT.T[:,-1]
    h /= h[-1]
    H = h.reshape(3, 3)
    return H


def ransac(src_pts, dest_pts, sample_num, iter_num, error_thres, inlier_thres):
    max_inliers = []
    optimal_h = None
    pt_num = len(src_pts)
    for i in range(iter_num):
        rand_idx = np.arange(pt_num)
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:sample_num]
        sample_src_pts = src_pts[rand_idx]
        sample_dest_pts = dest_pts[rand_idx]

        h = get_homography(sample_src_pts, sample_dest_pts)
        inliers = []

        src_pts_ext = np.hstack((src_pts, np.ones((pt_num, 1))))
        dest_pts_ext = np.hstack((dest_pts, np.ones((pt_num, 1))))

        estimate = (h@src_pts_ext.T).T
        for j in range(pt_num):
            error = norm(estimate[j]-dest_pts_ext[j])        
            if error < error_thres:
                inliers.append((src_pts[j], dest_pts[j]))
        if len(inliers) > len(max_inliers):
            max_inlier = inliers
            optimal_h = h
        if len(max_inliers) > pt_num*inlier_thres:
            break

    return optimal_h, max_inliers


if __name__ == '__main__':
    img1 = cv2.imread('./data/1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./data/2.jpg', cv2.IMREAD_GRAYSCALE)
    kp1, des1 = sift(img1)
    kp2, des2 = sift(img2)
    src_pts, dest_pts = match_feature(img1, kp1, des1, img2, kp2, des2, 0.6)
    h, inliers = ransac(src_pts, dest_pts, 10, 1000, 10, 0.7)
