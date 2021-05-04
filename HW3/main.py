# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd, inv


def sift(img):
    sift_descriptor = cv2.SIFT_create()
    kp, des = sift_descriptor.detectAndCompute(img, None)
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
    # for i in range(iter_num):
    while True:
        rand_idx = np.arange(pt_num)
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:sample_num]
        sample_src_pts = src_pts[rand_idx]
        sample_dest_pts = dest_pts[rand_idx]

        h = get_homography(sample_src_pts, sample_dest_pts)
        inliers = []

        src_pts_ext = np.hstack((src_pts, np.ones((pt_num, 1))))
        dest_pts_ext = np.hstack((dest_pts, np.ones((pt_num, 1))))

        estimate = h @ src_pts_ext.T
        estimate = estimate / estimate[-1,:]
        estimate = estimate.T
        for j in range(pt_num):
            error = norm(estimate[j]-dest_pts_ext[j])        
            if error < error_thres:
                inliers.append((src_pts[j], dest_pts[j]))

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            optimal_h = h
        if len(max_inliers) > pt_num*inlier_thres:
            break

    return optimal_h, max_inliers


def get_image_coors(h, w):
    coors = np.empty((h, w, 2), dtype=np.float32)
    for i, t in enumerate(np.ix_(np.arange(h), np.arange(w))):
        coors[...,i] = t
    return coors


def bilinear(img, x, y, x1, y1, x2, y2):
    res = (img[x1,y1]*(x2-x)*(y2-y)
            + img[x1,y2]*(x2-x)*(y-y1)
            + img[x2,y1]*(x-x1)*(y2-y)
            + img[x2,y2]*(x-x1)*(y-y1))
    return res.astype(np.uint8)


def nn(img, x, y, x1, y1, x2, y2):
    nn_x = x2 if x-x1>0.5 else x1
    nn_y = y2 if y-y1>0.5 else y1
    return img[nn_x,nn_y].astype(np.uint8)


def mapping(img, coors):
    ret = np.zeros((coors.shape[0], coors.shape[1], img.shape[2]), dtype=np.uint8)
    h, w, _ = coors.shape
    for i in range(h):
        for j in range(w):
            x, y = coors[i,j,1], coors[i,j,0]
            x1, y1 = int(x), int(y)
            x2, y2 = x1+1, y1+1
            if x1>=0 and x2<img.shape[0] and y1>=0 and y2<img.shape[1]:
                ret[i,j] = bilinear(img, x, y, x1, y1, x2, y2)
    return ret


def transform_coors(coors, trans):
    m_ext = np.hstack((coors, np.ones((coors.shape[0], 1))))
    res = trans @ m_ext.T
    res = res / res[-1,:]
    res = res.T
    return res


def warpping(img1, img2, homo):
    warp_x = img1.shape[0]
    warp_y = img1.shape[1] + img2.shape[1]
    image_coors = get_image_coors(warp_y, warp_x)

    image_coors = image_coors.reshape(-1, 2)
    image_coors_ori = transform_coors(image_coors, inv(homo))
    image_coors_ori = image_coors_ori.reshape(warp_y, warp_x, -1)[...,:2]
    resample = mapping(img1, image_coors_ori)
    return resample.transpose(1, 0, 2)


if __name__ == '__main__':
    img1 = cv2.imread('./data/2.jpg', cv2.IMREAD_COLOR)[:,:,::-1]
    img2 = cv2.imread('./data/1.jpg', cv2.IMREAD_COLOR)[:,:,::-1]
    kp1, des1 = sift(img1)
    kp2, des2 = sift(img2)
    src_pts, dest_pts = match_feature(img1, kp1, des1, img2, kp2, des2, 0.5)
    h, inliers = ransac(src_pts, dest_pts, 5, 3000, 5, 0.9)
    print(h)
    res = warpping(img1, img2, h)
    plt.imshow(res)
    plt.savefig('3_warpping.png', dpi=300)
