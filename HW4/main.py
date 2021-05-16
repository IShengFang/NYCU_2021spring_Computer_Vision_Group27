# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd


def sift(img):
    sift_descriptor = cv2.SIFT_create()
    kp, des = sift_descriptor.detectAndCompute(img, None)
    return kp, des


def matching(des1, des2):
    matches = []
    for i in range(des1.shape[0]):
        dis = [(norm(des1[i]-des2[j]), j, i) for j in range(des2.shape[0])]
        dis = sorted(dis, key=lambda x: x[0])
        dms = []
        for t in range(2):
            dm = cv2.DMatch(
                    _distance=dis[t][0],
                    _trainIdx=dis[t][1],
                    _queryIdx=dis[t][2])
            dms.append(dm)
        matches.append(dms)
    return matches


def match_feature(img1, kp1, des1, img2, kp2, des2, ratio, results_dir):
    matches = matching(des1, des2)
    good_match = []
    for m, n in matches:
        if m.distance/n.distance < ratio:
            good_match.append(m)
    plot_match = cv2.drawMatches(img1, kp1, img2, kp2, good_match[:50], None, flags=2)
    plt.imsave(f'{results_dir}/feature_matching.png', plot_match.astype(np.uint8))
    match_kp1 = []
    match_kp2 = []
    for m in good_match:
        match_kp1.append(kp1[m.queryIdx].pt)
        match_kp2.append(kp2[m.trainIdx].pt)
    match_kp1 = np.array(match_kp1)
    match_kp2 = np.array(match_kp2)
    return match_kp1, match_kp2


def normalize_ext(pts):
    center = np.mean(pts, axis=0)
    scale = np.sqrt(2) / norm(center-pts, axis=1).mean()
    T = np.array([
            [scale, 0, -scale*center[0]],
            [0, scale, -scale*center[1]],
            [0, 0, 1]
        ])
    pts_ext = np.hstack((pts, np.ones((pts.shape[0], 1))))
    norm_pts_ext = (T@pts_ext.T).T
    return norm_pts_ext, T


def run_8_point(match_kp1, match_kp2):
    assert match_kp1.shape[0]==match_kp2.shape[0] and match_kp1.shape[0]==8
    norm_kp1, T1 = normalize_ext(match_kp1)
    norm_kp2, T2 = normalize_ext(match_kp2)
    A = []
    for i in range(8):
        A.append([
            norm_kp2[i,0]*norm_kp1[i,0],
            norm_kp2[i,0]*norm_kp1[i,1],
            norm_kp2[i,0]*norm_kp1[i,2],
            norm_kp2[i,1]*norm_kp1[i,0],
            norm_kp2[i,1]*norm_kp1[i,1],
            norm_kp2[i,1]*norm_kp1[i,2],
            norm_kp2[i,2]*norm_kp1[i,0],
            norm_kp2[i,2]*norm_kp1[i,1],
            norm_kp2[i,2]*norm_kp1[i,2]
        ])
    A = np.array(A)

    U, S, V = svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    F = T2.T @ F @ T1
    F /= F[2,2]
    return F


def sampson_distance(match_kp1, match_kp2, F):
    match_kp1_ext = np.hstack((match_kp1, np.ones((match_kp1.shape[0], 1))))
    match_kp2_ext = np.hstack((match_kp2, np.ones((match_kp2.shape[0], 1))))
    Fx1 = F @ match_kp1_ext.T
    Fx2 = F.T @ match_kp2_ext.T
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    denom = denom.reshape(-1, 1)
    err = np.diag(match_kp2_ext @ F @ match_kp1_ext.T)**2
    err = err.reshape(-1, 1) / denom
    return err


def RANSAC(match_kp1, match_kp2, thres=1, max_iter=10000, conf=0.9):
    num_match = match_kp1.shape[0]
    best_F = None
    best_kp1 = None
    best_kp2 = None
    max_num_inlier = 0

    for i in range(max_iter):
        idx = np.arange(num_match)
        np.random.shuffle(idx)
        sample_kp1 = match_kp1[idx[:8]]
        sample_kp2 = match_kp2[idx[:8]]
        F = run_8_point(sample_kp1, sample_kp2)
        err = sampson_distance(match_kp1, match_kp2, F)
        inlier_idx = err[:,0] < thres
        num_inlier = inlier_idx.astype(np.int32).sum()
        if num_inlier>=max_num_inlier and num_inlier>=num_match*conf:
            print(f'Iter: {i:5d}, {num_inlier} inliers ({num_match}x{conf}={int(num_match*conf)})')
            best_F = np.array(F)
            best_kp1 = match_kp1[inlier_idx]
            best_kp2 = match_kp2[inlier_idx]
            max_num_inlier = num_inlier
    return best_F, best_kp1, best_kp2


def compute_epipolar_line(F, pts):
    pts_ext = np.hstack((pts, np.ones((pts.shape[0], 1))))
    lines = (F @ pts_ext.T).T
    n = np.sqrt(np.sum(lines[:,:2]**2, axis=1)).reshape(-1, 1)
    return lines / n * -1


def drawlines(img1, pts1, img2, pts2, lines):
    w = img1.shape[1]
    new_img1 = np.array(img1)
    new_img2 = np.array(img2)
    for coef, pt1, pt2 in zip(lines, pts1.astype(np.int32), pts2.astype(np.int32)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -coef[2]/coef[1]])
        x1, y1 = map(int, [w, -(coef[2]+coef[0]*w)/coef[1]])
        new_img1 = cv2.line(new_img1, (x0, y0), (x1, y1), color, 1, lineType=4)
        new_img1 = cv2.circle(new_img1, tuple(pt1[0:2]), 5, color, -1)
        new_img2 = cv2.circle(new_img2, tuple(pt2[0:2]), 5, color, -1)
    return new_img1, new_img2


if __name__ == '__main__':
    img1 = cv2.imread('./Mesona1.JPG')
    img2 = cv2.imread('./Mesona2.JPG')

    K1 = np.array([
        [1.4219, 0.0005, 0.5092],
        [0, 1.4219, 0],
        [0, 0, 0.0010]])*1000
    K2 = np.array([
        [1.4219, 0.0005, 0.5092],
        [0, 1.4219, 0.3802],
        [0, 0, 0.0010]])*1000

    # img1 = cv2.imread('./Statue1.bmp')
    # img2 = cv2.imread('./Statue2.bmp')

    # K1 = np.array([[5426.566895, 0.678017, 330.09668],
    #                [0, 5423.133301, 648.950012],
    #                [0, 0, 1]])
    # K2 = np.array([[5426.566895, 0.678017, 387.430023],
    #                [0, 5423.133301, 620.616699],
    #                [0, 0, 1]])

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print('SIFT....')
    kp1, des1 = sift(img1)
    kp2, des2 = sift(img2)

    print('Feature matching....')
    match_kp1, match_kp2 = match_feature(img1, kp1, des1, img2, kp2, des2, 0.5, '.')

    print('RANSAC....')
    F, best_match_kp1, best_match_kp2 = RANSAC(match_kp1, match_kp2)

    print('Draw epipolar lines on image 1....')
    lines_on_img1 = compute_epipolar_line(F.T, best_match_kp2)
    new_img1, new_img2 = drawlines(img1, best_match_kp1, img2, best_match_kp2, lines_on_img1)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(new_img1)
    axs[0].set_title('Epipolar lines on image 1')
    axs[0].axis('off')
    axs[1].imshow(new_img2)
    axs[1].set_title('Interest points on image 2')
    axs[1].axis('off')
    plt.savefig('lines_on_img1.png', dpi=300)

    print('Draw epipolar lines on image 2....')
    lines_on_img2 = compute_epipolar_line(F, best_match_kp1)
    new_img2, new_img1 = drawlines(img2, best_match_kp2, img1, best_match_kp1, lines_on_img2)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(new_img1)
    axs[0].set_title('Interest points on image 1')
    axs[0].axis('off')
    axs[1].imshow(new_img2)
    axs[1].set_title('Epipolar lines on image 2')
    axs[1].axis('off')
    plt.savefig('lines_on_img2.png', dpi=300)
    