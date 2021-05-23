# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D


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
    print('des1', des1.shape)
    print('des2', des2.shape)
    matches = matching(des1, des2)
    good_match = []
    for m, n in matches:
        if m.distance/n.distance < ratio:
            good_match.append(m)
    plot_match = cv2.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=2)
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
            norm_kp2[i,0] * norm_kp1[i,0],
            norm_kp2[i,0] * norm_kp1[i,1],
            norm_kp2[i,0] * norm_kp1[i,2],
            norm_kp2[i,1] * norm_kp1[i,0],
            norm_kp2[i,1] * norm_kp1[i,1],
            norm_kp2[i,1] * norm_kp1[i,2],
            norm_kp2[i,2] * norm_kp1[i,0],
            norm_kp2[i,2] * norm_kp1[i,1],
            norm_kp2[i,2] * norm_kp1[i,2]
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


def RANSAC(match_kp1, match_kp2, thres=1, max_iter=10000, conf=0.9, quiet=True):
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
            if not quiet:
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


def essential_matrix(K1, K2, F):
    E = K1.T @ F @ K2
    U,S,V = np.linalg.svd(E)
    m = (S[0]+S[1]) / 2
    E = U @ np.diag((m, m, 0)) @ V
    return E


def four_possible_solution_of_second_camera_matrix(E):
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(U@V) < 0:
        V = -V
    W = np.array([[0,-1, 0],
                  [1, 0, 0], 
                  [0, 0, 1]])
    R = U @ W @ V
    t = U[:,2:]
    P2_0 = np.hstack((R, t))
    P2_1 = np.hstack((R, -t))
    R = U @ W.T @ V
    P2_2 = np.hstack((R, t))
    P2_3 = np.hstack((R, -t))

    return [P2_0, P2_1, P2_2, P2_3]


def triangulation(x1, x2, P1, P2):
    pred_pt = np.ones((x1.shape[0], 4))
    C = P2[:,:3].T @ P2[:,3:]
    infront_num = 0
    for i in range(x1.shape[0]):
        A = np.asarray([
            (x1[i,0] * P1[2,:].T - P1[0,:].T),
            (x1[i,1] * P1[2,:].T - P1[1,:].T),
            (x2[i,0] * P2[2,:].T - P2[0,:].T),
            (x2[i,1] * P2[2,:].T - P2[1,:].T)
        ])
        U, S, V = np.linalg.svd(A)
        pred_pt_i = V[-1] / V[-1,3]
        pred_pt[i,:] = pred_pt_i
        if np.dot((pred_pt_i[:3]-C.reshape(-1)), P2[2,:3]) > 0:
            infront_num += 1
    return pred_pt, infront_num


def plot_pred_points(pred_pts):
    plt.clf()
    fig = plt.figure()
    ax = Axes3D(fig)
    # for i in range(pred_pts.shape[0]):
    #     ax.scatter(pred_pts[i,0], pred_pts[i,1], pred_pts[i,2], c='r', marker='o')
    ax.scatter(pred_pts[:,0], pred_pts[:,1], pred_pts[:,2], c='r', marker='o')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    return fig


def use_matlab_for_final_results(pred_pts, keypoints, P, img_name, output_dir):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.obj_main(
        matlab.double(pred_pts[:,:3].tolist()),
        matlab.double(keypoints.tolist()),
        matlab.double(P.tolist()),
        './{}'.format(img_name),
        1.0,
        output_dir,
        nargout=0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--image_set', type=str, default='mesona', help='mesona, statue, ours')
    parser.add_argument('--match_ratio', type=float, default=0.5)
    parser.add_argument('--ransac_conf', type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.image_set == 'mesona':
        img1_name = 'Mesona1.JPG'
        img2_name = 'Mesona2.JPG'
        K1 = np.array([[1.4219, 0.0005, 0.5092],
                      [     0, 1.4219,      0],
                      [     0,      0, 0.0010]])
        K2 = np.array([[1.4219, 0.0005, 0.5092],
                      [     0, 1.4219, 0.3802],
                      [     0,      0, 0.0010]])
    elif args.image_set == 'statue':
        img1_name = './Statue1.bmp'
        img2_name = './Statue2.bmp'
        K1 = np.array([[5426.566895,    0.678017, 330.096680],
                       [          0, 5423.133301, 648.950012],
                       [          0,           0,          1]])
        K2 = np.array([[5426.566895,    0.678017, 387.430023],
                       [          0, 5423.133301, 620.616699],
                       [          0,           0,          1]])
    elif args.image_set == 'ours':
        img1_name = './BOX_0.jpg'
        img2_name = './BOX_1.jpg'
        K1 = np.load('intrinsic.npy')
        K2 = np.load('intrinsic.npy')
        # [[1.52327854e+03 0.00000000e+00 7.55865517e+02]
        #  [0.00000000e+00 1.52097830e+03 9.90642289e+02]
        #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    else:
        print('Unknown image set')
        raise NotImplementedError

    img1 = cv2.imread(f'./{img1_name}')
    img2 = cv2.imread(f'./{img2_name}')
    print(f'Image 1: {img1_name}')
    print(f'Image 2: {img2_name}')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print('SIFT....')
    kp1, des1 = sift(img1)
    kp2, des2 = sift(img2)
    img1_kp = cv2.drawKeypoints(img1, kp1, None)
    img2_kp = cv2.drawKeypoints(img2, kp2, None)
    plt.imsave(os.path.join(args.output_dir, 'img1_kp.png'), img1_kp)
    plt.imsave(os.path.join(args.output_dir, 'img2_kp.png'), img2_kp)

    print('Feature matching....')
    match_kp1, match_kp2 = match_feature(img1, kp1, des1, img2, kp2, des2, args.match_ratio, args.output_dir)
    print('match_kp1', match_kp1.shape)
    print('match_kp2', match_kp2.shape)

    print('RANSAC....')
    F, best_match_kp1, best_match_kp2 = RANSAC(match_kp1, match_kp2, conf=args.ransac_conf, quiet=False)

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
    plt.savefig(os.path.join(args.output_dir, 'epipolar-lines_on_img1.png'), dpi=300)

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
    plt.savefig(os.path.join(args.output_dir, 'epipolar-lines_on_img2.png'), dpi=300)
    
    print('Get 4 possible solutions of essential matrix from fundamental matrix....')
    E = essential_matrix(K1, K2, F)
    P2s = four_possible_solution_of_second_camera_matrix(E)

    print('Find out the most appropriate solution of essential matrix....')
    P1 = K1 @ np.eye(3, 4)
    largest_infornt_num = 0
    for P2 in P2s:
        P2 = K2 @ P2
        pred_pt, infront_num = triangulation(best_match_kp1, best_match_kp2, P1, P2)
        # C = P2[:,:3] @ P2[:,3].T
        if infront_num > largest_infornt_num:
            largest_infornt_num = infront_num
            most_apprx_P2 = P2
            most_apprx_pred_pt = pred_pt

    print('Apply triangulation to get 3D points....')
    fig = plot_pred_points(most_apprx_pred_pt)
    fig.savefig(os.path.join(args.output_dir, 'pred_points.png'), dpi=300)
    use_matlab_for_final_results(most_apprx_pred_pt, best_match_kp1, P1, img1_name, args.output_dir)
    os.rename('model1.mtl', os.path.join(args.output_dir, 'model1.mtl'))
    os.rename('model1.obj', os.path.join(args.output_dir, 'model1.obj'))
