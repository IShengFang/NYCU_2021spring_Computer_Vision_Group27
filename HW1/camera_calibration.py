# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import camera_calibration_show_extrinsics as show


def get_homography(objpoints, imgpoints):
    # 02-camera.pdf p.74
    H_mats = []
    for objpts, imgpts in zip(objpoints, imgpoints):
        # H, mask = cv2.findHomography(cv2.UMat(objpts), cv2.UMat(imgpts))
        P = np.zeros((objpts.shape[0]*2, 9))
        for i, (objpt, imgpt) in enumerate(zip(objpts, imgpts)):
            PT_i = np.array([objpt[0], objpt[1], 1])
            P[i*2,:] = [*(-1*PT_i), 0, 0, 0, *(imgpt[0]*PT_i)]
            P[i*2+1,:] = [0, 0, 0, *(-1*PT_i), *(imgpt[1]*PT_i)]
        U, D, VT = np.linalg.svd(P, full_matrices=False)
        h = VT.T[:,-1]
        h /= h[-1]
        H = h.reshape(3, 3)
        H_mats.append(H)

    return H_mats


def get_intrinsic(H_mats):
    # 02-camera.pdf p.80
    V = np.zeros((2*len(H_mats), 6))
    for i, H in enumerate(H_mats):
        V[2*i,:] = [
            H[0,0]*H[0,1],
            H[1,0]*H[0,1] + H[0,0]*H[1,1],
            H[2,0]*H[0,1] + H[0,0]*H[2,1],
            H[1,0]*H[1,1],
            H[2,0]*H[1,1] + H[1,0]*H[2,1],
            H[2,0]*H[2,1]
        ]
        V[2*i+1,:] = [
            H[0,0]**2 - H[0,1]**2,
            2*H[0,0]*H[1,0] - 2*H[0,1]*H[1,1],
            2*H[0,0]*H[2,0] - 2*H[0,1]*H[2,1],
            H[1,0]**2 - H[1,1]**2,
            2*H[1,0]*H[2,0] - 2*H[1,1]*H[2,1],
            H[2,0]**2 - H[2,1]**2
        ]
    U, D, VT = np.linalg.svd(V, full_matrices=False)
    b = VT.T[:,-1]
    B = np.array([
        [b[0], b[1], b[2]],
        [b[1], b[3], b[4]],
        [b[2], b[4], b[5]]
    ])
    if not np.all(np.linalg.eigvals(B)>0):
        B *= -1

    # 02-camera.pdf p.79
    KT_inv = np.linalg.cholesky(B)
    K_inv = KT_inv.T
    K = np.linalg.inv(KT_inv.T)
    K /= K[2,2]

    return K, K_inv


def get_extrinsic(H_mats, K, K_inv):
    # 02-camera.pdf p.80
    Rt_mats = np.zeros((len(H_mats), 3, 4))
    for i, H in enumerate(H_mats):
        l = 1 / np.linalg.norm(K_inv@H[:,0])
        r1 = l * K_inv @ H[:,0]
        r2 = l * K_inv @ H[:,1]
        r3 = np.cross(r1, r2)
        t = l * K_inv @ H[:,2]

        Rt = np.vstack((r1, r2, r3, t)).T
        Rt_mats[i,:,:] = Rt

    return Rt_mats


def calibration(objpoints, imgpoints):
    H_mats = get_homography(objpoints, imgpoints)
    K, K_inv = get_intrinsic(H_mats)
    Rt_mats = get_extrinsic(H_mats, K, K_inv)
    return K, Rt_mats


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--corner_x', type=int, default=7)
    parser.add_argument('--corner_y', type=int, default=7)
    parser.add_argument('--load_npy', action='store_true', default=False)
    parser.add_argument('--cv2_calibrate', action='store_true', default=False)
    parser.add_argument('--image_dir', type=str, default='data')
    opts = parser.parse_args()

    # prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0) ...., (6, 6, 0)
    # (8, 6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone), 
    # you need to set the corresponding numbers.

    corner_x = opts.corner_x
    corner_y = opts.corner_y
    objp = np.zeros((corner_x*corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

    images = list(set(glob.glob(f'{opts.image_dir}/*.JPG')+glob.glob(f'{opts.image_dir}/*.jpg')))

    # Make a list of calibration images
    if not opts.load_npy:
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        print('Start finding chessboard corners...')
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # plt.imshow(gray)

            # Find the chessboard corners
            print('find the chessboard corners of', fname)
            ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
                #plt.imshow(img)
                folder = fname.split('/')[0]
                os.makedirs(f'plot_corners/{folder}', exist_ok=True)
                #plt.savefig(f'plot_corners/{fname}')
                cv2.imwrite(f'plot_corners/{fname}', img)
            else:
                print('cannot find the chessboard corners of', fname)

        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints).reshape((len(imgpoints), (corner_x*corner_y), 2))
        np.save(f'{opts.image_dir}_objpoints.npy', objpoints)
        np.save(f'{opts.image_dir}_imgpoints.npy', imgpoints)
    else:
        objpoints = np.load(f'{opts.image_dir}_objpoints.npy')
        imgpoints = np.load(f'{opts.image_dir}_imgpoints.npy')

    #######################################################################################################
    #                                Homework 1 Camera Calibration                                        #
    #               You need to implement camera calibration(02-camera p.76-80) here.                     #
    #   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
    #                                          H I N T                                                    #
    #                        1.Use the points in each images to find Hi                                   #
    #                        2.Use Hi to find out the intrinsic matrix K                                  #
    #                        3.Find out the extrensics matrix of each images.                             #
    #######################################################################################################

    print('Camera calibration...')
    # You need to comment these functions and write your calibration function from scratch.
    # Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
    # In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num, 3, 4], and use them to plot.

    if opts.cv2_calibrate:
        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        Vr = np.array(rvecs)
        Tr = np.array(tvecs)
        extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)
    else:
        '''
        Write your code here
        '''
        K, Rt_mats = calibration(objpoints, imgpoints)
        mtx = K
        extrinsics = Rt_mats

    # show the camera extrinsics
    print('Show the camera extrinsics')
    # plot setting
    # You can modify it for better visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # camera setting
    camera_matrix = mtx
    cam_width = 0.064 / 0.1
    cam_height = 0.032 / 0.1
    scale_focal = 1600
    # chess board setting
    board_width = 8
    board_height = 6
    square_size = 1
    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(
        ax, camera_matrix, cam_width, cam_height,
        scale_focal, extrinsics, board_width,
        board_height, square_size, True)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_y-max_range, 0)
    ax.set_zlim(mid_z-max_range, mid_z+max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')
    plt.show()

    # animation for rotating plot
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
