# -*- coding: utf-8 -*-
import cv2
import glob
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--corner_x', type=int, default=7)
    parser.add_argument('--corner_y', type=int, default=7)
    parser.add_argument('--image_dir', type=str, default='data')
    args = parser.parse_args()

    corner_x = args.corner_x
    corner_y = args.corner_y
    objp = np.zeros((corner_x*corner_y, 3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1, 2)

    images = list(glob.glob(f'{args.image_dir}/*.jpg'))

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    print('Start finding chessboard corners...')
    pbar_images = tqdm(images)
    found, not_found = 0, 0
    for idx, fname in enumerate(pbar_images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            found += 1
        else:
            not_found += 1
        pbar_images.set_description(f'{found} found/{not_found} not found')

    objpoints = np.array(objpoints)
    imgpoints = np.array(imgpoints).reshape((len(imgpoints), (corner_x*corner_y), 2))
    # np.save(f'objpoints.npy', objpoints)
    # np.save(f'imgpoints.npy', imgpoints)

    print('Camera calibration...')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    Vr = np.array(rvecs)
    Tr = np.array(tvecs)
    extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)
    print(mtx)
    np.save('intrinsic.npy', mtx)
