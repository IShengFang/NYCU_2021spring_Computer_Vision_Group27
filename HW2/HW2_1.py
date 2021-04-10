# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def filtering(img, ratio, high_pass, filter_type):
    h, w, _ = img.shape
    fltr = np.zeros((h, w))
    cutoff_freq = min(h, w) / 2 * ratio
    cutoff_freq = int(cutoff_freq)

    center_h, center_w = h//2, w//2
    for i in range(h):
        for j in range(w):
            u, v = i-center_h, j-center_w
            if filter_type == 'gaussian':
                fltr[i,j] = np.exp(-(u**2+v**2)/(2*cutoff_freq**2))
            elif filter_type == 'ideal':
                fltr[i,j] = 1 if u**2+v**2<=cutoff_freq**2 else 0
            else:
                print('Unknown filter type')
                raise NotImplementedError

    if high_pass:
        fltr = 1 - fltr

    freq = [fftshift(fft2(img[:,:,c])) for c in range(img.shape[2])]
    mag = [np.log(np.abs(f)+1e-5) for f in freq]
    filtered_freq = [fftshift(fft2(img[:,:,c]))*fltr for c in range(img.shape[2])]
    filtered_mag = [np.log(np.abs(f)+1e-5) for f in filtered_freq]
    filtered_channel = [ifft2(ifftshift(f)).real for f in filtered_freq]

    return freq, mag, filtered_freq, filtered_mag, filtered_channel


def resize(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    if h1 > h2:
        left = (h1-h2) // 2
        img1 = img1[left:left+h2,:,:]
    elif h1 < h2:
        left = (h2-h1) // 2
        img2 = img2[left:left+h1,:,:]
    if w1 > w2:
        up = (w1-w2) // 2
        img1 = img1[:,up:up+w2,:]
    elif w1 < w2:
        up = (w2-w1) // 2
        img2 = img2[:,up:up+w1,:]
    return img1, img2


def clip_and_cast(array):
    return array.clip(0, 255).astype(np.uint8)


def plot_hybrid(img1, img2, filtered_img1, filtered_img2, save_path):
    plt.subplot(2, 3, 1)
    plt.imshow(img1)
    plt.title('Low-pass (origin)'), plt.xticks([], []), plt.yticks([], [])

    plt.subplot(2, 3, 2)
    plt.imshow(clip_and_cast(filtered_img1))
    plt.title('Low-pass (filtered)'), plt.xticks([], []), plt.yticks([], [])

    plt.subplot(2, 3, 4)
    plt.imshow(img2)
    plt.title('High-pass (origin)'), plt.xticks([], []), plt.yticks([], [])

    plt.subplot(2, 3, 5)
    plt.imshow(clip_and_cast(filtered_img2))
    plt.title('High-pass (filtered)'), plt.xticks([], []), plt.yticks([], [])

    plt.subplot(1, 3, 3)
    hybrid_img = clip_and_cast(filtered_img1+filtered_img2)
    plt.imshow(hybrid_img)
    plt.title('Hybrid'), plt.xticks([], []), plt.yticks([], [])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300), plt.clf()


def plot_filtering(
        img1, img2, img1_mag, img2_mag,
        filtered_img1, filtered_img2, filtered_img1_mag, filtered_img2_mag,
        save_path):
    color = ['R', 'G', 'B']
    fs = 9

    plt.subplot(4, 4, 1)
    plt.imshow(img1)
    plt.title('Origin (low-pass)', fontsize=fs)
    plt.xticks([], []), plt.yticks([], [])
    for i in range(3):
        plt.subplot(4, 4, i+2)
        plt.imshow(img1_mag[i], cmap='gray')
        plt.title(color[i], fontsize=fs)
        plt.xticks([], []), plt.yticks([], [])

    plt.subplot(4, 4, 4+1)
    plt.imshow(clip_and_cast(filtered_img1))
    plt.title('Filtered (low-pass)', fontsize=fs)
    plt.xticks([], []), plt.yticks([], [])
    for i in range(3):
        plt.subplot(4, 4, 4+i+2)
        plt.imshow(filtered_img1_mag[i], cmap='gray')
        plt.title(color[i], fontsize=fs)
        plt.xticks([], []), plt.yticks([], [])

    plt.subplot(4, 4, 8+1)
    plt.imshow(img2)
    plt.title('Origin (high-pass)', fontsize=fs)
    plt.xticks([], []), plt.yticks([], [])
    for i in range(3):
        plt.subplot(4, 4, 8+i+2)
        plt.imshow(img2_mag[i], cmap='gray')
        plt.title(color[i], fontsize=fs)
        plt.xticks([], []), plt.yticks([], [])

    plt.subplot(4, 4, 12+1)
    plt.imshow(clip_and_cast(filtered_img2))
    plt.title('Filtered (high-pass)', fontsize=fs)
    plt.xticks([], []), plt.yticks([], [])
    for i in range(3):
        plt.subplot(4, 4, 12+i+2)
        plt.imshow(filtered_img2_mag[i], cmap='gray')
        plt.title(color[i], fontsize=fs)
        plt.xticks([], []), plt.yticks([], [])

    plt.savefig(save_path, dpi=300), plt.clf()


if __name__=='__main__':
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_before.jpg'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_after.jpg'
    img1_path = './hw2_data/task1,2_hybrid_pyramid/1_bicycle.bmp'
    img2_path = './hw2_data/task1,2_hybrid_pyramid/1_motorcycle.bmp'
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/2_bird.bmp'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/2_plane.bmp'
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/3_cat.bmp'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/3_dog.bmp'
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/4_einstein.bmp'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/4_marilyn.bmp'
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/5_fish.bmp'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/5_submarine.bmp'
    # img1_path = './hw2_data/task1,2_hybrid_pyramid/6_makeup_before.jpg'
    # img2_path = './hw2_data/task1,2_hybrid_pyramid/6_makeup_after.jpg'
    # ideal, gaussian
    filter_type = 'ideal'
    low_pass_ratio = 0.04
    high_pass_ratio = 0.05

    set_idx = re.sub(r'\..+', '', img1_path.split('/')[-1]).split('_')[0]
    os.makedirs('task1_result', exist_ok=True)

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)[:,:,::-1]
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)[:,:,::-1]

    img1, img2 = resize(img1, img2)
    assert img1.shape == img2.shape, f'shape of image1 {img1.shape} != shape of image2 {img2.shape}'

    _, img1_mag, _, filtered_img1_mag, filtered_img1_channel = filtering(img1, low_pass_ratio, False, filter_type)
    filtered_img1 = np.stack(filtered_img1_channel, axis=2)
    _, img2_mag, _, filtered_img2_mag, filtered_img2_channel = filtering(img2, high_pass_ratio, True, filter_type)
    filtered_img2 = np.stack(filtered_img2_channel, axis=2)

    # before/after hybrid
    save_path = os.path.join('task1_result', f'{set_idx}_hybrid.png')
    plot_hybrid(img1, img2, filtered_img1, filtered_img2, save_path)

    # before/after filtering
    save_path = os.path.join('task1_result', f'{set_idx}_filtering.png')
    plot_filtering(
        img1, img2, img1_mag, img2_mag,
        filtered_img1, filtered_img2, filtered_img1_mag, filtered_img2_mag,
        save_path)
