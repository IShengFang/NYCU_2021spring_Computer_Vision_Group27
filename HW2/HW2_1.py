# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
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
    fs = 10
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    axs = axs.flat
    for ax in axs:
        ax.set_xticks([], []), ax.set_yticks([], [])

    axs[0].imshow(img1)
    axs[0].set_title('Low-pass (origin)', fontsize=fs)

    axs[1].imshow(clip_and_cast(filtered_img1))
    axs[1].set_title('Low-pass (filtered)', fontsize=fs)

    axs[3].imshow(img2)
    axs[3].set_title('Low-pass (origin)', fontsize=fs)

    axs[4].imshow(clip_and_cast(filtered_img2))
    axs[4].set_title('Low-pass (filtered)', fontsize=fs)

    plt.subplot(1, 3, 3)
    hybrid_img = clip_and_cast(filtered_img1+filtered_img2)
    plt.imshow(hybrid_img)
    plt.title('Hybrid'), plt.xticks([], []), plt.yticks([], [])

    plt.savefig(save_path, dpi=300), plt.clf()


def plot_filtering(
        img1, img2, img1_mag, img2_mag,
        filtered_img1, filtered_img2, filtered_img1_mag, filtered_img2_mag,
        save_path):
    color = ['R', 'G', 'B']
    fs = 9

    fig, axs = plt.subplots(4, 4, constrained_layout=True)
    axs = axs.flat
    for ax in axs:
        ax.set_xticks([], []), ax.set_yticks([], [])

    axs[0].imshow(img1)
    axs[0].set_title('Origin (low-pass)', fontsize=fs)
    for i in range(3):
        axs[i+1].imshow(img1_mag[i], cmap='gray')
        axs[i+1].set_title(color[i], fontsize=fs)

    axs[4].imshow(clip_and_cast(filtered_img1))
    axs[4].set_title('Filtered (low-pass)', fontsize=fs)
    for i in range(3):
        axs[4+i+1].imshow(filtered_img1_mag[i], cmap='gray')
        axs[4+i+1].set_title(color[i], fontsize=fs)

    axs[8].imshow(img2)
    axs[8].set_title('Origin (high-pass)', fontsize=fs)
    for i in range(3):
        axs[8+i+1].imshow(img2_mag[i], cmap='gray')
        axs[8+i+1].set_title(color[i], fontsize=fs)

    axs[12].imshow(clip_and_cast(filtered_img2))
    axs[12].set_title('Filtered (high-pass)', fontsize=fs)
    for i in range(3):
        axs[12+i+1].imshow(filtered_img2_mag[i], cmap='gray')
        axs[12+i+1].set_title(color[i], fontsize=fs)

    plt.savefig(save_path, dpi=300), plt.clf()


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--img1_path', type=str, 
                        default='./hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_before.jpg')
    parser.add_argument('--img2_path', type=str, 
                        default='./hw2_data/task1,2_hybrid_pyramid/0_Afghan_girl_after.jpg')
    parser.add_argument('--low_pass_ratio', type=float, default=0.08)
    parser.add_argument('--high_pass_ratio', type=float, default=0.08)
    parser.add_argument('--filter_type', type=str, default='both', help='ideal, gaussian, both')
    parser.add_argument('--result_name', type=str, default=None)
    args = parser.parse_args()
    
    if args.result_name is None:
        args.result_name = re.sub(r'\..+', '', args.img1_path.split('/')[-1]).split('.')[0]

    os.makedirs('task1_result', exist_ok=True)

    img1 = cv2.imread(args.img1_path, cv2.IMREAD_COLOR)[:,:,::-1]
    img2 = cv2.imread(args.img2_path, cv2.IMREAD_COLOR)[:,:,::-1]

    img1, img2 = resize(img1, img2)
    assert img1.shape == img2.shape, f'shape of image1 {img1.shape} != shape of image2 {img2.shape}'
    if args.filter_type=='ideal' or args.filter_type=='both':
        filter_type = 'ideal'
        _, img1_mag, _, filtered_img1_mag, filtered_img1_channel = filtering(img1, args.low_pass_ratio, False, filter_type)
        filtered_img1 = np.stack(filtered_img1_channel, axis=2)
        _, img2_mag, _, filtered_img2_mag, filtered_img2_channel = filtering(img2, args.high_pass_ratio, True, filter_type)
        filtered_img2 = np.stack(filtered_img2_channel, axis=2)

        # before/after hybrid
        save_path = os.path.join('task1_result', f'{args.result_name }_hybrid_ideal_low_{args.low_pass_ratio}_high_{args.high_pass_ratio}.png')
        plot_hybrid(img1, img2, filtered_img1, filtered_img2, save_path)

        # before/after filtering
        save_path = os.path.join('task1_result', f'{args.result_name }_filtering_ideal_low_{args.low_pass_ratio}_high_{args.high_pass_ratio}.png')
        plot_filtering(
            img1, img2, img1_mag, img2_mag,
            filtered_img1, filtered_img2, filtered_img1_mag, filtered_img2_mag,
            save_path)
    if args.filter_type=='gaussian' or args.filter_type=='both':
        filter_type = 'gaussian'
        _, img1_mag, _, filtered_img1_mag, filtered_img1_channel = filtering(img1, args.low_pass_ratio, False, filter_type)
        filtered_img1 = np.stack(filtered_img1_channel, axis=2)
        _, img2_mag, _, filtered_img2_mag, filtered_img2_channel = filtering(img2, args.high_pass_ratio, True, filter_type)
        filtered_img2 = np.stack(filtered_img2_channel, axis=2)

        # before/after hybrid
        save_path = os.path.join('task1_result', f'{args.result_name}_hybrid_gaussian_low_{args.low_pass_ratio}_high_{args.high_pass_ratio}.png')
        plot_hybrid(img1, img2, filtered_img1, filtered_img2, save_path)

        # before/after filtering
        save_path = os.path.join('task1_result', f'{args.result_name}_filtering_gaussian_low_{args.low_pass_ratio}_high_{args.high_pass_ratio}.png')
        plot_filtering(
            img1, img2, img1_mag, img2_mag,
            filtered_img1, filtered_img2, filtered_img1_mag, filtered_img2_mag,
            save_path)