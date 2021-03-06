# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from HW2_2 import gaussian_filter, gaussian_pyramid


def remove_white_border(img):
    white = 215
    while np.all(img[0,:]>=white):
        img = img[1:,:]
    while np.all(img[-1,:]>=white):
        img = img[:-1,:]
    while np.all(img[:,0]>=white):
        img = img[:,1:]
    while np.all(img[:,-1]>=white):
        img = img[:,:-1]
    return img


def remove_black_border(img):
    h, w = img.shape
    new_h, new_w = int(0.93*h), int(0.93*w)
    return img[h-new_h:new_h,w-new_w:new_w]


def euclidean(img1, img2):
    return np.sqrt(np.sum((img1-img2)**2))


def manhattan(img1, img2):
    return np.sum(np.abs(img1-img2))


def zncc(img1, img2):
    mu_img1 = np.mean(img1)
    mu_img2 = np.mean(img2)
    std_img1 = np.std(img1)
    std_img2 = np.std(img2)
    return -1 * ((img1-mu_img1)*(img2-mu_img2)/(std_img1*std_img2)).mean()


def ncc(img1, img2):
    return -1 * ((img1/np.std(img1))*(img2/np.std(img2))).mean()


def ssim(img1, img2):
    c1 = 0.01**2
    c2 = 0.03**2
    mu_img1 = np.mean(img1)
    mu_img2 = np.mean(img2)
    std_img1 = np.sqrt(np.mean(img1**2)-mu_img1**2)
    std_img2 = np.sqrt(np.mean(img2**2)-mu_img2**2)
    cov = np.mean(img1*img2) - mu_img1*mu_img2
    numerator = (2*mu_img1*mu_img2+c1) * (2*cov+c1)
    denominator = (mu_img1**2+mu_img2**2+c1) * (std_img1**2+std_img2**2+c2)
    return -1 * numerator / denominator


def shift(img, displ):
    shifted_img = np.roll(img, displ, axis=(0, 1))
    # if displ[0] > 0:
    #     shifted_img[:displ[0],:] = 0
    # elif displ[0] < 0:
    #     shifted_img[displ[0]:,:] = 0
    # if displ[1] > 0:
    #     shifted_img[:,:displ[1]] = 0
    # elif displ[1] < 0:
    #     shifted_img[:,displ[1]:] = 0
    return shifted_img


def align(img1, img2, displ, search_range, measure):
    min_diff = float('inf')
    img1 = img1 / 255
    img2 = img2 / 255
    best_displ = [0, 0]
    for i in range(displ[0]-search_range, displ[0]+search_range):
        for j in range(displ[1]-search_range, displ[1]+search_range):
            shifted_img2 = shift(img2, [i, j])
            diff = measure(img1, shifted_img2)
            if diff < min_diff:
                min_diff = diff
                best_displ = [i, j]
    return best_displ


def colorize(
        save_dir, img_name,
        r, g, b, base_channel,
        pyramid_layer, filter_size, filter_sigma,
        measure):
    CH1, CH2, BASE = 0, 1, 2
    if pyramid_layer == -1:
        if base_channel == 'r':
            channel = [g, b, r]
        elif base_channel == 'g':
            channel = [r, b, g]
        elif base_channel == 'b':
            channel = [r, g, b]
        else:
            print('Unknown channel')
            raise NotImplementedError

        displ = [
            align(channel[BASE], channel[CH1], [0, 0], 15, measure),
            align(channel[BASE], channel[CH2], [0, 0], 15, measure)
        ]
        shifted = [
            shift(channel[CH1], displ[CH1]),
            shift(channel[CH2], displ[CH2]),
        ]
    else:
        kernel = gaussian_filter(filter_size, filter_sigma)
        g_pyramid_r = gaussian_pyramid(r, pyramid_layer, kernel)[::-1]
        g_pyramid_g = gaussian_pyramid(g, pyramid_layer, kernel)[::-1]
        g_pyramid_b = gaussian_pyramid(b, pyramid_layer, kernel)[::-1]

        # show gaussian pyramid of each channel
        text_kwargs = {
            'size': 18,
            'ha': 'left',
            'va': 'center',
        }
        for i in range(pyramid_layer):
            plt.subplot(3, pyramid_layer, i+1)
            plt.imshow(g_pyramid_r[i], cmap='gray')
            plt.title(f'Level {pyramid_layer-i-1}')
            plt.xticks([]), plt.yticks([])
            if i == 0:
                plt.gca().text(-0.4, 0.5, 'R', transform=plt.gca().transAxes, **text_kwargs)

            plt.subplot(3, pyramid_layer, pyramid_layer+i+1)
            plt.imshow(g_pyramid_g[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
            if i == 0:
                plt.gca().text(-0.4, 0.5, 'G', transform=plt.gca().transAxes, **text_kwargs)

            plt.subplot(3, pyramid_layer, pyramid_layer*2+i+1)
            plt.imshow(g_pyramid_b[i], cmap='gray')
            plt.xticks([]), plt.yticks([])
            if i == 0:
                plt.gca().text(-0.4, 0.5, 'B', transform=plt.gca().transAxes, **text_kwargs)
        plt.savefig(os.path.join(save_dir, f'{img_name}_pyramid_pyd_{pyramid_layer}_size_{filter_size}_sigma_{filter_sigma}.png'), dpi=200), plt.clf()

        if base_channel == 'r':
            channel = [g, b, r]
            g_pyramid = [g_pyramid_g, g_pyramid_b, g_pyramid_r]
        elif base_channel == 'g':
            channel = [r, b, g]
            g_pyramid = [g_pyramid_r, g_pyramid_b, g_pyramid_g]
        elif base_channel == 'b':
            channel = [r, g, b]
            g_pyramid = [g_pyramid_r, g_pyramid_g, g_pyramid_b]
        else:
            print('Unknown channel')
            raise NotImplementedError

        displ = [
            align(g_pyramid[BASE][0], g_pyramid[CH1][0], [0, 0], 15, measure),
            align(g_pyramid[BASE][0], g_pyramid[CH2][0], [0, 0], 15, measure)
        ]
        print(f'Aligning level {pyramid_layer}')
        for i in range(1, len(g_pyramid[BASE])):
            displ[CH1] = [d*2 for d in displ[CH1]]
            displ[CH2] = [d*2 for d in displ[CH2]]
            displ[CH1] = align(g_pyramid[BASE][i], g_pyramid[CH1][i], displ[CH1], 5, measure)
            displ[CH2] = align(g_pyramid[BASE][i], g_pyramid[CH2][i], displ[CH2], 5, measure)
            print(f'Aligning level {pyramid_layer-i}')
        shifted = [
            shift(channel[CH1], displ[CH1]),
            shift(channel[CH2], displ[CH2]),
        ]

    if base_channel == 'r':
        result = [r, shifted[CH1], shifted[CH2]]
        print('Best displacement of G', displ[CH1])
        print('Best displacement of B', displ[CH2])
    elif base_channel == 'g':
        result = [shifted[CH1], g, shifted[CH2]]
        print('Best displacement of R', displ[CH1])
        print('Best displacement of B', displ[CH2])
    elif base_channel == 'b':
        result = [shifted[CH1], shifted[CH2], b]
        print('Best displacement of R', displ[CH1])
        print('Best displacement of G', displ[CH2])

    return np.stack(result[::-1], axis=2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, 
                        default='./hw2_data/task3_colorizing/cathedral.jpg')
    parser.add_argument('--base_channel', type=str, default='g', help='r, g, b')
    parser.add_argument('--measure', type=str, default='euclidean', help='ssim, euclidean, manhattan, ncc, zncc')
    parser.add_argument('--pyramid_layer', type=int, default=5)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--filter_sigma', type=float, default=1.)    
    args = parser.parse_args()

    base_channel = args.base_channel
    # -1 to disable pyramid aligning
    pyramid_layer = args.pyramid_layer
    filter_size = args.filter_size
    filter_sigma = args.filter_sigma
    if args.measure not in ['ssim', 'euclidean', 'manhattan', 'ncc', 'zncc']:
        print('Unknown measurement')
        raise NotImplementedError
    else:
        measure = vars()[args.measure]

    img_name = re.sub(r'\..+', '', args.img_path.split('/')[-1])
    save_dir = os.path.join('task3_result', img_name)
    os.makedirs(save_dir, exist_ok=True)

    print(args.img_path)
    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    print(f'Image: {img_name} {img.shape}')

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.title('Origin')

    img = remove_white_border(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.title('Remove\nwhite border')

    h, w = img.shape
    h = h // 3
    b = remove_black_border(img[:h,:])
    g = remove_black_border(img[h:h*2,:])
    r = remove_black_border(img[h*2:h*3,:])

    plt.subplot(3, 3, 3)
    plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.title('Remove\ndark border')
    plt.subplot(3, 3, 6)
    plt.imshow(g, cmap='gray'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 9)
    plt.imshow(r, cmap='gray'), plt.xticks([]), plt.yticks([])

    plt.savefig(os.path.join(save_dir, f'{img_name}_remove-border.png'), dpi=200), plt.clf()

    # without alignment
    cv2.imwrite(os.path.join(save_dir, f'{img_name}_no-align.png'), np.stack((b, g, r), axis=2))

    # with alignment
    result = colorize(
        save_dir, img_name,
        r, g, b, base_channel,
        pyramid_layer, filter_size, filter_sigma,
        measure)
    cv2.imwrite(os.path.join(save_dir, f'{img_name}_base_{args.base_channel}_align_pyd_{args.pyramid_layer}_size_{args.filter_size}_sigma_{args.filter_sigma}_{args.measure}.png'), result)
