# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from numpy.fft import fft2, ifft2, fftshift


def gaussian_filter(filter_size, sigma):
    kernel = np.zeros((filter_size, filter_size))
    center = filter_size // 2
    for i in range(filter_size):
        for j in range(filter_size):
            g = np.exp(-1*((center-i)**2+(center-j)**2)/(2*(sigma**2)))
            g /= 2 * np.pi * (sigma**2)
            kernel[i,j] = g
    kernel /= kernel.sum()
    return kernel


def smooth(img, kernel):
    return ifft2(fft2(img, img.shape)*fft2(kernel, img.shape)).real


def gaussian_pyramid(img, num_layers, kernel):
    res = [np.array(img)]
    for i in range(num_layers):
        img = smooth(img, kernel)
        img = img[::2,::2]
        res.append(np.array(img))
    return res


def magnitude_spectrum(img):
    fshift = fftshift(fft2(img))
    return np.log(np.abs(fshift)+1e-5)


def laplacian_pyramid(g_pyramid, num_layers, kernel):
    res = []
    for i in range(num_layers):
        upsample = g_pyramid[i+1]
        upsample = upsample.repeat(2, axis=0).repeat(2, axis=1)
        if g_pyramid[i].shape[0] % 2:
            upsample = upsample[:-1,:]
        if g_pyramid[i].shape[1] % 2:
            upsample = upsample[:,:-1]
        upsample = smooth(upsample, kernel)
        res.append(g_pyramid[i]-upsample)
    return res


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, 
                        default='./hw2_data/task1,2_hybrid_pyramid/5_submarine.bmp')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--filter_sigma', type=float, default=1.)
    args = parser.parse_args()

    img_name = re.sub(r'\..+', '', args.img_path.split('/')[-1])
    os.makedirs('task2_result', exist_ok=True)

    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)

    kernel = gaussian_filter(args.filter_size, args.filter_sigma)
    g_pyramid = gaussian_pyramid(img, args.num_layers, kernel)
    l_pyramid = laplacian_pyramid(g_pyramid, args.num_layers, kernel)

    plt.suptitle(f'filter size={args.filter_size}x{args.filter_size}, sigma={args.filter_sigma}')
    plt.subplot(4, args.num_layers+1, 1)
    plt.imshow(g_pyramid[0], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
    plt.title('Level 0', fontsize=12)

    text_kwargs = {
        'size': 10,
        'ha': 'right',
        'va': 'center',
        'rotation': 'vertical'
    }
    for i in range(args.num_layers):
        plt.subplot(4, args.num_layers+1, i+2)
        plt.imshow(g_pyramid[i+1], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
        plt.title(f'Level {i+1}', fontsize=12)
        if i == args.num_layers-1:
            plt.gca().text(1.3, 0.5, 'Gaussian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, args.num_layers+1, (args.num_layers+1)+2+i)
        plt.imshow(l_pyramid[i], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
        if i == args.num_layers-1:
            plt.gca().text(1.3, 0.5, 'Laplacian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, args.num_layers+1, (args.num_layers+1)*2+2+i)
        plt.imshow(magnitude_spectrum(g_pyramid[i+1])), plt.xticks([], []), plt.yticks([], [])
        if i == args.num_layers-1:
            plt.gca().text(1.3, 0.5, 'Gaussian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, args.num_layers+1, (args.num_layers+1)*3+2+i)
        plt.imshow(magnitude_spectrum(l_pyramid[i])), plt.xticks([], []), plt.yticks([], [])
        if i == args.num_layers-1:
            plt.gca().text(1.3, 0.5, 'Laplacian', transform=plt.gca().transAxes, **text_kwargs)

    plt.savefig(os.path.join('task2_result', f'{img_name}_pyramid.png'), dpi=300)
