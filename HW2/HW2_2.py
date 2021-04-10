# -*- coding: utf-8 -*-
import os
import re
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2, fftshift


# def gaussian_filter(x, sigma):
#     i = np.arange(x)
#     c = np.ones(x) * (x//2)
#     c_i_2 = ((c-i)**2).reshape(-1, 1).repeat(x, axis=1)
#     c_j_2 = ((c-i)**2).reshape(1, -1).repeat(x, axis=0)
#     g1 = np.exp(-1*c_i_2/(2*(sigma**2)))
#     g2 = np.exp(-1*c_j_2/(2*(sigma**2)))
#     kernel = g1 * g2 / (2*np.pi*(sigma**2))
#     return kernel


def gaussian_filter(x, sigma):
    kernel = np.zeros((x, x))
    center = x // 2
    for i in range(x):
        for j in range(x):
            g = np.exp(-1*((center-i)**2+(center-j)**2)/(2*(sigma**2)))
            g /= 2 * np.pi * (sigma**2)
            kernel[i,j] = g
    kernel /= kernel.sum()
    return kernel


# def smooth(img, kernel):
#     h, w = img.shape
#     num = kernel.shape[0] // 2
#     img = np.pad(img, ((num, num), (num, num)), 'edge')
#     result = np.zeros((h, w))
#     for i in range(h):
#         for j in range(w):
#             result[i,j] = np.sum(img[i:i+(2*num)+1,j:j+(2*num)+1]*kernel)
#     return result


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
    img_path = './hw2_data/task1,2_hybrid_pyramid/4_marilyn.bmp'
    num_layers = 5
    filter_size = 5
    filter_sigma = 1.0

    img_name = re.sub(r'\..+', '', img_path.split('/')[-1])
    os.makedirs('task2_result', exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kernel = gaussian_filter(filter_size, filter_sigma)
    g_pyramid = gaussian_pyramid(img, num_layers, kernel)
    l_pyramid = laplacian_pyramid(g_pyramid, num_layers, kernel)

    plt.suptitle(f'filter size={filter_size}x{filter_size}, sigma={filter_sigma}')
    plt.subplot(4, num_layers+1, 1)
    plt.imshow(g_pyramid[0], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
    plt.title('Level 0', fontsize=12)

    text_kwargs = {
        'size': 10,
        'ha': 'right',
        'va': 'center',
        'rotation': 'vertical'
    }
    for i in range(num_layers):
        plt.subplot(4, num_layers+1, i+2)
        plt.imshow(g_pyramid[i+1], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
        plt.title(f'Level {i+1}', fontsize=12)
        if i == num_layers-1:
            plt.gca().text(1.3, 0.5, 'Gaussian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, num_layers+1, (num_layers+1)+2+i)
        plt.imshow(l_pyramid[i], cmap='gray'), plt.xticks([], []), plt.yticks([], [])
        if i == num_layers-1:
            plt.gca().text(1.3, 0.5, 'Laplacian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, num_layers+1, (num_layers+1)*2+2+i)
        plt.imshow(magnitude_spectrum(g_pyramid[i+1])), plt.xticks([], []), plt.yticks([], [])
        if i == num_layers-1:
            plt.gca().text(1.3, 0.5, 'Gaussian', transform=plt.gca().transAxes, **text_kwargs)

        plt.subplot(4, num_layers+1, (num_layers+1)*3+2+i)
        plt.imshow(magnitude_spectrum(l_pyramid[i])), plt.xticks([], []), plt.yticks([], [])
        if i == num_layers-1:
            plt.gca().text(1.3, 0.5, 'Laplacian', transform=plt.gca().transAxes, **text_kwargs)

    plt.savefig(os.path.join('task2_result', f'{img_name}_pyramid.png'), dpi=300)
