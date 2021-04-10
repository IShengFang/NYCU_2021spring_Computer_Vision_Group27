# -*- coding: utf-8 -*-
import math
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def decide_size(image, div):
    if image.shape[0]%2 == 0:
        center_x = image.shape[0] / div
    if image.shape[0]%2 == 1:
        center_x = image.shape[0]/div + 1
    if image.shape[1]%2 == 0:
        center_y = image.shape[1]/div
    if image.shape[1]%2 == 1: 
        center_y = image.shape[1]/div + 1
    return center_x, center_y


def gaussianfilter(image, sigma, highlow=True):
    def gaussian(i, j):
        entry = math.exp(-1.0*((i-center_x)**2+(j-center_y)**2)/(2*sigma**2))
        if highlow == True:
            return entry
        if highlow == False:
            return 1 - entry
    center_x, center_y = decide_size(image, 2)
    shift_DFT = fftshift(fft2(image))
    filter_DFT = shift_DFT * (np.array([[gaussian(i, j) for j in range(image.shape[1])] for i in range(image.shape[0])]))
    return np.real(ifft2(ifftshift(filter_DFT)))


def idealfilter(image, sigma, highlow=True):
    def ideal(i, j):
        entry = math.sqrt((i-center_x)**2+(j-center_y)**2)
        if highlow == True:
            return entry
        if highlow == False:
            return 1 - entry
    center_x, center_y = decide_size(image, 2)
    shift_DFT = fftshift(fft2(image))
    filter_DFT = shift_DFT * (np.array([[ideal(i, j) for j in range(image.shape[1])] for i in range(image.shape[0])]))
    return np.real(ifft2(ifftshift(filter_DFT)))


def hybrid(image1, image2, highsigma, lowsigma, filtertype):
    if filtertype == 0:
        return gaussianfilter(image1, highsigma, highlow=True) + gaussianfilter(image2, lowsigma, highlow=False)
    if filtertype == 1:
        return idealfilter(image1, highsigma, highlow=True) + idealfilter(image2, lowsigma, highlow=False)


def input():
    image1 = imageio.imread('test02.jpg', as_gray=True)
    image2 = imageio.imread('test01.jpg', as_gray=True)
    resize(image2, (image1.shape[0], image1.shape[1]))
    return image1, image2


def output(image, h_sigma, l_sigma, filtertype):
    plt.imshow(image , cmap='gray')
    if filtertype == 0:
        title = 'Gaussion => ' 'high_sigma : ', h_sigma, 'low_sigma : ', l_sigma
        plt.title(title, fontsize=12)
        plt.savefig('hw2_gaussian.jpg')
    if filtertype == 1:
        title = 'Ideal => ' 'high_sigma : ', h_sigma, 'low_sigma : ', l_sigma
        plt.title(title, fontsize=12)
        plt.savefig('hw2_ideal.jpg')     


if __name__=='__main__':
    # Input
    image1 , image2 = input()

    # Gaussian
    gaussian_high_sigma = 10
    gaussian_low_sigma = 10

    result = hybrid(image1, image2, gaussian_high_sigma, gaussian_low_sigma, 0)
    output(result, gaussian_high_sigma, gaussian_low_sigma, 0)

    # Ideal
    ideal_high_sigma = 10
    ideal_low_sigma = 10

    result = hybrid(image1, image2, ideal_high_sigma, ideal_low_sigma, 1)
    output(result, ideal_high_sigma, ideal_low_sigma, 1)

    print('finish')
