import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import imageio
import matplotlib.pyplot as plt
from skimage.transform import resize

def makeIdealFilter(n_row, n_col, sigma, highPass=True):
    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)
    def ideal(i, j):
        coefficient = math.sqrt((i - center_x)**2 + (j - center_y)**2)
        return 1 - coefficient if highPass else coefficient
    return np.array([[ideal(i, j) for j in range(n_col)] for i in range(n_row)])

def makeGaussianFilter(n_row, n_col, sigma, highPass=True):
    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)
    def gaussian(i, j):
        coefficient = math.exp(-1.0 * ((i - center_x) **2 + (j - center_y)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient
    return np.array([[gaussian(i, j) for j in range(n_col)] for i in range(n_row)])

def filter(image, sigma,flag,isHigh):
    shiftedDFT = fftshift(fft2(image))
    #makeGaussianFilter
    if flag==0:
        filteredDFT = shiftedDFT * makeGaussianFilter(image.shape[0], image.shape[1], sigma, highPass=isHigh)
    #makeIdealFilter  
    if flag==1:
        filteredDFT = shiftedDFT * makeIdealFilter(image.shape[0], image.shape[1], sigma, highPass=isHigh)   
    res = ifft2(ifftshift(filteredDFT))
    return np.real(res)


def hybrid_img(high_img, low_img, sigma_h, sigma_l,flag):
    res = filter(high_img, sigma_h,flag,isHigh=True) + filter(low_img, sigma_l,flag,isHigh=False)
    return res

if __name__=="__main__":
    #input
    img1 = imageio.imread("test01.jpg", as_gray=True)
    img2 = imageio.imread("test02.jpg", as_gray=True)
    resize(img2, (img1.shape[0], img1.shape[1]))
    #Gaussian
    plt.imshow(hybrid_img(img1, img2, 10, 10,0), cmap='gray')
    plt.savefig("hw2_gaussian.jpg")
    #Ideal
    plt.imshow(hybrid_img(img1, img2, 10, 10,1), cmap='gray')
    plt.savefig("hw2_ideal.jpg")

    print ("Finish")
