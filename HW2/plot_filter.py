# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


if __name__ == '__main__':
    h, w = 300, 300
    ratio = 0.28
    filter_type = 'ideal'
    high_pass = True

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

    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    Z = fltr

    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xticks([], []), ax.set_yticks([], []), ax.set_zticks([], [])
    ax.view_init(15, -45)

    ax = plt.subplot(1, 2, 2)
    im = ax.imshow(fltr, cmap='viridis')
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xticks([], []), ax.set_yticks([], [])

    plt.tight_layout()

    pass_type = 'high' if high_pass else 'low'
    plt.savefig(f'{h}x{w}_ratio{ratio}_{filter_type}_{pass_type}.png', dpi=300)
