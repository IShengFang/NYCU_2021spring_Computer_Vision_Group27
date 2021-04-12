# Homework 2: Hybrid image, Image pyramid, Colorizing the Russian Empire

## Submission

- Deadline: 2021/4/15 23:55:00
- The report should include:
    - your introduction
    - implementation procedure
    - experimental results (of course you should also try your own images)
    - discussion (what difficulties you have met? how you resolve them?)
    - conclusion
    - work assignment plan between team members

## Task 1: Hybrid image

### Overview

A hybrid image is the sum of a low-pass filtered version of the one image and a high-pass filtered version of a second image. There is a free parameter, which can be  tuned for each image pair, which controls how much high frequency to remove from the first image and how much low frequency to leave in the second image. This is called the "cutoff frequency".

### hybrid.py

- Change the following variables for different experiments
    - image paths in `input()` function
    - `gaussian_high_sigma`
    - `gaussian_low_sigma`
    - `ideal_high_sigma`
    - `ideal_low_sigma`

### HW2_1.py

- Change the following variables for different experiments
    - `img1_path`: str
    - `img2_path`: str
    - `filter_type`: str (ideal, gaussian)
    - `low_pass_ratio`: float (0~1)
    - `high_pass_ratio`: float (0~1)

### exprimeent
- NYMU, NCTU
    - 校徽跟文字 
- 人物的照片？
    -  彩色人像

## Task 2: Image pyramid

### Overview

An image pyramid is a collection of representations of an image.

### HW2_2.py

- Change the following variables for different experiments
    - `img_path`: str
    - `num_layers`: int
    - `filter_size`: int (odd number)
    - `filter_sigma`: float (0~1)

#### experiment 
    - `filter_size`: int (odd number)
    - `filter_sigma`: float (0~1)

## Task 3: Colorizing the Russian Empire

### Overview

Goal: automatically produce a color image from the digitized Prokudin-Gorskii glass plate images with as few visual artifacts as possible.The glass plate images record three exposures of every scene onto a glass plate using a red, a green, and a blue filter. In order to do this, extract the three color channels from the glass plate, then place and align one above the other so that the combination forms a single RGB color image Assume that a simple x,y translation model is sufficient for proper alignment. However,  the full-size glass plate images are very large,  the alignment procedure will need to be fast and efficient.

### HW2_3.py

- Change the following variables for different experiments
    - `img_path`: str
    - `base_channel`: str, (r, g, b)
    - `pyramid_layer`: int (>=5)
    - `measure`: (ssim, euclidean, manhattan, ncc)

### Experiment 
- speed
    - pyramind or not
- pyaamind layer num 4~6
- measure 4
