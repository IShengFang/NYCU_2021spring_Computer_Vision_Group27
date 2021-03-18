# Homework 1: 2D Camera Calibration

## Assignment describe
- In this assignment, you will practice how to implement camera calibration.
- For implement details, please refer to the slides 02-camera p.76-80.
- We will provide an example code, you need to revise it by your calibration function.
- **DO NOT** use the `cv2.calibrateCamera` or other calibration functions, you need to implement it from scratch.

- In example code (camera_calibration.py), the code of loading data is provided. ✔ command: python camera_calibration.py
Camera calibration:
  - First, figure out the Hi of each images.
  - Use Hi to find B, and calculate intrinsic matrix K from B by using Cholesky factorization.
  - Then, get extrinsic matrix [R|t] for each images by K and H (p.79, 80).
- After you find out the intrinsic matrix and extrinsic matrixes, plot it like p.86 result.
  - Plot code is given, you only need to feed the data in.
- For mathematic details, please refer to slides 02-camera p.76-80.

- Two types of data you should try: 
  - images we provided in data folder
  - images captured by your smartphone
    - We have provided the chessboard image, print it out and take photo with it.
    - NOTICE that you should close the AF(auto focus) function of your camera, and set a fix focus.
    - If you don’t know how to fix focus of your camera, please google it or ask TAs.

### Submission
- Deadline: 2021/3/22 23:55:00
- Hand in your report and code on New E3.
- The report should include:
  - your introduction
  - implementation procedure
  - experimental result (of course you should also try your own images)
  - discussion
  - conclusion
  - work assignment plan between team members

## To do
- [x] Code
  - Eugene Yang (NO WOOD:rage::cursing_face::cursing_face::+1::clap::+1::clap:
  - [x] Use `cv2.findHomography` or not -> use our own
- [x] Collect our own data
  - AF w/ 11 pro
  - MF w/ 11 pro
  - (t)ele, (w)ide, (u)ltra wide
- [x] Paper
  - LaTeX on Overleaf and back up in github?  
  - [x] Introduction
    - Jacky
  - [x] Implementation procedure
    - Eugene
  - [x] Experimental result
    - Ethan
  - [x] Discussion
    - Ethan
  - [x] Conclusion
    - Jacky

---

- [x] CPT 1: 3/17
- [x] FINISH: 3/18
