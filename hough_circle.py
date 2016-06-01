

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


# Load picture and detect edges
image_rgb = plt.imread('car3.png');
image = color.rgb2gray(image_rgb);
image = img_as_ubyte(image)
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

# Detect two radii
hough_radii = np.arange(140, 160, 2)
hough_radii = np.arange(80, 100, 2)
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    num_peaks = 2
    peaks = peak_local_max(h, num_peaks=num_peaks)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

# Draw the most prominent 5 circles
image = color.gray2rgb(image)
plt.imsave('car.png', image)
for idx in np.argsort(accums)[::-1][:4]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    image[cy, cx] = (220, 20, 20)
    image[cy+1, cx] = (220, 20, 20)
    image[cy-1, cx] = (220, 20, 20)
    image[cy, cx+1] = (220, 20, 20)
    image[cy, cx-1] = (220, 20, 20)

plt.imsave('hough.png',image)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()



