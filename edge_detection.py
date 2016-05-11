import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage

def edge_detection(im):
    edge1 = scipy.ndimage.sobel(im, 0)
    edge2 = scipy.ndimage.sobel(im, 1)
    im_edge1 = (edge1**2 + edge2**2)**0.5
    f1 = np.array([[1,2,1],
                   [0,0,0],
                  [-1,-2,-1]])
    f2 = np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]]);
    edge1 = scipy.ndimage.filters.correlate(im, f1);
    edge2 = scipy.ndimage.filters.correlate(im, f2);
    im_edge2 = (edge1**2 + edge2**2)**0.5
    return im_edge1, im_edge2




def call_edge_detection():
    img = scipy.misc.lena()
    im_edge1, im_edge2 = edge_detection(img)
    f=plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    f.add_subplot(1,3,2)
    plt.imshow(im_edge1, cmap='gray')
    f.add_subplot(1,3,3)
    plt.imshow(im_edge2, cmap='gray')
    plt.show();


if __name__ == '__main__':
    call_edge_detection();
