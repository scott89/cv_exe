import numpy as np
import scipy
import matplotlib.pyplot as plt
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def extract_sift(im):
    im = np.double(im);
    cell_size = 16;
    if im.shape[2]!=0:
        im = rgb2gray(im);
    gy,gx = np.gradient(im, 0.5);
    num = np.array(im.shape)/cell_size
    im = im[:num[0] * cell_size, :num[1]*cell_size];
    gx = gx[:num[0] * cell_size, :num[1]*cell_size];
    gy = gy[:num[0] * cell_size, :num[1]*cell_size];
    gm = (gx**2 + gy**2)**0.5
    ori = np.arctan(gy/(gx+0.0001));


    patch_mag = gm.reshape(num[0], cell_size, num[1], cell_size)
    patch_mag = np.swapaxes(patch_mag, 1, 2)
    patch_mag = np.swapaxes(patch_mag, 0, 2);
    patch_mag = np.swapaxes(patch_mag, 1, 3);
    patch_mag = patch_mag.reshape(cell_size, cell_size,-1)
    #
    ori = ori + np.double(gx<0) * np.ones(ori.shape) *  scipy.pi
    patch_ori = ori.reshape(num[0], cell_size, num[1], cell_size)
    patch_ori = np.swapaxes(patch_ori, 1, 2)
    patch_ori = np.swapaxes(patch_ori, 0, 2);
    patch_ori = np.swapaxes(patch_ori, 1, 3);
    patch_ori = patch_ori.reshape(cell_size, cell_size, -1)
    #
    patch_ori = np.floor((patch_ori + scipy.pi/2)/(scipy.pi/4))
    patch_ori = patch_ori.astype(np.int)


    # 

    hist = np.zeros([patch_ori.shape[2], 128], dtype=np.double);
    for h in range(0, 16):
        for w in range(0,16):
            ind = 128 * np.arange(0,patch_ori.shape[2]) + np.int(8 * (np.floor(h/4) * 4 + np.floor(w/4))) + patch_ori[0,0,:]
            hist.flat[ind] = patch_mag[h,w,:];
            return hist
