import numpy as np
import filter as f

im = np.array([[1.0,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])
k = np.array([[1.0,1],[1,1]]);

a = f.Filt(im, k, 0);
