import scipy
import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='prediction over SSPs')
parser.add_argument('-a', help='SSPs image')
parser.add_argument('-o', help='output classif image')
args = parser.parse_args()

f_txt = args.i
f_img = args.a
out = args.o
offset = int(65 / 2)

txt = np.loadtxt(f_txt)
img = scipy.ndimage.imread(f_img)
img = img[offset:img.shape[0] - offset, offset:img.shape[1] - offset]
nl, nc = img.shape

output = np.empty((nl, nc), dtype=int)

bar = progressbar.ProgressBar(maxval=nl*nc).start()
count = 0
# Mean
for i in range(nl):
    for j in range(nc):
        # ~ seg_id = img[i,j]+1
        seg_pix = img[i, j]
        seg_id = np.where(txt[:, 0] == seg_pix)
        if seg_id[0].shape[0] == 1:
            seg = txt[seg_id][0, 1:6]
            label = seg.argmax()
            output[i, j] = label + 1
        bar.update(count)
        count += 1

imsave(out, output)
