import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='pixelwise predictions')
parser.add_argument('-o', help='output classif image')
args = parser.parse_args()

f_txt =args.i
out = args.o

nl, nc = 2036

offset = int(65 / 2)
txt = np.loadtxt(f_txt)

output = np.empty((nl, nc), dtype=int)

bar = progressbar.ProgressBar(maxval=nl*nc).start()
count=0
for l in range(nl):
    for c in range(nc):
        line = txt[l*nc + c, :]
        label = line.argmax()
        output[l, c] = label + 1
        bar.update(count)
        count = count + 1

imsave(out, output)
