import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar


def SPImg(pix_pred, out_img):
    txt = np.loadtxt(pix_pred)
    nl, nc = 2036, 2036

    output = np.empty((nl, nc), dtype=int)

    bar = progressbar.ProgressBar(maxval=nl*nc).start()
    count = 0
    for l in range(nl):
        for c in range(nc):
            line = txt[l*nc + c, :]
            label = line.argmax()
            output[l, c] = label + 1
            bar.update(count)
            count = count + 1

    scipy.misc.toimage(output, cmin=0, cmax=255).save(out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='pixelwise predictions')
    parser.add_argument('-o', help='output classification image')
    args = parser.parse_args()

    f_txt = args.i
    out = args.o
