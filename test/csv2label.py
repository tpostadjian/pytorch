import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar


def SPImg(pix_pred, out_img, patchsize=65):
    txt = np.loadtxt(pix_pred)
    nl, nc = 2100, 2100
    offset = int(patchsize / 2)

    output = np.empty((nl, nc), dtype=int)

    bar = progressbar.ProgressBar(maxval=(nl-2*offset)*(nc-2*offset)).start()
    count = 1
    for l in range(offset, nl-offset):
        for c in range(offset, nc-offset):
            line = txt[(l-offset) * (nc - 2*offset) + c-offset, :]
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
