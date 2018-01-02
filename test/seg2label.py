import scipy
import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar


def SSImg(seg_pred, seg_img, out_img, patchsize):
    txt = np.loadtxt(seg_pred)
    img = scipy.ndimage.imread(seg_img)
    offset = int(patchsize / 2)
    img = img[offset:img.shape[0] - offset, offset:img.shape[1] - offset]
    nl, nc = img.shape

    output = np.empty((nl, nc), dtype=int)

    bar = progressbar.ProgressBar(maxval=nl * nc).start()
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

    imsave(out_img, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='prediction over SSPs')
    parser.add_argument('-a', help='SSPs image')
    parser.add_argument('-o', help='output classification image')
    args = parser.parse_args()

    f_txt = args.i
    f_img = args.a
    out = args.o
    patch_size = 65

    SSImg(f_txt,f_img, out, patch_size)