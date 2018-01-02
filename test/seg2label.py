import scipy
import scipy.ndimage
import numpy as np
from scipy.misc import imsave
import argparse
import progressbar


def SSImg(seg_pred, seg_img, out_img, patchsize=65):
    txt = np.loadtxt(seg_pred)
    img = scipy.ndimage.imread(seg_img)
    offset = int(patchsize / 2)
    nl, nc = img.shape

    output = np.empty(img.shape, dtype=int)

    bar = progressbar.ProgressBar(maxval=nl * nc).start()
    count = 0

    for i in range(offset, nl-offset):
        for j in range(offset, nc-offset):
            seg_pix = img[i, j]
            seg_id = np.where(txt[:, 0] == seg_pix)
            if seg_id[0].shape[0] == 1:
                seg = txt[seg_id][0, 1:6]
                label = seg.argmax()
                output[i, j] = label + 1
            bar.update(count)
            count += 1

    scipy.misc.toimage(output, cmin=0, cmax=255).save(out_img)


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

    SSImg(f_txt, f_img, out, patch_size)