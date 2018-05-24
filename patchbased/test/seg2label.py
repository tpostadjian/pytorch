import scipy
import scipy.ndimage
import numpy as np
import argparse
from scipy.sparse import csr_matrix


def compute_sparse(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_sparse(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def SSImg(seg_pred, seg_img, out_img, patchsize=65):
    txt = np.loadtxt(seg_pred)
    img = scipy.ndimage.imread(seg_img)
    offset = int(patchsize / 2)
    nl, nc = img.shape

    output = np.empty(img.shape, dtype=int)

    seg_r = img[offset:nl - offset, offset:nc - offset]
    tmp = np.empty(seg_r.shape, dtype=int)

    seg_inds = get_indices_sparse(seg_r)
    seg_inds.remove(seg_inds[0])

    count = 0

    for s in seg_inds:
        if not (len(s[0]) == 0 or int(len(s[0])*0.2) == 0):
            line = txt[count][1:6]
            label = line.argmax() + 1
            tmp[s[0], s[1]] = label
            count += 1

    output[offset:nl - offset, offset:nc - offset] = tmp
    scipy.misc.toimage(output, cmin=0, cmax=255).save(out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help='prediction over SSPs')
    parser.add_argument('a', help='SSPs image')
    parser.add_argument('o', help='output classification image')
    args = parser.parse_args()

    f_txt = args.i
    f_img = args.a
    out = args.o
    patch_size = 65

    SSImg(f_txt, f_img, out, patch_size)