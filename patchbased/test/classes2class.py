import numpy as np
import argparse
from scipy.sparse import csr_matrix
import scipy.ndimage


def compute_sparse(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_sparse(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def classDecision(all_pred, seg_pred, seg_img):
    preds = np.loadtxt(all_pred)
    img = scipy.ndimage.imread(seg_img)

    offset = 32
    nc, nl = img.shape

    seg_r = img[offset:nl - offset, offset:nc - offset]
    tmp = np.empty(seg_r.shape, dtype=int)

    seg = get_indices_sparse(preds[:, 0])
    seg.remove(seg[0])

    out = open(seg_pred, 'w')
    count = 0

    for s in seg:
        p = preds[s][:, 1:6]
        n_pix = p.shape[0]
        if n_pix == 0:
            continue
        else:
            proba_seg = np.sum(p, axis=0) / n_pix
            out.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (
                count, proba_seg[0], proba_seg[1], proba_seg[2], proba_seg[3], proba_seg[4]))
        count += 1

    '''
    # Bayesian





    # Vote
    '''

    out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='prediction over SSPs')
    parser.add_argument('-o', help='output merged predictions')
    parser.add_argument('-s', help='segmentation')
    args = parser.parse_args()

    f_txt = args.i
    f_out = args.o
    segm = args.s

    classDecision(f_txt, f_out, segm)
