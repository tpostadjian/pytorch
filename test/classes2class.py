import numpy as np
import argparse
import progressbar
from scipy.sparse import csr_matrix
import scipy.ndimage

def compute_sparse(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_sparse(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def classDecision2(all_pred, seg_pred, seg_img):
    preds = np.loadtxt(all_pred)
    img = scipy.ndimage.imread(seg_img)

    offset = 32
    nc, nl = img.shape

    seg_r = img[offset:nl - offset, offset:nc - offset]
    tmp = np.empty(seg_r.shape, dtype=int)

    seg_inds = get_indices_sparse(preds[:, 0])
    seg_inds.remove(seg_inds[0])

    print(seg_inds[1])

    n_seg = preds.max()



    out = open(seg_pred, 'w')
    # bar = progressbar.ProgressBar(maxval=n_seg).start()
    #
    # for i in range(0, int(n_seg) + 1):
    #     ind_pixs = np.where(preds[:, 0] == i)
    #     pixs = preds[ind_pixs][:, 1:6]
    #     n_pixs = pixs.shape[0]
    #     if n_pixs == 0:
    #         continue
    #     else:
    #         proba_seg = np.sum(pixs, axis=0) / n_pixs
    #         out.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (
    #             i, proba_seg[0], proba_seg[1], proba_seg[2], proba_seg[3], proba_seg[4]))
    #     bar.update(i)

    '''
    # Bayesian





    # Vote
    '''

    out.close()


def classDecision(all_pred, seg_pred):
    preds = np.loadtxt(all_pred)
    n_seg = preds.max()
    out = open(seg_pred, 'w')
    bar = progressbar.ProgressBar(maxval=n_seg).start()

    for i in range(0, int(n_seg)+1):
        ind_pixs = np.where(preds[:, 0] == i)
        pixs = preds[ind_pixs][:, 1:6]
        n_pixs = pixs.shape[0]
        if n_pixs == 0:
            continue
        else:
            proba_seg = np.sum(pixs, axis=0) / n_pixs
            out.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (
                i, proba_seg[0], proba_seg[1], proba_seg[2], proba_seg[3], proba_seg[4]))
        bar.update(i)

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

    classDecision2(f_txt, f_out, segm)
