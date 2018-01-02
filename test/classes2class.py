import numpy as np
import argparse
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='prediction over SSPs')
parser.add_argument('-o', help='output merged predictions')
args = parser.parse_args()


def classDecision(all_pred, seg_pred):
    f_txt = np.loadtxt(all_pred)
    n_seg = f_txt.max()
    out = open(seg_pred, 'w')
    bar = progressbar.ProgressBar(maxval=n_seg).start()

    for i in range(0, int(n_seg)):
        ind_pixs = np.where(f_txt[:, 0] == i)
        pixs = f_txt[ind_pixs][:, 1:6]
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
    f_txt = args.i
    f_out = args.o

    classDecision(f_txt, f_out)
