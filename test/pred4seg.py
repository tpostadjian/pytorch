import torch
from torch.utils.serialization import load_lua
from seg import PFF
import subprocess
import numpy as np
import time
import os
import h5py
from skimage import io
# import progressbar
import tqdm
import argparse
from scipy.sparse import csr_matrix


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_sparse(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_sparse(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


# @profile
def prediction(work_dir, img, seg_flag=False, model=None, ratio_pix2class=0.2,
               patch_size=65):

    offset = int(patch_size / 2)
    net = load_lua(model)
    net.modules[1].modules[0] = torch.legacy.nn.View(1, 2048)
    net = net.cuda()

    tile = os.path.basename(img)
    img_name = tile.split('.')[0]

    # outputs directory
    out_dir = work_dir + '/' + img_name
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    # conversion to hdf5
    pythonString = "tif2h5.py " \
                   + img + " " \
                   + out_dir + "/" + img_name + ".h5"
    subprocess.call(pythonString, shell=True)

    # hdf5 --> numpy --> torch float cuda
    data = h5py.File(out_dir + "/" + img_name + ".h5")
    img_h5 = data["img_1"]
    img_np = np.array(img_h5)
    img_torch = torch.from_numpy(img_np)
    img_cuda = img_torch.float().cuda()

    # Handle edges
    img_noEdge = img_np[0:3, offset:img_np.shape[1] - offset, offset:img_np.shape[2] - offset]
    nb, nl, nc = img_noEdge.shape

    if seg_flag:
        # image segmentation
        if not os.path.isfile(out_dir + "/" + img_name + "_seg.tif"):
            print('----- segmentation -----')
            PFF(img, out_dir)

        # how many segments ?
        seg = io.imread(out_dir + "/" + img_name + "_seg.tif")
        seg = seg[offset:img_np.shape[1] - offset, offset:img_np.shape[2] - offset]
        n_s = np.unique(seg)

        # loop over segments (0.015s/segment)
        start_time = time.time()

        # file to store class probabilities
        f = open(out_dir + "/" + img_name + "_pred_pix_" + str(int(ratio_pix2class * 100)) + ".txt", "w")

        seg_ind = get_indices_sparse(seg)

        print("----- classification is running ! -----")
        # bar = progressbar.ProgressBar(maxval=n_s.shape[0]).start()
        # count = 1
        for id_seg in tqdm(n_s):
            # retrieve pixels indices for given segment
            n_pix_seg = seg_ind[id_seg][0].shape[0]

            # randomly draw pixels within the current segment
            n_pix2class = int(n_pix_seg * ratio_pix2class)
            pix2class = np.random.randint(n_pix_seg, size=n_pix2class)
            ind_pix2class = [seg_ind[id_seg][0][pix2class], seg_ind[id_seg][1][pix2class]]

            # Classify each picked pixel
            for pix in range(n_pix2class):
                x_pix = ind_pix2class[0][pix] + offset
                y_pix = ind_pix2class[1][pix] + offset
                patch = img_cuda[:, x_pix - offset:x_pix + offset + 1, y_pix - offset:y_pix + offset + 1]
                preds = net.forward(patch)
                probas = preds.exp()
                f.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (
                    id_seg, probas[0, 0], probas[0, 1], probas[0, 2], probas[0, 3], probas[0, 4]))
            # bar.update(count)
            # count = count + 1
        print("Classification took: %s seconds -----" % (time.time() - start_time))

    else:
        start_time = time.time()

        # file to store class probabilities
        f = open(out_dir + "/" + img_name + "_pred_pix.txt", "w")

        print("----- classification is running ! -----")
        # bar = progressbar.ProgressBar(maxval=nl * nc).start()
        # count = 0
        for l in tqdm(range(nl)):
            for c in range(nc):
                x_pix = l + offset
                y_pix = c + offset
                patch = img_cuda[:, x_pix - offset:x_pix + offset + 1, y_pix - offset:y_pix + offset + 1]
                preds = net.forward(patch)
                probas = preds.exp()
                f.write("%.3f %.3f %.3f %.3f %.3f\n" % (
                    probas[0, 0], probas[0, 1], probas[0, 2], probas[0, 3], probas[0, 4]))
                # bar.update(count)
                # count = count + 1
        print("Classification took: %s seconds -----" % (time.time() - start_time))
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='results directory')
    parser.add_argument("-s", type=str2bool, nargs='?',
                        const=True, help="segmentation flag.")
    args = parser.parse_args()

    prediction(args.d, seg_flag=args.s)
