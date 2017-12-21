import torch
from torch.utils.serialization import load_lua
from seg import PFF
import subprocess
import numpy as np
import time
from glob import glob as glob
import os
import h5py
from skimage import io
import progressbar


ratio_pix2class = 0.4
patch_size = 65
offset = int(patch_size / 2)
model = '/media/tpostadjian/Data/These/Test/Results/GPU/test_101/model_float.net'
net = load_lua(model)
net.modules[1].modules[0] = torch.legacy.nn.View(1, 2048)
net = net.cuda()

list_img = glob("../test_set/tile_16500_38500.tif")
directory = os.path.dirname(list_img[0])
count = 0

# Loop over tiles covering the ROI
for img in list_img:
    # image segmentation
    PFF(img)

    # conversion to hdf5
    tile = os.path.basename(img)
    img_name = tile.split('.')[0]
    pythonString = "/usr/bin/python2.7 tif2h5.py " \
                    + img + " " \
                    + directory + "/" + img_name + ".h5"
    subprocess.call(pythonString, shell=True)
    data = h5py.File(directory + "/" + img_name + ".h5")
    img = data["img_1"]
    img_np = np.array(img)

    # Reduce boundaries to avoid computing on "no data" areas
    img_noEdge = img_np[0:3, offset:img.shape[1] - offset, offset:img.shape[2] - offset]
    nb, nl, nc = img_noEdge.shape

    # how many segments ?
    seg = io.imread(directory + "/" + img_name + "_seg.tif")
    seg = seg[offset:img.shape[1] - offset, offset:img.shape[2] - offset]
    n_s = np.unique(seg)
    # loop over segments (0.015s/segment)
    start_time = time.time()
    # file to store class probabilities
    f = open(img_name + "_pred40%.txt", "w")

    print("********************* Classification running *********************")

    bar = progressbar.ProgressBar(maxval=n_s.shape[0]).start()
    for id_seg in n_s:

        # retrieve pixels indices for given segment (np.where takes time!)
        seg_ind = np.where(seg == id_seg)
        n_pix_seg = seg_ind[0].shape[0]

        # randomly draw pixels within the current segment
        n_pix2class = int(n_pix_seg * ratio_pix2class)
        pix2class = np.random.randint(n_pix_seg, size=n_pix2class)
        ind_pix2class = [seg_ind[0][pix2class], seg_ind[1][pix2class]]

        # Classify each picked pixel
        for pix in range(n_pix2class):
            x_pix = ind_pix2class[0][pix] + offset
            y_pix = ind_pix2class[1][pix] + offset
            patch = img_np[:, x_pix - offset:x_pix + offset + 1, y_pix - offset:y_pix + offset + 1]
            patch_torch = torch.from_numpy(patch)
            # cast net to cuda
            patch_torch_f = patch_torch.float().cuda()
            preds = net.forward(patch_torch_f)
            probas = preds.exp()
            probas_np = np.array([probas[0, 0], probas[0, 1], probas[0, 2], probas[0, 3], probas[0, 4]])
            f.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (
                id_seg + 1, probas_np[0], probas_np[1], probas_np[2], probas_np[3], probas_np[4]))
        bar.update(id_seg)
        count = count + 1
    print("******************** Classification: %s seconds ******************" % (time.time() - start_time))
