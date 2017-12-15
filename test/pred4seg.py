from seg import SLIC, PFF
import subprocess
import numpy as np
import time
from glob import glob as glob
import os
import tif2h5
import h5py

ratio_pix2class = 0.4
patch_size = 65
offset = int(patch_size / 2)
list_img = glob('tile_16500_38500_seg.tif')
count = 0

# Loop over tiles covering the ROI
for img in list_img:
    tile = os.path.basename(img)
    img_name = tile.split('.')[0]
    pythonString = "/usr/bin/python2.7 tiff2hdf5_1file.py " + tifDirectory + "/" + img_name + ".tif " + h5Directory + "/" + img_name + ".h5"
    subprocess.call(pythonString)
    data = h5py.File(h5Directory + "/" + img_name + '.h5')
    img = data['img_1']
    img_np = np.array(img)
    # Reduce boundaries to avoid computing on "no data" areas
    seg = img[offset:img.shape[0] - offset, offset:img.shape[1] - offset]
    nl, nc = seg.shape
    print(nl, nc)
    # how many segments ?
    n_s = np.unique(seg)
    # loop over segments (0.0166s/segment)
    start_time = time.time()
    # file to store class probabilities
    f = open(img_name + '_pred40%.txt', 'w')
    print("********************* Classification running *********************")
    for id_seg in n_s:
        # retrieve pixels indices for given segment (np.where takes time!)
        seg_ind = np.where(seg == id_seg)
        n_pix_seg = seg_ind[0].shape[0]
        # randomly pick pixels in the current segment
        n_pix2class = int(n_pix_seg * ratio_pix2class)
        pix2class = np.random.randint(n_pix_seg, size=n_pix2class)
        ind_pix2class = [seg_ind[0][pix2class], seg_ind[1][pix2class]]
        # Classify each picked pixel
        for pix in range(n_pix2class):
            x_pix = ind_pix2class[0][pix]
            y_pix = ind_pix2class[1][pix]
            line = listlabel[x_pix * nc + y_pix, :]
            f.write("%d %.3f %.3f %.3f %.3f %.3f\n" % (id_seg + 1, line[0], line[1], line[2], line[3], line[4]))
            count = count + 1

    print("******************** Classification: %s seconds ******************" % (time.time() - start_time))
    print(x_pix * nl + y_pix)
