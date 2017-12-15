import numpy as np
from skimage import io
from PIL import Image
from skimage.segmentation import slic, felzenszwalb
from skimage.segmentation import mark_boundaries
from scipy.misc import imsave
import matplotlib.pyplot as plt
import os
import time
import subprocess
import matplotlib.pyplot as plt

def SLIC(img_name, img, n_seg=1000):
    # ~ ds_arr_int = skimage.img_as_int(img)[0:3,:,:]
    # ~ ds_arr_f = skimage.img_as_float(img)[0:3,:,:]
    # ~ ds_arr_int = ds_arr_int.transpose(1, 2, 0)
    # ~ ds_arr_f = ds_arr_f.transpose(1, 2, 0)
    img = img.astype(np.float)
    img_rgb = img[0:3, :, :]
    # rescale intensity
    mr = np.min(img_rgb[0, :, :])
    mg = np.min(img_rgb[1, :, :])
    mb = np.min(img_rgb[2, :, :])
    Mr = np.max(img_rgb[0, :, :])
    Mg = np.max(img_rgb[1, :, :])
    Mb = np.max(img_rgb[2, :, :])
    r_float = (img_rgb[0, :, :] - mr) / (Mr - mr)
    g_float = (img_rgb[1, :, :] - mg) / (Mg - mg)
    b_float = (img_rgb[2, :, :] - mb) / (Mb - mb)
    r_rescaled = np.array([r_float, g_float, b_float])
    img_rgb = img[0:3, :, :].transpose(1, 2, 0)
    r_rescaled = r_rescaled.transpose(1, 2, 0)

    print("********************** Segmentation running **********************")
    start_time = time.time()
    segments = slic(r_rescaled, n_segments=n_seg, compactness=15, convert2lab=True)
    imsave('Results/GPU/finistere_slic/' + img_name + '_slic.png', segments)
    print("********************* Segmentation: %s seconds *******************" % (time.time() - start_time))

    # ~ fig = plt.figure("segmentation_15_%d" %(n_seg),figsize=(2.100, 2.100), dpi=100)
    # ~ ax = fig.add_subplot(1,1,1)
    # ~ segmentation = mark_boundaries(img_rgb,segments)
    # ~ print(segmentation.shape)
    # ~ ax.imshow(segmentation)
    # ~ plt.axis("off")
    # ~ plt.show()
    # ~ plt.savefig("segmentation_15_%d.png" %(n_seg), dpi=1000)
    return segments


def PFF(img, cir=False, sigma=0.8, k=30, min_size=10):
    name = os.path.basename(img)
    out_dir = os.path.dirname(img)
    name = name.split('.')[0]
    # To Byte
    ByteName = name + '_byte.tif'
    Bytepath = os.path.join(out_dir, ByteName)
    ByteString = 'Ech_noifst ReetalQuantile ' + img + ' 0.1 0.1 ' + Bytepath
    subprocess.call(ByteString, shell=True)

    if cir == True:
        print('CIR mode')
        CIRName = name + '_byteCIR.tif'
        CIRpath = os.path.join(out_dir, CIRName)
        CIRstring = 'gdal_translate -of GTIFF -b 4 -b 1 -b 2 ' + Bytepath + ' ' + CIRpath
        subprocess.call(CIRstring, shell=True)
        in_img = CIRpath
        PFFname = name + '_CIRseg.tif'
    else:
        in_img = Bytepath
        PFFname = name + '_seg.tif'
    # Segmentation
    PFFpath = os.path.join(out_dir, PFFname)
    PFFstring = 'SegmentationPFFst ' + str(sigma) + ' ' + str(k) + ' ' + str(min_size) + ' ' + in_img + ' ' + PFFpath
    subprocess.call(PFFstring, shell=True)
# ~
# ~ img = 'tile_16500_38500.tif'
# ~ PFF(img,cir=False)

def PFF_scikit(img, scale=60, sigma=0.8, min_size=20):
    img = io.imread(img)
    img = img.astype(np.uint8)
    img = np.reshape(img, (4,2100,2100))
    img_rgb = img[0:3, :, :]
    # rescale intensity
    mr = np.min(img_rgb[0, :, :])
    mg = np.min(img_rgb[1, :, :])
    mb = np.min(img_rgb[2, :, :])
    Mr = np.max(img_rgb[0, :, :])
    Mg = np.max(img_rgb[1, :, :])
    Mb = np.max(img_rgb[2, :, :])
    r_float = (img_rgb[0, :, :] - mr) / (Mr - mr)
    g_float = (img_rgb[1, :, :] - mg) / (Mg - mg)
    b_float = (img_rgb[2, :, :] - mb) / (Mb - mb)
    r_rescaled = np.array([r_float, g_float, b_float])
    img_rgb = img[0:3, :, :].transpose(1, 2, 0)
    r_rescaled = r_rescaled.transpose(1, 2, 0)

    print("********************** Segmentation running **********************")
    start_time = time.time()
    seg = felzenszwalb(r_rescaled, scale, sigma, min_size)
    n_seg = np.unique(seg)
    print(n_seg.shape)
    imsave('/media/tpostadjian/Data/These/Test/slic2label/tile_16500_38500_pff.png', seg)
    print("********************* Segmentation: %s seconds *******************" % (time.time() - start_time))
    return seg


# img = '/media/tpostadjian/Data/These/Test/slic2label/tile_16500_38500_byte.tif'
# PFF_scikit(img)