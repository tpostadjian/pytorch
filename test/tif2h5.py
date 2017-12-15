import gdal
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('f', help='tiff image')
parser.add_argument('l', help='output hdf5 file name')
args = parser.parse_args()

imgPath = args.f
output = args.l

img = gdal.Open(imgPath)
n_band = img.RasterCount
h = img.RasterXSize
w = img.RasterYSize

h5f = h5py.File(output, 'w')
arr = np.empty([n_band, w, h])
for k in range(0, n_band):
    band = img.GetRasterBand(k + 1).ReadAsArray()
    arr[k, :, :] = band

h5f.create_dataset('img_' + str(1), data=arr, compression="gzip")

h5f.close()
