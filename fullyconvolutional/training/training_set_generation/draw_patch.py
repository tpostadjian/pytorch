# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
from scipy import ndimage
from rasterProcessing import array2raster
import os


# Crop raster around a given pixel i j
def cropAroundPixel(base_raster, l_bands, out_crop, i, j, patch_size):
    offset = int(patch_size / 2)
    list_raster_arr = []
    for band in range(len(l_bands)):
        current_band = l_bands[band]
        patch_band = current_band.ReadAsArray(j - offset, i - offset, patch_size, patch_size)
        list_raster_arr.append(patch_band)
    array2raster(base_raster, out_crop, list_raster_arr, False, True, patch_size, i, j)


# Draw random polygon of a class within layer for training
def DrawTrainingSample(base_raster, label_raster, base_name, n_training, patch_size, data_dir, label_dir):
    # Erosion for edge problems
    ds_raster = gdal.Open(base_raster)
    data_bands = []
    for band in range(ds_raster.RasterCount):
        data_bands.append(ds_raster.GetRasterBand(band + 1))
    ds_arr = ds_raster.GetRasterBand(1).ReadAsArray()
    noData_arr = ds_arr
    struct = np.ones((patch_size, patch_size), dtype=bool)
    ds_noData_arr = ndimage.binary_erosion(noData_arr, struct)

    # Binary raster (class "A" / not class "A")
    ds_label = gdal.Open(label_raster)
    band = ds_label.GetRasterBand(1)
    label_band = [band]
    ds_label_arr = band.ReadAsArray()
    ds_label_arr[ds_noData_arr == False] = 0
    nc, nl = ds_label_arr.shape

    # Number of pixels belonging to class "A"
    n_pix = np.count_nonzero(ds_label_arr != 0)

    # Check if current tile has enough labelled pixels
    if n_pix >= nc*nl/4:
        # Stride to pick a training sample
        s = int(n_pix / n_training)

        # Get indices of class "A" pixels
        pix_255 = np.where(ds_label_arr != 0)

        # Keep one pixel every s pixels
        pix_training = (pix_255[0][0:pix_255[0].shape[0]:s], pix_255[1][0:pix_255[0].shape[0]:s])

        # Run through the ROI to pick training patches, beginning upper left corner
        offset = int(patch_size / 2)

        for k in range(0, n_training):
            i = pix_training[0][k]
            j = pix_training[1][k]

            out_label = label_dir + "/label_" + str(k + 1) + ".tif"
            list_label_arr = []
            for band in range(len(label_band)):
                current_band = label_band[band]
                patch_band = current_band.ReadAsArray(j - offset, i - offset, patch_size, patch_size)
                list_label_arr.append(patch_band)
            # check if 50% at most of pixels within patch are no data pixels.
            if np.count_nonzero(patch_band) >= patch_size**2/2:
                array2raster(base_raster, out_label, list_label_arr, False, True, patch_size, i, j)

            # out_label = label_dir + "/label_" + str(k + 1) + ".tif"
            # cropAroundPixel(base_raster, label_band, out_label, pix_training[0][k], pix_training[1][k], patch_size)

                out_data = data_dir + "/data_" + str(k + 1) + ".tif"
                list_data_arr = []
                for band in range(len(data_bands)):
                    current_band = data_bands[band]
                    patch_band = current_band.ReadAsArray(j - offset, i - offset, patch_size, patch_size)
                    list_data_arr.append(patch_band)
                array2raster(base_raster, out_data, list_data_arr, False, True, patch_size, i, j)

            # out_data = data_dir + "/data_" + str(k + 1) + ".tif"
            # cropAroundPixel(base_raster, data_bands, out_data, pix_training[0][k], pix_training[1][k], patch_size)

    else:
        print(base_raster + ': too few reference data')