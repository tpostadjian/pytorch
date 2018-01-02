# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, gdalnumeric, ogr, osr
import Image, ImageDraw
import numpy as np
from scipy import ndimage
from rasterProcessing import raster2array, array2raster
import os, time


# Draw random polygon of a class within layer for training
def Erode(base_raster, base_name, bin_dir):
    # Erosion for edge problems
    ds = gdal.Open(base_raster)
    ds_arr = ds.GetRasterBand(1).ReadAsArray()
    noData_arr = ds_arr
    struct = np.ones((3, 3))
    debut = time.time()
    ds_noData_arr = ndimage.binary_erosion(noData_arr, struct)
    fin = time.time()
    print("time to compute erosion for training selection : " + str(fin - debut))

    # Binary raster (class "A" / not class "A")
    ds_bin = gdal.Open(bin_dir + "/" + base_name + "_rasterfinal.tif")
    geoTransform = ds_bin.GetGeoTransform()
    band = ds_bin.GetRasterBand(1)
    ds_bin_arr = band.ReadAsArray()
    ds_bin_arr[ds_noData_arr == False] = 0
    array2raster(base_raster, bin_dir + "/" + base_name + "_erosion.tif", [ds_bin_arr], 1, False, False)
