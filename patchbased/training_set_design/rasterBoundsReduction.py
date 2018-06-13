# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, ogr
import numpy as np
from rasterProcessing import raster2array, array2raster
import math


def reduceBounds(base_raster, out_raster, patch_size):
    ds_raster = gdal.Open(base_raster)
    geoTransform = ds_raster.GetGeoTransform()

    base_raster_arr = raster2array(base_raster)

    offset = int(patch_size / 2)
    out_raster_arr = base_raster_arr[offset:np.shape(base_raster_arr)[0] - offset, \
                     offset:np.shape(base_raster_arr)[1] - offset]

    array2raster(base_raster, out_raster, out_raster_arr, True, False, patch_size)
