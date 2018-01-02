# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, ogr
import numpy as np
from rasterProcessing import raster2array, getNoDataValue, array2raster


# Set nodata pixels to raster from another same extended raster which has nodata values
def eliminateNoDataPix(raster_ROI, binary_call_mask, binary_call_mask_corrected):
    list_band_ROI_arr = raster2array(raster_ROI)
    noDataValue = getNoDataValue(raster_ROI)
    list_raster_arr_correct = raster2array(binary_call_mask)

    raster_arr_correct = list_raster_arr_correct[0]
    raster_arr_correct[list_band_ROI_arr[0] == noDataValue] = 0
    array2raster(raster_ROI, binary_call_mask_corrected, [raster_arr_correct], 1)
