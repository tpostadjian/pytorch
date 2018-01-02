# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, ogr
import math
import numpy as np


#
def raster2array(srcImg):
    ds_raster = gdal.Open(srcImg)
    rasterBandCount = ds_raster.RasterCount
    listOfArrays = []
    for band in range(0, rasterBandCount):
        listOfArrays.append(ds_raster.GetRasterBand(band + 1).ReadAsArray())
    return listOfArrays


#
def getNoDataValue(srcImg):
    ds_raster = gdal.Open(srcImg)
    band = ds_raster.GetRasterBand(1)
    return band.GetNoDataValue()


#
def array2raster(srcImg, newRaster, listOfArrays, n_bands, reductionBounds=False, trainingSelection=False, patch_size=0,
                 pixX=0, pixY=0):
    ds_raster = gdal.Open(srcImg)
    rasterBandCount = ds_raster.RasterCount
    geoTransform = ds_raster.GetGeoTransform()
    proj = ds_raster.GetProjection()

    if reductionBounds == True:
        driver = gdal.GetDriverByName('GTiff')
        cols = ds_raster.RasterXSize - patch_size
        rows = ds_raster.RasterYSize - patch_size
        outRaster = driver.Create(newRaster, cols, rows, rasterBandCount, gdal.GDT_Byte, ['COMPRESS=LZW'])
        outRaster.SetGeoTransform(( \
            geoTransform[0] + int(geoTransform[1] * patch_size / 2), \
            geoTransform[1], \
            geoTransform[2], \
            geoTransform[3] + int(geoTransform[5] * patch_size / 2), \
            geoTransform[4], \
            geoTransform[5]))

    elif trainingSelection == True:
        driver = gdal.GetDriverByName('GTiff')
        cols = patch_size
        rows = patch_size
        outRaster = driver.Create(newRaster, cols, rows, rasterBandCount, gdal.GDT_Int32)
        outRaster.SetGeoTransform(( \
            geoTransform[0] + geoTransform[1] * (pixY - int(patch_size / 2)), \
            geoTransform[1], \
            geoTransform[2], \
            geoTransform[3] + geoTransform[5] * (pixX - int(patch_size / 2)), \
            geoTransform[4], \
            geoTransform[5]))

    else:
        driver = gdal.GetDriverByName('GTiff')
        cols = ds_raster.RasterXSize
        rows = ds_raster.RasterYSize
        outRaster = driver.Create(newRaster, cols, rows, n_bands, gdal.GDT_Byte)  # , ['COMPRESS=LZW'])
        outRaster.SetGeoTransform(geoTransform)

    for band in range(0, len(listOfArrays)):
        outband = outRaster.GetRasterBand(band + 1)
        outband.WriteArray(listOfArrays[band])
        outband.FlushCache()

    outRaster.SetProjection(proj)
