# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

from osgeo import gdal, ogr
import subprocess


# Create a raster with parameters of another raster
def rasterFromBase(raster_ROI, out_raster, format, datatype):
    geoTransform = raster_ROI.GetGeoTransform()

    # buffering to handle edge issues
    cols = raster_ROI.RasterXSize
    rows = raster_ROI.RasterYSize
    proj = raster_ROI.GetProjection()
    bands = raster_ROI.RasterCount

    driver = gdal.GetDriverByName(format)
    newRaster = driver.Create(out_raster, cols, rows, bands, datatype)
    newRaster.SetProjection(proj)
    newRaster.SetGeoTransform(geoTransform)

    return newRaster


# Binary rasterization (255 : polygons belonging to the desired class
#						0 : others polygons)
def raster_mask(path2shapefile, raster_ROI, field, attribute_filter, class_selection, splitting=True):
    RASTERIZE_COLOR_FIELD = "__color__"

    # Input image
    ds_raster = gdal.Open(raster_ROI)

    # Open the shapefile
    ds_shapefile = ogr.Open(path2shapefile, 0)
    dsCopy = ogr.GetDriverByName('Memory').CopyDataSource(ds_shapefile, "")
    layer = dsCopy.GetLayer()

    fieldDef = ogr.FieldDefn(RASTERIZE_COLOR_FIELD, ogr.OFTInteger)
    layer.CreateField(fieldDef)
    layer_def = layer.GetLayerDefn()
    fieldIndex = layer_def.GetFieldIndex(RASTERIZE_COLOR_FIELD)

    if splitting == True:
        # Query to keep only class-belonging polygons
        for feature in layer:
            if feature.GetField(field) == attribute_filter:
                feature.SetField(fieldIndex, 255)
                layer.SetFeature(feature)
            else:
                feature.SetField(fieldIndex, 0)
                layer.SetFeature(feature)

        # Rasterization of polygons of interest
        raster_out_filtered = class_selection + "_rasterized.tif"
        raster_ds_filtered = rasterFromBase(ds_raster, raster_out_filtered, 'GTiff', gdal.GDT_Int32)
        gdal.RasterizeLayer(raster_ds_filtered, [1], layer, burn_values=[255],
                            options=["ATTRIBUTE=%s" % RASTERIZE_COLOR_FIELD])

    else:
        raster_out_filtered = class_selection + "_rasterized.tif"
        print(raster_out_filtered)
        raster_ds_filtered = rasterFromBase(ds_raster, raster_out_filtered, 'GTiff', gdal.GDT_Int32)
        gdal.RasterizeLayer(raster_ds_filtered, [1], layer, None, None, burn_values=[255])
