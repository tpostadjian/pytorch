# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

import gdal, ogr
from get_class_shapefile import getClasses
from rasterize import raster_mask
from noDataValue import eliminateNoDataPix
from rasterBoundsReduction import reduceBounds
from erosion import Erode
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("v", type=str, help="Input shapefile")
parser.add_argument("r", type=str, help="Input raster")
parser.add_argument("o", type=str, help="Output directory")
parser.add_argument("f", type=str, help="Field to filter")
parser.add_argument("m", action='store_false',
                    help="Split classes within same shapefile or consider whole shapefile as a unique class")
args = parser.parse_args()

shapefile = args.v
raster = args.r
field = args.f
splitting = args.m
outDir = args.o

# Directory for binary rasters
bin_dir = outDir + "/bin_rasters/"
try:
    os.makedirs(bin_dir)
except OSError:
    pass

if splitting == True:

    print("Splitting mode")

    fieldValues = getClasses(shapefile, field)
    result = []
    n_values = len(fieldValues)

    # Ignore under represented classes (training set has to count at most
    # for 10% of the total set for each class)
    shp = ogr.Open(shapefile)
    for i in range(0, n_values):
        lyr = shp.GetLayer()
        lyr.SetAttributeFilter('%s = "%s"' % (field, fieldValues[i]))
        if lyr.GetFeatureCount() > 5 * patch_size:
            result.append(fieldValues[i])
    fieldValues = result

    for i in range(0, len(fieldValues)):
        base_name = fieldValues[i]
        if ' ' in base_name:
            base_name.replace(' ', '_')

        bin_dir = bin_dir + base_name
        try:
            os.mkdir(bin_dir)
        except OSError:
            pass

        raster_mask(shapefile, raster, field, fieldValues[i], \
                    bin_dir + "/" + base_name, splitting)

        eliminateNoDataPix(raster, bin_dir + "/" + base_name + "_rasterized.tif",
                           bin_dir + "/" + base_name + "_noDataCorrected.tif")

        Erode(raster, bin_dir + "/" + base_name + "_rasterfinal.tif", fieldValues[i])

        print(fieldValues[i] + " : Processing Done")

else:

    print("No splitting mode")
    fieldValues = None
    base_name = os.path.splitext(os.path.split(os.path.abspath(shapefile))[1])[0]

    bin_dir = bin_dir + base_name
    try:
        os.mkdir(bin_dir)
    except OSError:
        pass

    bin_name = bin_dir + "/" + base_name
    raster_mask(shapefile, raster, field, fieldValues, bin_name, splitting)

    eliminateNoDataPix(raster, bin_name + "_rasterized.tif", bin_name + "_rasterfinal.tif")

# Erode(raster, base_name, bin_dir)
