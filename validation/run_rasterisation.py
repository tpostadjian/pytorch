# !/usr/bin/python2.7
# -*-coding:Latin-1 -*

import gdal, ogr
from get_class_shapefile import getClasses
from rasterize import raster_mask
from noDataValue import eliminateNoDataPix
from erosion import Erode
import argparse, os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def rasterisation(input_shapefile, input_raster, bool_split, split_field, result_dir):
    # Directory for binary rasters
    bin_dir = result_dir + "/bin_rasters/"
    patch_size = 65
    try:
        os.makedirs(bin_dir)
    except OSError:
        pass

    if bool_split:
        print("Splitting mode")

        fieldValues = getClasses(input_shapefile, split_field)
        result = []
        n_values = len(fieldValues)

        # Ignore under represented classes (training set has to count at most
        # for 10% of the total set for each class)
        shp = ogr.Open(input_shapefile)
        for i in range(0, n_values):
            lyr = shp.GetLayer()
            lyr.SetAttributeFilter('%s = "%s"' % (split_field, fieldValues[i]))
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
        base_name = os.path.splitext(os.path.basename(input_shapefile))[0]

        bin_dir = bin_dir + base_name
        try:
            os.mkdir(bin_dir)
        except OSError:
            pass

        bin_name = bin_dir + "/" + base_name
        raster_mask(input_shapefile, input_raster, split_field, fieldValues, bin_name, bool_split)

        eliminateNoDataPix(input_raster, bin_name + "_rasterized.tif", bin_name + "_rasterfinal.tif")

        return bin_name + "_rasterfinal.tif"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="Input shapefile")
    parser.add_argument("-r", type=str, help="Input raster")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-f", type=str, help="Field to filter")
    parser.add_argument("-s", type=str2bool, nargs='?', const=True, help='Split classes within same shapefile - vs. - '
                                                                         'consider whole shapefile as a unique class')
    args = parser.parse_args()

    shapefile = args.v
    raster = args.r
    field = args.f
    splitting = args.m
    outDir = args.o

    rasterisation(shapefile, raster, field, splitting, outDir)
