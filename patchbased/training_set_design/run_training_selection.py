from osgeo import gdal, ogr
from get_class_shapefile import getClasses
from rasterize import raster_mask
from noDataValue import eliminateNoDataPix
from rasterBoundsReduction import reduceBounds
from select_training import DrawSampleTraining
import argparse, os
import shutil


def training_selection(shapefile, base_raster, patch_size, field, n_training, splitting, outDir):
    # Directory where training samples must be stored
    training_dir = outDir + "/"
    try:
        os.makedirs(training_dir)
    except OSError:
        pass

    # Directory for temporary files
    temp_dir = outDir + "/temp/"
    try:
        os.makedirs(temp_dir)
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

            try:
                os.mkdir(temp_dir + base_name)
            except OSError:
                pass

            raster_mask(shapefile, base_raster, field, fieldValues[i], \
                        temp_dir + base_name + "/" + base_name, splitting)

            eliminateNoDataPix(base_raster, temp_dir + base_name + "/" + base_name + "_rasterized.tif", \
                               temp_dir + base_name + "/" + base_name + "_noDataCorrected.tif")

            DrawSampleTraining(base_raster, temp_dir + base_name + "/" + base_name + "_rasterfinal.tif", \
                               fieldValues[i], n_training, patch_size)

            print(fieldValues[i] + " : Processing Done")

    else:

        print("No splitting mode")
        fieldValues = None
        base_name = os.path.splitext(os.path.split(os.path.abspath(shapefile))[1])[0]

        try:
            os.mkdir(temp_dir + base_name)
        except OSError:
            pass

        raster_mask(shapefile, base_raster, field, fieldValues, \
                    temp_dir + base_name + "/" + base_name, splitting)

        eliminateNoDataPix(base_raster, temp_dir + base_name + "/" + base_name + "_rasterized.tif", \
                           temp_dir + base_name + "/" + base_name + "_rasterfinal.tif")

        DrawSampleTraining(base_raster, temp_dir + base_name + "/" + base_name + "_rasterfinal.tif", \
                           base_name, n_training, patch_size, temp_dir, training_dir)

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("v", type=str, help="Input shapefile")
    parser.add_argument("r", type=str, help="Input raster")
    parser.add_argument("o", type=str, help="Output directory")
    parser.add_argument("s", type=int, help="Size of net inputs (patch)")
    parser.add_argument("f", type=str, help="Field to filter")
    parser.add_argument("n", type=int, help="Number of training patches")
    parser.add_argument("m", action='store_false',
                        help="Split classes within same shapefile or consider whole shapefile as a unique class")
    args = parser.parse_args()

    training_selection(args.v, args.r, args.s, args.f, args.n, args.m, args.o)
