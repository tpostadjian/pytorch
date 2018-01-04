import subprocess
from run_rasterisation import rasterisation
import argparse
import os
import shutil


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("-l", nargs='+', help="Input shapefile list", required=True)
parser.add_argument("-c", type=str, help="Input classification", required=True)
parser.add_argument("-r", type=str, help="Input raster", required=True)
parser.add_argument("-o", type=str, help="Output directory", required=True)
parser.add_argument("-f", type=str, help="Field to filter")
parser.add_argument("-s", type=str2bool, nargs='?', const=True, help="Split classes within same shapefile - vs. - "
                                                                     "consider whole shapefile as a unique class",
                    required=True)
args = parser.parse_args()

list_shapefile = args.l
classif = args.c
raster = args.r
outDir = args.o
field = args.f
splitting = args.s

img_name = os.path.splitext(os.path.basename(raster))[0]
classif_name = os.path.splitext(os.path.basename(classif))[0]
mask_dir = outDir + '/masks/' + img_name
valid_dir = outDir + '/cross_validation/' + classif_name

try:
    os.makedirs(mask_dir)
except OSError:
    pass

try:
    os.makedirs(valid_dir)
except OSError:
    pass

for shp in list_shapefile:
    cls_name = os.path.splitext(os.path.basename(shp))[0]
    if os.path.isfile(mask_dir + '/gt.tif'):
        continue
    else:
        output = rasterisation(shp, classif, splitting, field, outDir)
        shutil.move(output, mask_dir + '/' + cls_name + '.tif')
#
concat_str = "Legendest masques2label:sansconflit legende_for_concac_" + os.path.basename(outDir) + ".txt " + mask_dir + "/" + " " + mask_dir + "/gt.tif "
subprocess.call(concat_str, shell=True)

eval_str = "Evalst " + classif + " " + mask_dir + "/gt.tif " + valid_dir + "/bm.tif legende.txt " + valid_dir + "/mat2conf.txt --Kappa"
subprocess.call(eval_str, shell=True)
