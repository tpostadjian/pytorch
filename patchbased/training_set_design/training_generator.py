from run_training_selection import training_selection
from glob import glob as glob
import subprocess
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-i", nargs='+', help="Input shapefile list", required=True)
parser.add_argument("-d", type=str, help="Raster/tile directory (tif format)", required=True)
parser.add_argument("-p", type=int, help="Input patch size", required=True)
parser.add_argument("-f", type=str, help="field to filter", required=True)
parser.add_argument("-o", type=str, help="Output directory", required=True)
parser.add_argument("-t", type=int, help="Number of training samples per tile", required=True)
parser.add_argument('-m', type=str2bool, nargs='?',
                    const=False, help='Split classes within same shapefile or consider whole shapefile as a unique class.')
args = parser.parse_args()

shp_list = args.i
raster_dir = args.d
size = args.p
field = args.f
out_dir = args.o
n_train = args.t
splitting = args.m
#
tiles = glob(raster_dir+'/*.tif')
for t in tiles:
    t_name = os.path.splitext(os.path.basename(t))[0]
    current_train_dir = out_dir+'/'+t_name
    print(t_name)
    for shp in shp_list:
        training_selection(shp, t, size, field, n_train, splitting, current_train_dir)
