from rasterize import raster_mask
from draw_patch import DrawTrainingSample
from glob import glob as glob
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", type=str, help="Raster/tile directory (tif format)", required=True)
parser.add_argument("-i", nargs='+', help="Input shapefile list", required=True)
parser.add_argument("-l", type=str, help="Legend for ground-truth fusion", required=True)
parser.add_argument("-o", type=str, help="Output directory", required=True)
parser.add_argument("-t", type=str, help="Number of training samples per tile", required=True)
args = parser.parse_args()

raster_dir = args.d
shp_list = args.i
legend = args.l
out_dir = args.o
n_train = args.t
#
tiles = glob(raster_dir+'/*.tif')
for t in tiles:
    t_name = os.path.splitext(os.path.basename(t))[0]
    gtFuse_dir = out_dir + '/' + t_name + '/gt'
    mask_dir = './'+gtFuse_dir + '/mask'
    data_dir = out_dir + '/'+t_name+'/patch/data'
    label_dir = out_dir + '/'+t_name+'/patch/label'
    try:
        os.makedirs(mask_dir)
        os.makedirs(label_dir)
        os.makedirs(data_dir)
    except OSError:
        pass
    print(t_name)
    for shp in shp_list:
        class_name = os.path.splitext(os.path.basename(shp))[0]
        shp_rasterized = mask_dir+'/'+class_name+'.tif'
        raster_mask(shp, t, shp_rasterized)
    gtFuse_str = 'Legendest masques2label:sansconflit '+legend+' '+mask_dir+'/ '+gtFuse_dir+'/'+t_name+'.tif'
    #print(gtFuse_str)
    subprocess.call(gtFuse_str, shell=True)

    DrawTrainingSample(t, gtFuse_dir+'/'+t_name+'.tif', t_name, int(n_train), 65, data_dir, label_dir)
