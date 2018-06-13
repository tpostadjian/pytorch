import gdal
from glob import glob as glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, help="Directory with images to rescale")
parser.add_argument("-o", type=str, help="Directory zith rescaled images (output)")
args = parser.parse_args()

d_in = args.i
d_out = args.o

try:
    os.makedirs(d_out)
except OSError:
    pass

imgs = glob(d_in + '/*.tif')

for im in imgs:
    im_name = os.path.basename(im)
    ds = gdal.Open(im)
    scale = ''
    for band in range(ds.RasterCount):
        dsband = ds.GetRasterBand(band + 1)
        stats = dsband.GetStatistics(True, True)
        min = stats[0]
        max = stats[1]
        scale += '-scale_' + str(band + 1) + ' ' + str(min) + ' ' + str(max) + ' '
    options_list = ['-ot Byte', '-of GTiff', scale]
    options_string = ' '.join(options_list)

    img_out = d_out + '/' + im_name
    ds = gdal.Translate(img_out, ds, options=options_string)
