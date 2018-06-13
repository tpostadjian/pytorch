import argparse, os, shutil
from glob import glob as glob
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("i", type=str, help="Directory with training sub-directories")
parser.add_argument("o", type=str, help="Output directory")
args = parser.parse_args()

input_dir = args.i
out_dir = args.o
try:
    os.makedirs(out_dir)
except OSError:
    pass

l_tiles = os.listdir(input_dir)

for tile in l_tiles:
    current_tile_path = os.path.join(input_dir, tile)
    classes = os.listdir(current_tile_path)
    for c in classes:
        if c != 'temp':
            current_class_path = os.path.join(current_tile_path, c)
            try:
                os.makedirs(out_dir+'/'+c)
            except OSError:
                pass
            imgs = glob(current_class_path+'/*.tif')
            j = 0
            n_dst = len(glob(out_dir+'/'+c+'/*tif')) + 1
            for img in imgs:
                src = os.path.join(current_class_path,img)
                dst = os.path.join(out_dir+'/'+c, 'training_' + str(n_dst + j) + '.tif')
                shutil.copy(img, dst)
                j+=1