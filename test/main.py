from pred4seg import prediction
from classes2class import classDecision
import argparse
from glob import glob as glob
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input image directory')
parser.add_argument('-d', help='results directory')
parser.add_argument('-s', type=str2bool, nargs='?',
                    const=True, help='Segmentation flag.')
parser.add_argument('-r', type=float, help='segmentation case: pixel density to classify - 1 is full prediction')
args = parser.parse_args()


# Pixelwise or Segmentwise
tag = args.s

work_dir = args.d
try:
    os.mkdir(work_dir)
except FileExistsError:
    pass
img_dir = args.i
list_img = glob(img_dir+'/*.tif')

# SSImg --> Semantic Segmentation Img
if tag:
    from seg2label import SSImg
    ratio = args.r

    for img in list_img:
        print(img)
        tile = os.path.basename(img)
        img_name = tile.split('.')[0]
        out_dir = work_dir + '/' + img_name

        # partial pixel prediction per segment
        prediction(work_dir, img, tag, ratio)

        # majority decision
        in_pred_pix = out_dir + "/" + img_name + "_pred_pix_" + str(int(ratio * 100)) + ".txt"
        out_pred_seg = out_dir + "/" + img_name + "_pred_seg_" + str(int(ratio * 100)) + ".txt"
        print("---- decision ----")
        classDecision(in_pred_pix, out_pred_seg)

        # classification image creation
        seg_img = out_dir+'/'+img_name+'_seg.tif'
        out_img = out_dir+'/'+img_name+'_classif_'+str(int(ratio * 100))+'.tif'
        print("---- classif image ----")
        SSImg(out_pred_seg, seg_img, out_img)

# SPImg --> Semantic Pixel Img
else:
    from csv2label import SPImg
    for img in list_img:

        tile = os.path.basename(img)
        img_name = tile.split('.')[0]
        out_dir = work_dir + '/' + img_name

        # full pixel prediction
        #prediction(work_dir, img, tag)

        # classification image creation
        preds = out_dir + "/" + img_name + "_pred_pix.txt"
        out = out_dir + '/' + img_name + '_classif_pix.tif'
        SPImg(preds, out)