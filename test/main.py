from pred4seg import prediction
from classes2class import classDecision
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-d', help='results directory')
parser.add_argument("-s", type=str2bool, nargs='?',
                    const=True, help="Segmentation flag.")
parser.add_argument('-r', help='segmentation case: pixel density to classify - 1 is full prediction')
args = parser.parse_args()


# Pixelwise or Segmentwise
tag = args.s
# SSImg --> Semantic Segmentation Img
# SPImg --> Semantic Pixel Img
if tag:
    from seg2label import SSImg
else:
    from csv2label import SPImg

work_dir = args.d

tile = os.path.basename(img)
img_name = tile.split('.')[0]

if tag:
    ratio = args.r
    prediction(work_dir, tag, ratio)


    classDecision(out_dir + "/" + img_name + "_pred_pix.txt")


    SSImg()

else:
    prediction(work_dir, tag)
    SPImg()