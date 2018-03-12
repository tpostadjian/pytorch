from fcn_dataset_loader import SPOT_dataset
from fcn_trainer import Trainer
import glob.glob as glob


label_dir = '../training_set/label/data_{}.tif'
data_dir = '../training_set/data/label_{}.tif'
all_files = glob(label_dir.replace('{}', '*.tif'))
all_ids = []
dataset = SPOT_dataset()