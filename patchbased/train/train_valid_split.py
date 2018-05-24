from glob import glob
import os
import random

def split_dataset(training_dir, ratio):

    list_classes = os.listdir(training_dir)
    dic_train_ids = {}
    dic_valid_ids = {}
    for c in list_classes:
        c_path = os.path.join(training_dir, c)
        list_img = glob(c_path+'/*.tif')
        list_img = [os.path.basename(img) for img in list_img]
        train_ids = random.sample(list_img, int(ratio*len(list_img)))
        valid_ids = list(set(list_img) - set(train_ids))
        dic_train_ids[c] = [train_ids]
        dic_valid_ids[c] = [valid_ids]
    return dic_train_ids, dic_valid_ids
