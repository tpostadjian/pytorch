from glob import glob
import os
import random


def split_dataset(training_dir, ratio):

    list_classes = os.listdir(training_dir)
    dic_train_ids = {}
    dic_test_ids = {}
    n_train = n_test = 0
    for c in list_classes:
        c_path = os.path.join(training_dir, c)
        list_img = glob(c_path+'/*.tif')
        list_img = [os.path.basename(img) for img in list_img]
        train_ids = random.sample(list_img, int(ratio*len(list_img)))
        test_ids = list(set(list_img) - set(train_ids))
        # test_ids = random.sample(test_ids, int(ratio*len(list_img)))  # just for quick debugging

        dic_train_ids[c] = [train_ids]
        dic_test_ids[c] = [test_ids]
        n_train += len(dic_train_ids[c][0])
        n_test += len(dic_test_ids[c][0])

    return dic_train_ids, dic_test_ids, n_train, n_test
