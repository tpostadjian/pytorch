import argparse, os, shutil

parser = argparse.ArgumentParser()
parser.add_argument("i", type=str, help="directory with training sub-directories")
args = parser.parse_args()

input_dir = args.i
label_dir = 'training_set/label'
data_dir = 'training_set/data'
try:
    os.makedirs(label_dir)
    os.makedirs(data_dir)
except OSError:
    pass

l_tiles = os.listdir(input_dir)

for i in range(len(l_tiles)):
    print(l_tiles[i])
    n_dst = len(os.listdir(label_dir)) + 1

    label_subdir = os.path.join(input_dir, l_tiles[i] + '/patch/label')
    data_subdir = os.path.join(input_dir, l_tiles[i] + '/patch/data')
    if os.path.lexists(label_subdir):
        label_imgs = os.listdir(label_subdir)
        data_imgs = os.listdir(data_subdir)

        if len(label_imgs) == len(data_imgs):
            for j in range(len(label_imgs)):
                label_src = os.path.join(label_subdir, label_imgs[j])
                label_dst = os.path.join(label_dir, 'label_' + str(n_dst + j) + '.tif')
                shutil.copy(label_src, label_dst)
                # os.rename(label_src, label_dst)
                data_src = os.path.join(data_subdir, data_imgs[j])
                data_dst = os.path.join(data_dir, 'data_' + str(n_dst + j) + '.tif')
                shutil.copy(data_src, data_dst)
                # os.rename(data_src, data_dst)
        else:
            raise KeyError("Mismatching data and label directories: check number of imgs in both dir")

        print(n_dst)
