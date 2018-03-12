from torch.utils.data import Dataset, DataLoader
import os


class SPOT_dataset(Dataset):

    def __init__(self, patch_ids, data_img, label_img):
        super(SPOT_dataset, self).__init__()

        self.data_files = [data_img.format(id) for id in patch_ids]
        self.label_files = [label_img.format(id) for id in patch_ids]

        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} : not a file '.format(f))