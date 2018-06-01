import torch
from seg import PFF
import numpy as np
import time
import os
from tqdm import tqdm
import argparse
from SPOTDataset_test import SPOTDataset_test
from scipy.sparse import csr_matrix
from torch.autograd import Variable

import torch.utils.data as data

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def compute_sparse(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_sparse(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def prediction(work_dir, img, net, net_state, seg_flag=False, ratio_pix2class=0.2,
               patch_size=65):

    state = torch.load(net_state)
    net.load_state_dict(state['params'])
    net.cuda()
    net.eval()
    tile = os.path.basename(img)
    img_name = tile.split('.')[0]

    # outputs directory
    out_dir = work_dir + '/' + img_name
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    spot = SPOTDataset_test(img, patch_size)
    spot_loader = data.DataLoader(dataset=spot, batch_size=256, shuffle=False)
    start_time = time.time()

    # file to store class probabilities
    f = open(out_dir + "/" + img_name + "_pred_pix.txt", "wb")

    print("----- classification is running ! -----")

    for _, batch in enumerate(tqdm(spot_loader)):
        spot_batch = Variable(batch)
        # forward
        spot_batch = spot_batch.cuda()
        output = net(spot_batch)
        # output = output.exp()
        output = output.data.cpu().numpy()
        np.savetxt(f, output, fmt='%.3f %.3f %.3f %.3f %.3f')
        # f.write("%.3f %.3f %.3f %.3f %.3f\n" % (
        #     output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]))

    print("Classification took: %s seconds -----" % (time.time() - start_time))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='results directory')
    parser.add_argument("-s", type=str2bool, nargs='?',
                        const=True, help="segmentation flag.")
    args = parser.parse_args()

    prediction(args.d, seg_flag=args.s)

