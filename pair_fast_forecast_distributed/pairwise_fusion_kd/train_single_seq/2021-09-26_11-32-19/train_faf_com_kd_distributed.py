from utils.model import forecast_lstm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import time
import sys
import argparse
import os
from shutil import copytree, copy
from utils.model import MotionNet
from utils.FaFModule import *
from utils.loss import *
from data.data_com_parallel import NuscenesDataset, CarscenesDataset
from data.config_com import Config, ConfigGlobal
from utils.mean_ap import eval_map
from tqdm import tqdm
from main_worker import main_worker


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

# def setup(rank = -1, world_size = -1):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# def cleanup():
#     dist.destroy_process_group()

# def main_worker(gpu, para_list):

# def init_process(rank, size, fn, backend='nccl'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29100'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
    parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
    parser.add_argument('--resume_teacher', default='/DATA_SSD/slren/teacher_aug_batch_4_epoch_100.pth', type=str, help='The path to the saved teacher model that is loaded to resume training')
    parser.add_argument('--kd', default=100000, type=float, help='kd_weight')
    parser.add_argument('--model_only', action='store_true', help='only load model')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--nepoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--layer', default=3, type=int, help='Communicate which layer')
    parser.add_argument('--nworker', default=0, type=int, help='Number of workers')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--log', action='store_true', help='Whether to log')
    parser.add_argument('--logpath', default='./log', help='The path to the output log file')
    parser.add_argument('--mode', default=None, help='Train/Val mode')
    parser.add_argument('--visualization', default=True, help='Visualize validation result')
    parser.add_argument('--binary', default=True, type=bool, help='Only detect car')
    parser.add_argument('--only_det', default=True, type=bool, help='Only do detection')
    parser.add_argument('--logname', default=None, type=str, help='log the detection performance')
    parser.add_argument('--forecast_num', default=4, type=int, help='How many frames do you want to use in forecast')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--ngpus_per_node', default=2, type=int)
    parser.add_argument('--gpu', default=2, type=int, help='GPU id to use.')
    parser.add_argument('--world_size', default=2, type=int, help='How many GPU to use.')


    # torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    print(args)
    config = Config('train', binary=args.binary, only_det=args.only_det)
    config_global = ConfigGlobal('train', binary=args.binary, only_det=args.only_det)
    world_size = args.world_size
    # processes = []
    # mp.set_start_method("spawn")
    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    mp.spawn(main_worker,
        args=(world_size, config, config_global, args,),
        nprocs=world_size,
        join = True)

if __name__ == "__main__":
    main()
    