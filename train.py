from data_iter import DisDataIter, GenDataIter
from Discriminator import Discriminator
from Generator import Generator
from utils import recurrent_func, loss_func, get_sample, get_rewards

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import glob
import json
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

def get_params(filePath):
    with open(filePath, 'r') as f:
        params = json.load(f)
    f.close()
    return params

def get_arguments():
    train_params = get_params("./params/train_params.json")
    leak_gan_params = get_params("./params/leak_gan_params.json")
    target_params = get_params("./params/target_params.json")
    dis_data_params = get_params("./params/dis_data_params.json")
    real_data_params = get_params("./params/real_data_params.json")
    return {
        "train_params": train_params,
        "leak_gan_params": leak_gan_params,
        "target_params": target_params,
        "dis_data_params": dis_data_params,
        "real_data_params" : real_data_params
    }
"""
def prepare_model_dict(use_cuda=False):
    f = open()
"""