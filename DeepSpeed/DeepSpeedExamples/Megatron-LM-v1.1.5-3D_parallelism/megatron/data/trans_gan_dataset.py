# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from functools import partial
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torch.utils.data import Dataset
import glob
from megatron import get_args
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from torchvision.datasets import VisionDataset
from PIL import Image
import os.path
import io
import string
from collections.abc import Iterable
import pickle
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

class FakeGenDataset(object):
    def __init__(self, bs=None):
        args = get_args()

        # let's set a dummy num_elems
        _sim_dataset_size = int(1e6)
        self.bs = args.batch_size
        #self.data = torch.cuda.FloatTensor(np.random.normal(0, 1, (_sim_dataset_size, args.latent_dim)))
        self.data = torch.FloatTensor(np.random.normal(0, 1, (_sim_dataset_size, args.latent_dim)))

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[:self.bs, :]
