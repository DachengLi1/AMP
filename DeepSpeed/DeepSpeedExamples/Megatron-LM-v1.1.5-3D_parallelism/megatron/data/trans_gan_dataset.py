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


class LSUNClass(VisionDataset):
    def __init__(
            self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.length


class LSUN(VisionDataset):
    """
    `LSUN <https://www.yf.io/p/lsun>`_ dataset.
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
            self,
            root: str,
            classes: Union[str, List[str]] = "church_outdoor_train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=os.path.join(root, f"{c}_lmdb"),
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes: Union[str, List[str]]) -> List[str]:
        #categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
        #              'conference_room', 'dining_room', 'kitchen',
        #              'living_room', 'restaurant', 'tower']
        #dset_opts = ['train', 'val', 'test']
        categories = ['church_outdoor']
        dset_opts = ['train', "val", "test"]
        try:
            classes = cast(str, classes)
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr_type = ("Expected type str for elements in argument classes, "
                               "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr_type.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target


    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)


class CelebA(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)
                
        # return the files list
        return files

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img
    
    
class FFHQ(Dataset):
    """ pyTorch Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = glob.glob(os.path.join(self.data_dir, "./*/*.png")) + \
                     glob.glob(os.path.join(self.data_dir, "./*.jpg")) + \
                    [y for x in os.walk(self.data_dir) for y in glob.glob(os.path.join(x[0], "*.webp"))]
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, root, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = root
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img, img


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
        #print(f"size ----------- {self.data.shape}, {self.data[idx:idx+self.bs, :].shape}")
        #print(f"data shape: {self.data.shape} {self.data[idx:idx+self.bs, :].shape}")
        return self.data[:self.bs, :]


class ImageDataset(object):
    def __init__(self, args, cur_img_size=None, bs=None):
        bs = args.dis_batch_size if bs == None else bs
        img_size = cur_img_size if args.fade_in > 0 else args.img_size
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 0
            self.train_dataset = Dt(root=args.data_path, train=True, transform=transform, download=True)
            self.val_dataset = Dt(root=args.data_path, train=False, transform=transform)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            self.test = self.valid
        
            
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            train_dataset = Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True)
            val_dataset = Dt(root=args.data_path, split='test', transform=transform)
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            else:
                train_sampler = None
                val_sampler = None
            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            self.test = self.valid
        elif args.dataset.lower() == 'celeba':
            Dt = CelebA
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            train_dataset = Dt(root=args.data_path, transform=transform)
            val_dataset = Dt(root=args.data_path, transform=transform)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            self.test = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
        elif args.dataset.lower() == 'ffhq':
            Dt = FFHQ
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            train_dataset = Dt(root=args.data_path, transform=transform)
            val_dataset = Dt(root=args.data_path, transform=transform)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.train_sampler = train_sampler
            
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            self.test = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
        elif args.dataset.lower() == 'bedroom':
            Dt = LSUN
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            train_dataset = Dt(root=args.data_path, classes=["bedroom_train"], transform=transform)
            val_dataset = Dt(root=args.data_path, classes=["bedroom_val"], transform=transform)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.train_sampler = train_sampler
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
                num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            self.test = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
        elif args.dataset.lower() == 'church':
            Dt = LSUN
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            
            self.train_dataset = Dt(root=args.data_path, classes=["church_outdoor_train"], transform=transform)
            self.val_dataset = Dt(root=args.data_path, classes=["church_outdoor_val"], transform=transform)
            
            # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            # self.train_sampler = train_sampler
            # self.train = torch.utils.data.DataLoader(
            #     train_dataset,
            #     batch_size=args.dis_batch_size, shuffle=(train_sampler is None),
            #     num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

            # self.valid = torch.utils.data.DataLoader(
            #     val_dataset,
            #     batch_size=args.dis_batch_size, shuffle=False,
            #     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

            # self.test = torch.utils.data.DataLoader(
            #     val_dataset,
            #     batch_size=args.dis_batch_size, shuffle=False,
            #     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))
