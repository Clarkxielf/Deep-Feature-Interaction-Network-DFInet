"""Data loader"""


import argparse
import logging
import os
import numpy as np
import torchvision
import glob
import scipy.io as sio
from torch.utils.data import Dataset

import DFInet_2.src.data_loader.transforms as Transforms


_logger = logging.getLogger()


def get_train_datasets(args: argparse.Namespace):

    train_transforms, val_transforms = get_transforms(args.num_points)

    _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.data_type == 'mat':
        train_data = DataMat(args.dataset_path, dataset=args.dataset, subset='train', transform=train_transforms)
        val_data = DataMat(args.dataset_path, dataset=args.dataset, subset='test', transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):

    _, test_transforms = get_transforms(args.num_points)

    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.data_type == 'mat':
        test_data = DataMat(args.dataset_path, dataset=args.dataset, subset='test', transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(num_points: int = 128):
    """Get the list of transformation to be used for training or evaluating RegistrationNet

    Args:
        num_points: Number of points to uniformly resample to.
        Note that this is with respect to the full point cloud.

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
    train_transforms = [
                        # Transforms.Resampler(num_points),
                        Transforms.RandomJitter(),
                        # Transforms.ShufflePoints()
                        ]

    test_transforms = [Transforms.SetDeterministic(),
                       # Transforms.Resampler(num_points),
                       # Transforms.RandomJitter(),
                       # Transforms.ShufflePoints()
                        ]

    return train_transforms, test_transforms


class DataMat(Dataset):
    def __init__(self, dataset_path: str, dataset: str, subset: str = 'train', transform=None):
        """Dataset from ../data/blade2_data_2D.py

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path

        self._logger.info('Loading data from {} for {}'.format(dataset_path+'/'+dataset, subset))

        mat_filelist = []
        for filename in glob.glob(os.path.join(dataset_path, dataset).format(subset)):
            mat_filelist.append(filename)

        self._src, self._ref, self._R, self._T, self._transform_gt = self._read_mat_files(mat_filelist)
        self._transform = transform
        self._logger.info('Loaded {} {} instances.'.format(self._src.shape[0], subset))

    def __getitem__(self, item):
        sample = {'points_src': self._src[item],
                  'points_ref': self._ref[item],
                  'R': self._R[item],
                  'T': self._T[item],
                  'transform_gt': self._transform_gt[item],
                  'idx': np.array(item, dtype=np.int32)}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._src.shape[0]

    @staticmethod
    def _read_mat_files(mat_fnames):

        all_src = []
        all_ref = []
        all_R = []
        all_T = []

        all_transform_gt = []
        for fname in mat_fnames:
            f = sio.loadmat(fname)
            src = f['Src']
            ref = f['Ref']
            R = f['Rot_mat']
            T = f['Translation']

            transform_gt = np.concatenate([R, T[..., None]], axis=-1)

            all_src.append(src)
            all_ref.append(ref)
            all_R.append(R)
            all_T.append(T)

            all_transform_gt.append(transform_gt)

        all_src = np.concatenate(all_src, axis=0).astype(np.float32)
        all_ref = np.concatenate(all_ref, axis=0).astype(np.float32)
        all_R = np.concatenate(all_R, axis=0).astype(np.float32)
        all_T = np.concatenate(all_T, axis=0).astype(np.float32)

        all_transform_gt = np.concatenate(all_transform_gt, axis=0).astype(np.float32)

        return all_src, all_ref, all_R, all_T, all_transform_gt
