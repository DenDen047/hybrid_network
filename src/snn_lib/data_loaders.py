# -*- coding: utf-8 -*-

"""
# File Name : data_loaders.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: data loaders.
"""

from torch.utils.data import Dataset, DataLoader

import numpy as np
from torchvision import transforms, utils
import torch
from pprint import pprint
from tqdm import tqdm


def get_rand_transform(transform_config):
    t1_size = transform_config['RandomResizedCrop']['size']
    t1_scale = transform_config['RandomResizedCrop']['scale']
    t1_ratio = transform_config['RandomResizedCrop']['ratio']
    t1 = transforms.RandomResizedCrop(t1_size, scale=tuple(t1_scale), ratio=tuple(t1_ratio), interpolation=2)

    t2_angle = transform_config['RandomRotation']['angle']
    t2 = transforms.RandomRotation(t2_angle, resample=False, expand=False, center=None)
    t3 = transforms.Compose([t1, t2])

    rand_transform = transforms.RandomApply(
        [t1, t2, t3],
        p=transform_config['RandomApply']['probability']
    )

    return rand_transform



def poisson_spike_train(x, length: int):
    # https://neuron.yale.edu/neuron/static/docs/neuronpython/spikeplot.html

    length = length[0]
    r = 1.0 - x

    dts = np.random.poisson(r * length, length) + 1 # r number of events per unit of time (length)
    spike_indices = np.cumsum(dts)
    spike_indices = spike_indices[spike_indices < length]

    spike_train = np.zeros((length,), dtype=np.float32)
    spike_train[spike_indices] = 1

    return spike_train


class TorchvisionDataset_Poisson_Spike(Dataset):
    """poisson dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    faltten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

        self.rng = np.random.default_rng()

        self.v_poisson_spike_train = np.vectorize(
            poisson_spike_train,
            signature=f'(),(1)->({self.length})'
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        #shape of image [h,w]
        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape

        #flatten image
        img = img.reshape(-1)

        # shape of spike_trains [h*w, length]
        spike_trains = np.zeros((len(img), self.length), dtype=np.float32)

        #extend last dimension for time, repeat image along the last dimension
        img_tile = np.expand_dims(img,1)
        img_tile = np.tile(img_tile, (1,self.length))
        rand = np.random.uniform(0,1,(len(img), self.length))
        spike_trains[np.where(img_tile > rand)] = 1

        if self.flatten == False:
            spike_trains = spike_trains.reshape([shape[0], shape[1], self.length])

        return spike_trains, self.dataset[idx][1]


class TorchvisionDataset(Dataset):
    """torchvision dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform
    """

    def __init__(
        self,
        torchvision_mnist,
        length=None,
        max_rate=1,
        flatten=False,
        image_mode: str = 'chw',
        transform=None
    ):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate
        self.image_mode = image_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape
        img_spike = None

        if self.image_mode == 'chw' and len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))

        if self.flatten == True:
            img = img.reshape(-1)

        return img, self.dataset[idx][1]


class FeatureDataset(object):
    def __init__(
        self,
        torchvision_dataset: Any,
        feature_extractor: Any,
        target_module: Any,
        device: Any,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        activation: Optional[Callable] = None,
    ):
        self.torchvision_dataset = torchvision_dataset
        self.feature_extractor = feature_extractor
        self.target_module = target_module
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.activation = activation

        self._generate_features()

    def _generate_features(self) -> None:
        self.data = []
        self.targets = []

        intermediate_info = {}
        def get_feature(name):
            def hook(model, input, output):
                intermediate_info[name] = output.detach()
            return hook

        self.target_module.register_forward_hook(get_feature('feature'))

        self.feature_extractor.eval()
        with torch.no_grad():
            for vector, target in tqdm(self.torchvision_dataset):
                if self.transform is not None:
                    vector = self.transform(vector)

                vector = torch.from_numpy(vector).unsqueeze(0).to(self.device)
                self.feature_extractor(vector)

                feature = torch.squeeze(intermediate_info['feature'])
                feature = feature if self.activation is None else self.activation(feature)
                self.data.append(feature.to(self.device))
                self.targets.append(target)

        # remove needless variables
        del self.feature_extractor

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)