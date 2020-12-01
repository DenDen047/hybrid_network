import numpy as np
from typing import Any, Callable, Optional, Tuple, List
import torch
from torchvision import datasets

from snn_lib.data_loaders import get_rand_transform


def bhwc2bchw(x):
    return x.permute(0, 3, 1, 2)

def np_hwc2chw(x):
    return np.transpose(x, (2, 0, 1))

def add_time_dim(x, length: int):
    img_shape = x.shape[1:] # the shape of x without batch_size
    if len(img_shape) == 1:
        x = x[:, None, :].repeat(1, length, 1)
        return x.permute(0, 2, 1)
    elif len(img_shape) == 2:
        x = x[:, None, :, :]
    elif len(img_shape) == 3:
        x = x.permute(0, 3, 1, 2)
    return x.repeat(length, 1, 1, 1, 1).permute(1, 2, 3, 4, 0)


def load_dataset(
    dataset_name: str,
    transform=None,
    size_ratio: List[float] = [0.6, 0.2, 0.2]
) -> (Any, Any, Any):
    # checking the arguments
    assert sum(size_ratio) == 1.0 and len(size_ratio) == 3

    # load training/test dataset with torchvision
    train_set = eval(f'datasets.{dataset_name}')(
        root='/dataset',
        train=True,
        download=True,
        transform=None
    )
    test_set = eval(f'datasets.{dataset_name}')(
        root='/dataset',
        train=False,
        download=True,
        transform=None
    )
    all_set = torch.utils.data.ConcatDataset([train_set, test_set])

    # split the dataset to train/val/test
    all_size = len(all_set)
    sizes = []
    for ratio in size_ratio:
        sizes.append(int(all_size * ratio))

    train_set, val_set, test_set = torch.utils.data.random_split(
        all_set,
        sizes
    )

    # define transform for train_set
    if transform is not None:
        train_set.transform = transform

    return train_set, val_set, test_set



########################### train function ###################################
def train(model, optimizer, scheduler, train_data_loader, device, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for i_batch, sample_batched in enumerate(train_data_loader):

        x_train = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        out_spike = model(x_train)

        spike_count = torch.sum(out_spike, dim=2)

        model.zero_grad()
        loss = criterion(spike_count, target.long())
        loss.backward()
        optimizer.step()

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

    # scheduler step
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()

    acc = correct_total / eval_image_number

    return acc, loss


def evaluate(model, test_data_loader, device, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    for i_batch, sample_batched in enumerate(test_data_loader):

        x_test = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        out_spike = model(x_test)

        spike_count = torch.sum(out_spike, dim=2)

        loss = criterion(spike_count, target.long())

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

    acc = correct_total / eval_image_number

    return acc, loss