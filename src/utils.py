import numpy as np
from typing import Any, Callable, Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import sklearn.metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from snn_lib.data_loaders import get_rand_transform
from snn_lib.data_loaders import TorchvisionDataset


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

def load_datasetloader(
    dataset_name: str,
    batch_size: int,
    length: int,
    flatten: bool,
    transform=None,
    size_ratio: List[float] = [0.6, 0.2, 0.2]
) -> (Any, Any, Any):
    train_set, val_set, test_set = load_dataset(
        dataset_name=dataset_name,
        transform=transform
    )

    train_dataloader = DataLoader(
        TorchvisionDataset(train_set, max_rate=1, flatten=flatten), batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        TorchvisionDataset(val_set, max_rate=1, flatten=flatten),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        TorchvisionDataset(test_set, max_rate=1, flatten=flatten),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return train_dataloader, val_dataloader, test_dataloader



########################### train function ###################################
def train(
    model,
    optimizer,
    scheduler,
    train_data_loader,
    device=torch.device('cpu'),
    writer=None,
    mode: str = 'spike'
):
    eval_image_number = 0
    correct_total = 0
    loss_total = 0
    wrong_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for i_batch, sample_batched in enumerate(train_data_loader):

        x_train = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        out_spike = model(x_train)

        if mode == 'spike':
            spike_count = torch.sum(out_spike, dim=2)
        elif mode == 'continue':
            spike_count = out_spike

        model.zero_grad()
        loss = criterion(spike_count, target.long())
        loss_total += loss.data.cpu().numpy()
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
    loss = loss_total / eval_image_number

    return acc, loss


def evaluate(
    model,
    test_data_loader,
    device=torch.device('cpu'),
    writer=None,
    mode: str = 'spike'
):
    eval_image_number = 0
    correct_total = 0
    loss_total = 0
    wrong_total = 0

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    for i_batch, sample_batched in enumerate(test_data_loader):

        x_test = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        out_spike = model(x_test)

        if mode == 'spike':
            spike_count = torch.sum(out_spike, dim=2)
        elif mode == 'continue':
            spike_count = out_spike

        # calculate loss
        loss = criterion(spike_count, target.long())
        loss_total += loss.data.cpu().numpy()
        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

    acc = correct_total / eval_image_number
    loss = loss_total / eval_image_number

    return acc, loss


def confusion_matrix(
    model,
    test_data_loader,
    device=torch.device('cpu'),
    class_mode='CIFAR10',
    mode: str = 'spike',
    output_fpath: str = 'confusion_matrix.pdf'
):
    model.eval()

    y_pred = []
    y_true = []
    for i_batch, sample_batched in enumerate(test_data_loader):
        x_test = sample_batched[0].to(device)
        target = sample_batched[1].to(device)
        out_spike = model(x_test)

        if mode == 'spike':
            spike_count = torch.sum(out_spike, dim=2)
        elif mode == 'continue':
            spike_count = out_spike

        # calculate acc
        _, indices = torch.max(spike_count, dim=1)

        output = indices.data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = target.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # constant for classes
    if class_mode == 'CIFAR10':
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif class_mode == 'MNIST':
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Build confusion matrix
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix) *10,
        index=[i for i in classes],
        columns=[i for i in classes]
    )
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(output_fpath)