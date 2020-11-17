import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
# from torchvision import get_video_backend
from torchvision import transforms
import os
from tqdm.auto import tqdm
import numpy as np
import time
# import av
import random
import subprocess

from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')



# Download HMDB51 data and splits from serre lab website
DATA_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
SPLITS_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'

ROOT_DIR = '/dataset/HMDB51'
VIDEO_DIR = os.path.join(ROOT_DIR, 'video_data')
SPLITS_DIR = os.path.join(ROOT_DIR, 'test_train_splits')

if __name__ == "__main__":
    # prepare the working directory
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for data_dir, url in [
        [VIDEO_DIR, DATA_URL],
        [SPLITS_DIR, SPLITS_URL]
    ]:
        filename = url.split('/')[-1]
        rar_fpath = os.path.join(ROOT_DIR, filename)
        data_fpath = os.path.join(data_dir, filename)

        # download files
        if not os.path.exists(rar_fpath):
            cmd = f'wget --no-check-certificate -P {ROOT_DIR} {url}'
            subprocess.run(cmd, shell=True)

        # extend rar files
        cmd = f'unrar e {rar_fpath} {data_dir}'
        subprocess.run(cmd, shell=True)

    for f in os.listdir(VIDEO_DIR):
        dir_name = f.split('.')[0]
        subprocess.run(f"mkdir -p {VIDEO_DIR}/{dir_name}", shell=True)
        subprocess.run(f"unrar e {VIDEO_DIR}/{f} {VIDEO_DIR}/{dir_name}", shell=True)



# memory footprint support libraries/code
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize

import psutil
import humanize
import os
import GPUtil as GPU

GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

# download video transformation functions from below link
! wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
import transforms as T

# set up a sample pytorch class to check if it works
class Model(nn.Module):
  def __init__(self):
      super(Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=False).children())[:-1])
      self.fc1 = nn.Linear(512, 51)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2)
      print("size after pretrained model ", out.size())
      out = torch.log_softmax(self.fc1(out), dim=1)
      return out

check = Model().cuda()
out = check(torch.randn(16, 3 , 8, 112,112).cuda())
out.size()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
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

    def update_acc(self, val, n=1):
        self.val = val/n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# define model using PyTorch
class VideoRecog_Model(nn.Module):
  def __init__(self):
      super(VideoRecog_Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=True).children())[:-1])
      self.fc1 = nn.Linear(512, 51)
      self.fc2 = nn.Linear(51, 51)
      self.dropout = nn.Dropout2d(0.3)

  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2) # output of base model is bs x 512 x 1 x 1 x 1
      out = F.relu(self.fc1(out))
      out = self.dropout(out)
      out = torch.log_softmax(self.fc2(out), dim=1)
      return out

def train(config, model, loader, optimizer, epoch):
    model.train()
    config = {}
    config['log_interval'] = 100
    correct = 0
    total_loss = 0.0
    flag = 0
    Loss, Acc = AverageMeter(), AverageMeter()
    start = time.time()
    for batch_id, data in enumerate(loader):
        data, target = data[0], data[-1]
        # print("here")

        if torch.cuda.is_available():
           data = data.cuda()
           target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        Loss.update(loss.item(), data.size(0))

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        num_corrects = pred.eq(target.view_as(pred)).sum().item()
        correct += num_corrects

        Acc.update_acc(num_corrects, data.size(0))

        if flag!= 0 and batch_id%config['log_interval'] == 0:
           print('Train Epoch: {} Batch [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.0f})%'.format(
                epoch, batch_id * len(data), len(loader.dataset),
                100. * batch_id / len(loader), Loss.avg, correct, Acc.count, 100. * Acc.avg))
        flag = 1

    #total_loss /= len(loader.dataset)
    print('Train Epoch: {} Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         epoch, Loss.avg, correct, Acc.count, 100. * Acc.avg ))
    print(f"Takes {time.time() - start}")

def test(config, model, loader, text='Validation'):
    model.eval()
    correct = 0
    total_loss = 0.0
    Loss, Acc = AverageMeter(), AverageMeter()
    with torch.no_grad():
         for batch_id, data in enumerate(loader):
             data, target = data[0], data[-1]

             if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

             output = model(data)
             loss = F.nll_loss(output, target)
             total_loss += loss.item()

             Loss.update(loss.item(), data.size(0))

             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
             num_corrects = pred.eq(target.view_as(pred)).sum().item()
             correct += num_corrects

             Acc.update_acc(num_corrects, data.size(0))

    total_loss /= len(loader.dataset)
    print(text + ' Average Loss: {:.6f} Average Accuracy: {}/{} ({:.0f})%'.format(
         Loss.avg, correct, Acc.count , 100. * Acc.avg ))

# Datasets and Dataloaders for model training ..

val_split = 0.05
num_frames = 16 # 16
clip_steps = 50
num_workers = 8
pin_memory = True
train_tfms = torchvision.transforms.Compose([
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((128, 171)),
                                 T.RandomHorizontalFlip(),
                                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((112, 112))
                               ])
test_tfms =  torchvision.transforms.Compose([
                                             T.ToFloatTensorInZeroOne(),
                                             T.Resize((128, 171)),
                                             T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                             T.CenterCrop((112, 112))
                                             ])
hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=True,
                                                transform=train_tfms, num_workers=num_workers)


hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=test_tfms, num_workers=num_workers)

total_train_samples = len(hmdb51_train)
total_val_samples = round(val_split * total_train_samples)

print(f"number of train samples {total_train_samples}")
print(f"number of validation samples {total_val_samples}")
print(f"number of test samples {len(hmdb51_test)}")

bs = 4
lr = 1e-2
gamma = 0.7
total_epochs = 10
config = {}
num_workers = 0

kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
#kwargs = {'num_workers':num_workers}
#kwargs = {}

hmdb51_train_v1, hmdb51_val_v1 = random_split(hmdb51_train, [total_train_samples - total_val_samples,
                                                                       total_val_samples])

#hmdb51_train_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_val_v1.video_clips.compute_clips(16, 1, frame_rate=30)
#hmdb51_test.video_clips.compute_clips(16, 1, frame_rate=30)

#train_sampler = RandomClipSampler(hmdb51_train_v1.video_clips, 5)
#test_sampler = UniformClipSampler(hmdb51_test.video_clips, 5)

train_loader = DataLoader(hmdb51_train_v1, batch_size=bs, shuffle=True, **kwargs)
val_loader   = DataLoader(hmdb51_val_v1, batch_size=bs, shuffle=True, **kwargs)
test_loader  = DataLoader(hmdb51_test, batch_size=bs, shuffle=False, **kwargs)

model = VideoRecog_Model()

if torch.cuda.is_available():
   model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

print("Launching Action Recognition Model training")
for epoch in range(1, total_epochs + 1):
    train(config, model, train_loader, optimizer, epoch)
    test(config, model, val_loader, text="Validation")
    scheduler.step()

test(config, model, test_loader, text="Test")

torch.cuda.is_available()

for i in DataLoader(hmdb51_train, batch_size=4):
    break