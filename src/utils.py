import numpy as np


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