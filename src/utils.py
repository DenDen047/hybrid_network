import numpy as np


def bhwc2bchw(x):
    return x.permute(0, 3, 1, 2)

def np_hwc2chw(x):
    return np.transpose(x, (2, 0, 1))