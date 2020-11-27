import torch


def expand_along_time(x, length: int):
    """
    x: input tensor without time dim ([batch_size, ...])
    length: number of steps of snn
    """
    img_shape = x.shape[1:]

    x = torch.unsqueeze(x, -1) # add time dimension

    if len(img_shape) == 1: # [n_feature,]
        x = x.repeat(1, 1, length)  # note: you should NOT use `torch.expand`(https://qiita.com/shinochin/items/c76616f8064f5710c895)
    elif len(img_shape) == 3:   # [c, h, w]
        x = x.repeat(1, 1, 1, 1, length)

    return x