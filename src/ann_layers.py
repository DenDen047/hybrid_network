import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_Module(torch.nn.Module):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.ann_layer = func(*args, **kwargs)

    def forward(self, input_vectors, steady_state=False):
        """
        :param input_vectors: [batch, dim0 ,dim1..]
        :param  prev_states: tuple (prev_v, prev_reset)
        :return:
        """

        # unbind along last dimension
        inputs = input_vectors.unbind(dim=-1)
        outputs = []
        if steady_state:
            output = self.ann_layer(inputs[0])
            outputs = [output for _ in range(len(inputs))]
        else:
            for i in range(len(inputs)):
                output = self.ann_layer(inputs[i])
                outputs += [output]
        return torch.stack(outputs, dim=-1)