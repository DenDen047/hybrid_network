import os, sys

import torch.nn as nn
import torch.nn.functional as F

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

from typing import Any, Callable, Optional, Tuple

from . import utils


class baseline_snn(torch.nn.Module):
    def __init__(self,
        in_channels: int, size_h: int, input_w: int,
        batch_size: int,
        n_class: int,
        length: int,
        train_coefficients: bool,
        train_bias: bool,
        membrane_filter: bool,
        tau_m: int,
        tau_s: int,
        input_type: str = 'image',
    ):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.axon1 = dual_exp_iir_layer((784,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn1 = neuron_layer(784, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon2 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn2 = neuron_layer(500, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # preparing
        axon1_states = self.axon1.create_init_states()
        snn1_states = self.snn1.create_init_states()
        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()
        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        # converting
        if self.input_type == 'image':
            coding_out = utils.expand_along_time(inputs, length=self.length)
        elif self.input_type == 'spike':
            coding_out = inputs

        # snn
        axon1_out, axon1_states = self.axon1(coding_out, axon1_states)
        spike_l1, snn1_states = self.snn1(axon1_out, snn1_states)
        axon2_out, axon2_states = self.axon2(spike_l1, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)
        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class ann1_snn2(torch.nn.Module):
    def __init__(self,
        in_channels: int, size_h: int, input_w: int,
        batch_size: int,
        n_class: int,
        length: int,
        train_coefficients: bool,
        train_bias: bool,
        membrane_filter: bool,
        tau_m: int,
        tau_s: int,
        input_type: str = 'image',
    ):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.feature_module = 'mlp1'
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.axon2 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn2 = neuron_layer(500, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # preparing
        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()
        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        if self.input_type == 'image':
            # ann
            ann_out = self.sigm(inputs)
            # converting
            coding_out = utils.expand_along_time(ann_out, length=self.length)
        elif self.input_type == 'spike':
            coding_out = inputs

        # snn
        axon2_out, axon2_states = self.axon2(coding_out, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)
        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class ann2_snn1(torch.nn.Module):
    def __init__(self,
        in_channels: int, size_h: int, input_w: int,
        batch_size: int,
        n_class: int,
        length: int,
        train_coefficients: bool,
        train_bias: bool,
        membrane_filter: bool,
        tau_m: int,
        tau_s: int,
        input_type: str = 'image',
    ):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.feature_module = 'mlp2'
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.input_type = input_type

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # preparing
        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        # ann
        if self.input_type == 'image':
            # converting
            ann_out = self.sigm(inputs)
            coding_out = utils.expand_along_time(ann_out, length=self.length)
        elif self.input_type == 'spike':
            coding_out = inputs

        # snn
        axon3_out, axon3_states = self.axon3(coding_out, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class baseline_ann(torch.nn.Module):
    def __init__(self,
        in_channels: int, size_h: int, input_w: int,
        batch_size: int,
        n_class: int,
        length: int,
        train_coefficients: bool,
        train_bias: bool,
        membrane_filter: bool,
        tau_m: int,
        tau_s: int,
    ):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.mlp1 = nn.Linear(in_features=784, out_features=500)
        self.act1 = nn.ReLU()

        self.mlp2 = nn.Linear(in_features=500, out_features=500)
        self.act2 = nn.ReLU()

        self.mlp3 = nn.Linear(in_features=500, out_features=10)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        ann_l1 = self.act1(self.mlp1(inputs))
        ann_l2 = self.act2(self.mlp2(ann_l1))

        ann_l3 = self.mlp3(ann_l2)
        output = F.log_softmax(ann_l3, dim=1)

        return output


class pretrained_model(torch.nn.Module):
    def __init__(self,
        input_shape: Tuple[int],
        n_class: int,
        batch_size: int,
        train_bias: bool,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.n_class = n_class

        self.mlp1 = nn.Linear(in_features=784, out_features=500)
        self.act1 = nn.Sigmoid()

        self.mlp2 = nn.Linear(in_features=500, out_features=500)
        self.act2 = nn.Sigmoid()

        self.mlp3 = nn.Linear(in_features=500, out_features=self.n_class)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        ann_l1 = self.act1(self.mlp1(inputs))
        ann_l2 = self.act2(self.mlp2(ann_l1))
        ann_l3 = self.mlp3(ann_l2)

        output = F.log_softmax(ann_l3, dim=1)

        return output
