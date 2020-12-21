import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

from . import utils


class baseline_snn(torch.nn.Module):
    def __init__(self,
        batch_size: int,
        length: int,
        in_channels: int,
        train_coefficients: bool,
        train_bias: bool,
        membrane_filter: bool,
        tau_m: int,
        tau_s: int,
        reset_state: bool = True,
    ):
        super().__init__()

        self.length = length
        self.batch_size = batch_size

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter
        self.reset_state = reset_state

        self.axon1 = dual_exp_iir_layer((784,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn1 = neuron_layer(784, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon2 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn2 = neuron_layer(500, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        # holding the initial states
        self.axon1_states = None
        self.snn1_states = None
        self.axon2_states = None
        self.snn2_states = None
        self.axon3_states = None
        self.snn3_states = None

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # prepare
        if self.axon1_states is None or self.reset_state:
            self.axon1_states = self.axon1.create_init_states()
        else:
            self.axon1_states = (self.axon1_states[0].detach(), self.axon1_states[1].detach())
        if self.snn1_states is None or self.reset_state:
            self.snn1_states = self.snn1.create_init_states()
        else:
            self.snn1_states = (self.snn1_states[0].detach(), self.snn1_states[1].detach())
        if self.axon2_states is None or self.reset_state:
            self.axon2_states = self.axon2.create_init_states()
        else:
            self.axon2_states = (self.axon2_states[0].detach(), self.axon2_states[1].detach())
        if self.snn2_states is None or self.reset_state:
            self.snn2_states = self.snn2.create_init_states()
        else:
            self.snn2_states = (self.snn2_states[0].detach(), self.snn2_states[1].detach())
        if self.axon3_states is None or self.reset_state:
            self.axon3_states = self.axon3.create_init_states()
        else:
            self.axon3_states = (self.axon3_states[0].detach(), self.axon3_states[1].detach())
        if self.snn3_states is None or self.reset_state:
            self.snn3_states = self.snn3.create_init_states()
        else:
            self.snn3_states = (self.snn3_states[0].detach(), self.snn3_states[1].detach())

        # converting
        coding_out = utils.expand_along_time(inputs, length=self.length)

        # snn
        axon1_out, self.axon1_states = self.axon1(coding_out, self.axon1_states)
        spike_l1, self.snn1_states = self.snn1(axon1_out, self.snn1_states)

        axon2_out, self.axon2_states = self.axon2(spike_l1, self.axon2_states)
        spike_l2, self.snn2_states = self.snn2(axon2_out, self.snn2_states)

        axon3_out, self.axon3_states = self.axon3(spike_l2, self.axon3_states)
        spike_l3, self.snn3_states = self.snn3(axon3_out, self.snn3_states)

        return spike_l3


class ann1_snn2(torch.nn.Module):
    def __init__(self,
        batch_size: int,
        length: int,
        in_channels: int,
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
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

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

        # ann
        ann_out = self.sigm(self.mlp1(inputs))

        # converting
        coding_out = utils.expand_along_time(ann_out, length=self.length)

        # snn
        axon2_out, axon2_states = self.axon2(coding_out, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)
        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class ann2_snn1(torch.nn.Module):
    def __init__(self,
        batch_size: int,
        length: int,
        in_channels: int,
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
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # prepare
        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        # ann
        ann_l1 = self.act1(self.mlp1(inputs))
        ann_out = self.sigm(self.mlp2(ann_l1))

        # converting
        coding_out = utils.expand_along_time(ann_out, length=self.length)

        axon3_out, axon3_states = self.axon3(coding_out, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class baseline_ann(torch.nn.Module):
    def __init__(self,
        batch_size: int,
        length: int,
        in_channels: int,
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
        # ann
        ann_l1 = self.act1(self.mlp1(inputs))
        ann_l2 = self.act2(self.mlp2(ann_l1))
        ann_l3 = self.mlp3(ann_l2)
        ann_out = F.log_softmax(ann_l3, dim=1)

        # converting
        coding_out = utils.expand_along_time(ann_out, length=self.length)

        return coding_out
