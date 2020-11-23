import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

from ann_layers import ANN_Module


class ReparameterizeBase(torch.nn.Module):
    def reparameterize(self, mu, ln_var):
        """
        :param mu: mean from the encoder's latent space
        :param ln_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * ln_var)  # standard deviation
        eps = torch.randn_like(std) # random numbers as we need the same size from a normal distribution with mean 0 and variance 1.
        sample = mu + (eps * std)   # sampling as if coming from the input space
        return sample


class ann1_poisson_snn2(ReparameterizeBase):
    def __init__(self,
        batch_size: int,
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

        self.features = 500
        self.mlp1 = ANN_Module(nn.Linear, in_features=784, out_features=self.features * 2)
        self.relu = nn.ReLU()

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

        # preprocess
        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()

        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        # ann layers
        ann_out = self.relu(self.mlp1(inputs, steady_state=True))

        # encoding
        ann_out = ann_out.view(self.batch_size, 2, self.features, self.length)
        mu = ann_out[:, 0, :, :]
        ln_var = ann_out[:, 1, :, :]
        coding_out = self.reparameterize(mu, ln_var)

        # snn layers
        axon2_out, axon2_states = self.axon2(coding_out, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)

        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class ann1_snn2(torch.nn.Module):
    def __init__(self,
        batch_size: int,
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

        self.mlp1 = ANN_Module(nn.Linear, in_features=784, out_features=500)
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.axon2 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn2 = neuron_layer(500, 500, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.dropout1 = nn.Dropout(p=0.3, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()

        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        ann_out = self.sigm(self.mlp1(inputs, steady_state=True))
        drop_1 = self.dropout1(ann_out)

        axon2_out, axon2_states = self.axon2(drop_1, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)
        drop_2 = self.dropout2(spike_l2)

        axon3_out, axon3_states = self.axon3(drop_2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class ann2_snn1(torch.nn.Module):
    def __init__(self,
        batch_size: int,
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

        self.mlp1 = ANN_Module(nn.Linear, in_features=784, out_features=500)
        self.relu1 = nn.ReLU()

        self.mlp2 = ANN_Module(nn.Linear, in_features=500, out_features=500)
        self.sigm = nn.Sigmoid()

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.axon3 = dual_exp_iir_layer((500,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn3 = neuron_layer(500, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.dropout1 = nn.Dropout(p=0.3, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()

        ann_l1 = self.relu1(self.mlp1(inputs, steady_state=True))
        drop_1 = self.dropout1(ann_l1)

        ann_l2 = self.sigm(self.mlp2(drop_1, steady_state=True))
        drop_2 = self.dropout2(ann_l2)

        axon3_out, axon3_states = self.axon3(drop_2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        return spike_l3


class baseline_ann(torch.nn.Module):
    def __init__(self,
        batch_size: int,
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

        self.mlp1 = ANN_Module(nn.Linear, in_features=784, out_features=500)
        self.relu1 = nn.ReLU()

        self.mlp2 = ANN_Module(nn.Linear, in_features=500, out_features=500)
        self.relu2 = nn.ReLU()

        self.mlp3 = ANN_Module(nn.Linear, in_features=500, out_features=10)

        self.dropout1 = nn.Dropout(p=0.3, inplace=False)
        self.dropout2 = nn.Dropout(p=0.3, inplace=False)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        ann_l1 = self.relu1(self.mlp1(inputs, steady_state=True))
        drop_1 = self.dropout1(ann_l1)

        ann_l2 = self.relu2(self.mlp2(drop_1, steady_state=True))
        drop_2 = self.dropout2(ann_l2)

        ann_l3 = self.mlp3(drop_2, steady_state=True)
        output = F.log_softmax(ann_l3, dim=1)

        return output
