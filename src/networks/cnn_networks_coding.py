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


class ReparameterizeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.batch_size = None
        self.features = None
        self.length = None

    def reparameterize(self, mu, ln_var):
        """
        :param mu: mean from the encoder's latent space
        :param ln_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * ln_var)  # standard deviation
        eps = torch.randn_like(std) # random numbers as we need the same size from a normal distribution with mean 0 and variance 1.
        sample = mu + (eps * std)   # sampling as if coming from the input space
        return sample

    def coding(self, x):
        batch_size, c, h, w, length = x.shape
        ann_out = x.view(batch_size, 2, c//2, h, w, self.length)
        mu = ann_out[:, 0, :, :, :, :]
        ln_var = ann_out[:, 1, :, :, :, :]
        out = self.reparameterize(mu, ln_var)

        return out


class baseline_snn(torch.nn.Module):
    def __init__(self,
        in_channels: int, input_h: int, input_w: int,
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
        self.in_channels = in_channels
        self.input_h = input_h
        self.input_w = input_w
        self.n_class = n_class

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        c, h, w = in_channels, input_h, input_w
        self.axon1 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn1 = conv2d_layer(
            c, h, w,
            out_channels=32,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        c, h, w = 32, h-2, w-2
        self.axon2 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn2 = conv2d_layer(
            c, h, w,
            out_channels=32,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        c, h, w = 32, h-2, w-2
        self.axon3 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn3 = conv2d_layer(
            c, h, w,
            out_channels=64,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        c, h, w = 64, h-2, w-2
        self.axon4 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn4 = maxpooling2d_layer(
            c, h, w,
            kernel_size=2,
            stride=2,
            padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size
        )

        c, h, w = c, h//2, w//2
        self.axon5 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn5 = conv2d_layer(
            c, h, w,
            out_channels=64,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        c, h, w = c, h-2, w-2
        self.axon6 = dual_exp_iir_layer(
            (c, h, w),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn6 = maxpooling2d_layer(
            c, h, w,
            kernel_size=2,
            stride=2,
            padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size
        )

        c, h, w = c, h//2, w//2
        n = c * h * w
        self.axon7 = dual_exp_iir_layer((n,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn7 = neuron_layer(n, 512, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon8 = dual_exp_iir_layer((512,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn8 = neuron_layer(512, n_class, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        axon1_states = self.axon1.create_init_states()
        snn1_states = self.snn1.create_init_states()
        axon2_states = self.axon2.create_init_states()
        snn2_states = self.snn2.create_init_states()
        axon3_states = self.axon3.create_init_states()
        snn3_states = self.snn3.create_init_states()
        axon4_states = self.axon4.create_init_states()
        axon5_states = self.axon5.create_init_states()
        snn5_states = self.snn5.create_init_states()
        axon6_states = self.axon6.create_init_states()
        axon7_states = self.axon7.create_init_states()
        snn7_states = self.snn7.create_init_states()
        axon8_states = self.axon8.create_init_states()
        snn8_states = self.snn8.create_init_states()

        axon1_out, axon1_states = self.axon1(inputs, axon1_states)
        spike_l1, snn1_states = self.snn1(axon1_out, snn1_states)

        axon2_out, axon2_states = self.axon2(spike_l1, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)

        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        axon4_out, axon4_states = self.axon4(spike_l3, axon4_states)
        spike_l4 = self.snn4(axon4_out)

        axon5_out, axon5_states = self.axon5(spike_l4, axon5_states)
        spike_l5, snn5_states = self.snn5(axon5_out, snn5_states)

        axon6_out, axon6_states = self.axon6(spike_l5, axon6_states)
        spike_l6 = self.snn6(axon6_out)

        flatten_spike_l6 = torch.flatten(spike_l6, start_dim=1, end_dim=-2)
        axon7_out, axon7_states = self.axon7(flatten_spike_l6, axon7_states)
        spike_l7, snn7_states = self.snn7(axon7_out, snn7_states)

        axon8_out, axon8_states = self.axon8(spike_l7, axon8_states)
        spike_l8, snn8_states = self.snn8(axon8_out, snn8_states)

        return spike_l8


class ann1_coding_snn7(ReparameterizeBase):
    def __init__(self,
        in_channels: int,
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
        self.in_channels = in_channels

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.ann1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=32 * 2,
            kernel_size=3,
            bias=self.train_bias
        )
        self.relu = nn.ReLU()

        self.axon2 = dual_exp_iir_layer(
            (32, 30, 30),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn2 = conv2d_layer(
            32, 30, 30,
            out_channels=32,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        self.axon3 = dual_exp_iir_layer(
            (32, 28, 28),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn3 = conv2d_layer(
            32, 28, 28,
            out_channels=64,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        self.axon4 = dual_exp_iir_layer(
            (64, 26, 26),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn4 = maxpooling2d_layer(
            26, 26, 64,
            kernel_size=2,
            stride=2,
            padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size
        )

        self.axon5 = dual_exp_iir_layer(
            (64, 13, 13),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn5 = conv2d_layer(
            64, 13, 13,
            out_channels=64,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        self.axon6 = dual_exp_iir_layer(
            (64, 11, 11),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn6 = maxpooling2d_layer(
            11, 11, 64,
            kernel_size=2,
            stride=2,
            padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size
        )

        self.axon7 = dual_exp_iir_layer((1600,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn7 = neuron_layer(1600, 512, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon8 = dual_exp_iir_layer((512,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn8 = neuron_layer(512, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

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
        axon4_states = self.axon4.create_init_states()
        axon5_states = self.axon5.create_init_states()
        snn5_states = self.snn5.create_init_states()
        axon6_states = self.axon6.create_init_states()
        axon7_states = self.axon7.create_init_states()
        snn7_states = self.snn7.create_init_states()
        axon8_states = self.axon8.create_init_states()
        snn8_states = self.snn8.create_init_states()

        # ann
        ann1_out = self.ann1(inputs)
        ann_out = self.relu(ann1_out)

        # encoding
        expanded_ann_out = utils.expand_along_time(ann_out, length=self.length)
        coding_out = self.coding(expanded_ann_out)

        # snn
        axon2_out, axon2_states = self.axon2(coding_out, axon2_states)
        spike_l2, snn2_states = self.snn2(axon2_out, snn2_states)

        axon3_out, axon3_states = self.axon3(spike_l2, axon3_states)
        spike_l3, snn3_states = self.snn3(axon3_out, snn3_states)

        axon4_out, axon4_states = self.axon4(spike_l3, axon4_states)
        spike_l4 = self.snn4(axon4_out)

        axon5_out, axon5_states = self.axon5(spike_l4, axon5_states)
        spike_l5, snn5_states = self.snn5(axon5_out, snn5_states)

        axon6_out, axon6_states = self.axon6(spike_l5, axon6_states)
        spike_l6 = self.snn6(axon6_out)

        flatten_spike_l6 = torch.flatten(spike_l6, start_dim=1, end_dim=-2)
        axon7_out, axon7_states = self.axon7(flatten_spike_l6, axon7_states)
        spike_l7, snn7_states = self.snn7(axon7_out, snn7_states)

        axon8_out, axon8_states = self.axon8(spike_l7, axon8_states)
        spike_l8, snn8_states = self.snn8(axon8_out, snn8_states)

        return spike_l8


class ann4_coding_snn4(ReparameterizeBase):
    def __init__(self,
        in_channels: int,
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
        self.in_channels = in_channels

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.ann1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann3 = nn.Conv2d(
            in_channels=32,
            out_channels=64 * 2,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.relu = nn.ReLU()

        self.axon5 = dual_exp_iir_layer(
            (64, 13, 13),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn5 = conv2d_layer(
            64, 13, 13,
            out_channels=64,
            kernel_size=3,
            stride=1, padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size,
            tau_m=tau_m,
            train_bias=self.train_bias,
            membrane_filter=self.membrane_filter
        )

        self.axon6 = dual_exp_iir_layer(
            (64, 11, 11),
            self.length, self.batch_size, tau_m, tau_s, train_coefficients
        )
        self.snn6 = maxpooling2d_layer(
            11, 11, 64,
            kernel_size=2,
            stride=2,
            padding=0, dilation=1,
            step_num=self.length,
            batch_size=self.batch_size
        )

        self.axon7 = dual_exp_iir_layer((1600,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn7 = neuron_layer(1600, 512, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon8 = dual_exp_iir_layer((512,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn8 = neuron_layer(512, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """
        # preparing
        axon5_states = self.axon5.create_init_states()
        snn5_states = self.snn5.create_init_states()
        axon6_states = self.axon6.create_init_states()
        axon7_states = self.axon7.create_init_states()
        snn7_states = self.snn7.create_init_states()
        axon8_states = self.axon8.create_init_states()
        snn8_states = self.snn8.create_init_states()

        # ann
        ann1_out = F.relu(self.ann1(inputs))
        ann2_out = F.relu(self.ann2(ann1_out))
        ann3_out = F.relu(self.ann3(ann2_out))
        ann_out = F.relu(self.ann4(ann3_out))

        # encoding
        expanded_ann_out = utils.expand_along_time(ann_out, length=self.length)
        coding_out = self.coding(expanded_ann_out)

        # snn
        axon5_out, axon5_states = self.axon5(coding_out, axon5_states)
        spike_l5, snn5_states = self.snn5(axon5_out, snn5_states)

        axon6_out, axon6_states = self.axon6(spike_l5, axon6_states)
        spike_l6 = self.snn6(axon6_out)

        flatten_spike_l6 = torch.flatten(spike_l6, start_dim=1, end_dim=-2)
        axon7_out, axon7_states = self.axon7(flatten_spike_l6, axon7_states)
        spike_l7, snn7_states = self.snn7(axon7_out, snn7_states)

        axon8_out, axon8_states = self.axon8(spike_l7, axon8_states)
        spike_l8, snn8_states = self.snn8(axon8_out, snn8_states)

        return spike_l8


class ann6_coding_snn2(ReparameterizeBase):
    def __init__(self,
        in_channels: int,
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
        self.in_channels = in_channels

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.ann1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.ann5 = nn.Conv2d(
            in_channels=64,
            out_channels=64 * 2,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann6 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.relu = nn.ReLU()

        self.axon7 = dual_exp_iir_layer((1600,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn7 = neuron_layer(1600, 512, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

        self.axon8 = dual_exp_iir_layer((512,), self.length, self.batch_size, tau_m, tau_s, train_coefficients)
        self.snn8 = neuron_layer(512, 10, self.length, self.batch_size, tau_m, self.train_bias, self.membrane_filter)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        axon7_states = self.axon7.create_init_states()
        snn7_states = self.snn7.create_init_states()
        axon8_states = self.axon8.create_init_states()
        snn8_states = self.snn8.create_init_states()

        # ann
        ann1_out = F.relu(self.ann1(inputs))
        ann2_out = F.relu(self.ann2(ann1_out))
        ann3_out = F.relu(self.ann3(ann2_out))
        ann4_out = self.ann4(ann3_out)
        ann5_out = F.relu(self.ann5(ann4_out))
        ann_out = F.relu(self.ann6(ann5_out))

        # encoding
        expanded_ann_out = utils.expand_along_time(ann_out, length=self.length)
        coding_out = self.coding(expanded_ann_out)

        # snn
        flatten_spike_l6 = torch.flatten(coding_out, start_dim=1, end_dim=-2)
        axon7_out, axon7_states = self.axon7(flatten_spike_l6, axon7_states)
        spike_l7, snn7_states = self.snn7(axon7_out, snn7_states)

        axon8_out, axon8_states = self.axon8(spike_l7, axon8_states)
        spike_l8, snn8_states = self.snn8(axon8_out, snn8_states)

        return spike_l8


class baseline_ann(torch.nn.Module):
    def __init__(self,
        in_channels: int,
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
        self.in_channels = in_channels

        self.train_coefficients = train_coefficients
        self.train_bias = train_bias
        self.membrane_filter = membrane_filter

        self.ann1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.ann5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            bias=self.train_bias
        )

        self.ann6 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.mlp7 = nn.Linear(in_features=1600, out_features=512)

        self.mlp8 = nn.Linear(in_features=512, out_features=10)

    def forward(self, inputs):
        """
        :param inputs: [batch, input_size, t]
        :return:
        """

        ann1_out = F.relu(self.ann1(inputs))
        ann2_out = F.relu(self.ann2(ann1_out))
        ann3_out = F.relu(self.ann3(ann2_out))
        ann4_out = self.ann4(ann3_out)
        ann5_out = F.relu(self.ann5(ann4_out))
        ann6_out = self.ann6(ann5_out)

        flatten_ann6_out = torch.flatten(ann6_out, start_dim=1, end_dim=-2)
        mlp7_out = F.relu(self.mlp7(flatten_ann6_out))
        mlp8_out = self.mlp8(mlp7_out)
        output = F.log_softmax(mlp8_out, dim=1)

        return output

