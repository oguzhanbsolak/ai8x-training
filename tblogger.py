###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Loggers frontends and backends.

- TensorBoardLogger logs to files that can be read by Google's TensorBoard.
- PythonLogger and CsvLogger enhance Distiller's logger to include 1D weights.

Note that not all loggers implement all logging methods.
"""

import csv
import torch
from torch.utils.tensorboard import SummaryWriter

import distiller
# pylint: disable=no-name-in-module
from distiller.data_loggers import logger as distiller_logger
# pylint: enable=no-name-in-module
from distiller.utils import density, norm_filters, size_to_str, sparsity, sparsity_2D, to_np


class TensorBoardLogger(distiller_logger.DataLogger):
    """
    TensorBoardLogger
    """
    def __init__(self, logdir, comment=''):
        super(TensorBoardLogger, self).__init__()
        # Set the tensorboard logger
        self.writer = SummaryWriter(logdir, comment=comment)
        print('\n--------------------------------------------------------')
        print('Logging to TensorBoard - remember to execute the server:')
        print('> tensorboard --logdir=\'./logs\'\n')

        # Hard-code these preferences for now
        self.log_gradients = False  # True
        self.logged_params = ['weight']  # ['weight', 'bias']

    def log_training_progress(
            self,
            stats_dict,
            epoch,
            completed,
            total,
            freq,  # pylint: disable=unused-argument
    ):
        """log_training_progress"""
        def total_steps(total, epoch, completed):
            return total*epoch + completed

        prefix = stats_dict[0]
        stats_dict = stats_dict[1]

        for tag, value in stats_dict.items():
            self.writer.add_scalar(prefix+tag.replace(' ', '_'), value,
                                   total_steps(total, epoch, completed))

    def log_activation_statistic(self, phase, stat_name, activation_stats, epoch):
        """log_activation_statistic"""
        group = stat_name + '/activations/' + phase + "/"
        for tag, value in activation_stats.items():
            self.writer.add_scalar(group+tag, value, epoch)

    def log_weights_sparsity(self, model, epoch):
        """log_weights_sparsity"""
        params_size = 0
        sparse_params_size = 0

        for name, param in model.state_dict().items():
            if param.dim() in [2, 3, 4]:
                _density = density(param)
                params_size += torch.numel(param)
                sparse_params_size += param.numel() * _density
                self.writer.add_scalar('sparsity/weights/' + name,
                                       sparsity(param)*100, epoch)
                self.writer.add_scalar('sparsity-2D/weights/' + name,
                                       sparsity_2D(param)*100, epoch)

        self.writer.add_scalar("sparsity/weights/total",
                               100*(1 - sparse_params_size/params_size), epoch)

    def log_weights_filter_magnitude(
            self,
            model,
            epoch,
            multi_graphs=False,  # pylint: disable=unused-argument
    ):
        """Log the L1-magnitude of the weights tensors.
        """
        for name, param in model.state_dict().items():
            if param.dim() in [4]:
                self.writer.add_scalars('magnitude/filters/' + name,
                                        list(to_np(norm_filters(param))), epoch)

    def log_weights_distribution(self, named_params, steps_completed):
        """log_weights_distribution"""
        if named_params is None:
            return
        for tag, value in named_params:
            tag = tag.replace('.', '/')
            if any(substring in tag for substring in self.logged_params):
                self.writer.add_histogram(tag, to_np(value), steps_completed)
            if self.log_gradients:
                self.writer.add_histogram(tag+'/grad', to_np(value.grad), steps_completed)

    def log_model_buffers(
            self,
            model,
            buffer_names,
            tag_prefix,
            epoch,
            completed,
            total,
            freq,  # pylint: disable=unused-argument
    ):
        """Logs values of model buffers.

        Notes:
            1. Buffers are logged separately per-layer (i.e. module) within model
            2. All values in a single buffer are logged such that they will be displayed on the
               same graph in TensorBoard
            3. Similarly, if multiple buffers are provided in buffer_names, all are presented on
               the same graph.
               If this is un-desirable, call the function separately for each buffer
            4. USE WITH CAUTION: While sometimes desirable, displaying multiple distinct values in
               a single graph isn't well supported in TensorBoard. It is achieved using a
               work-around, which slows down TensorBoard loading time considerably as the number
               of distinct values increases.
               Therefore, while not limited, this function is only meant for use with a very
               limited number of buffers and/or values, e.g. 2-5.

        """
        for module_name, module in model.named_modules():
            if distiller.has_children(module):
                continue

            sd = module.state_dict()
            values = []
            for buf_name in buffer_names:
                try:
                    values += sd[buf_name].view(-1).tolist()
                except KeyError:
                    continue

            if values:
                tag = '/'.join([tag_prefix, module_name])
                self.writer.add_scalars(tag, values, total * epoch + completed)


class PythonLogger(distiller_logger.PythonLogger):
    """
    Log using Python's facilities. Enhances Distiller's class to also log 1D weights.
    """
    def log_weights_sparsity(
            self,
            model,
            epoch,  # pylint: disable=unused-argument
    ):
        """log_weights_sparsity"""
        t, total = distiller.weights_sparsity_tbl_summary(model, return_total_sparsity=True,
                                                          param_dims=[2, 3, 4])
        self.pylogger.info("\nParameters:\n" + str(t))
        self.pylogger.info('Total sparsity: {:0.2f}\n'.format(total))


class CsvLogger(distiller_logger.CsvLogger):
    """
    Log as CSV. Enhances Distiller's class to also log 1D weights.
    """
    def log_weights_sparsity(
            self,
            model,
            epoch,  # pylint: disable=unused-argument
    ):
        """log_weights_sparsity"""
        fname = self.get_fname('weights_sparsity')
        with open(fname, 'w') as csv_file:
            params_size = 0
            sparse_params_size = 0

            writer = csv.writer(csv_file)
            # write the header
            writer.writerow(['parameter', 'shape', 'volume', 'sparse volume', 'sparsity level'])

            for name, param in model.state_dict().items():
                if param.dim() in [2, 3, 4]:
                    _density = density(param)
                    params_size += torch.numel(param)
                    sparse_params_size += param.numel() * _density
                    writer.writerow([name, size_to_str(param.size()),
                                     torch.numel(param),
                                     int(_density * param.numel()),
                                     (1-_density)*100])
