"""
Simulation settings description and JSON loader tools
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/simulator_awgn_python/.
# Copyright (c) 2023 Kirill Andreev.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import multiprocessing
import sys
import os
import dataclasses
import socket
import itertools
import numpy as np
from rich import print

from .tools import read_array, dir_exists, load_json, load_settings


HLINE_STR = '------------------------------------------------------------------------'


@dataclasses.dataclass
class SimulationSettings:
    """
    Simulation settings required by the simulator
    """
    # MATLAB-like SNR range string like '0:0.1:10'
    snr_range: str
    # Simulator keeps a unique set of SNR points.
    # To avoid floating numbers comparison, specify the SNR precision
    snr_precision: int

    # Each SNR points is simulated until (SNR point stop criterion):
    #  - Maximum number of errors has been reached
    #  - Maximum number of experiments has been reached
    max_errors: float  # Considered as float (may need for multiple access channel simulations)
    max_experiments: int  # The number of tests for each SNR point will not exceed this value

    # Simulation is performed batch-by-batch, each batch is evaluated using a multiprocessing pool.
    # Each batch consists of tasks with the same SNR value.
    # The batch size is selected in accordance with 'errors_per_batch' parameter:
    #  - batch size equals to 'errors_per_batch' divided by probability of error @SNR point
    #  - batch size from the previous SNR point is used if there is no statistics for this point
    # Total number of simultaneously scheduled batches equals to "look_ahead" parameter.
    # It is recommended to set look_ahead > 1 for better CPU load.
    #
    # NOTE: 1. Next task can start immediately after any worker in the pool has completed
    #          the previous task.
    #       2. 'look_ahead' is simultaneously the number of points to be simulated and the number
    #          of simultaneously scheduled batches.
    # Each time simulator schedules a batch for the SNR value from the scheduling list that has a
    # minimum number of errors. The last SNR point in the scheduling list is set as follows:
    #  - Find the last SNR point with error probability above 'min_error_prob' among all SNR points
    #    having positive number of collected errors.
    #  - Include next 'look_ahead' SNR points in the scheduled list.
    # Simulator will schedule all SNR points starting from the first point in the SNR range.
    # Simulation terminates when the SNR point stop criterion holds for all points
    # from the scheduling list.
    min_error_prob: float  # Minimum probability of error to be evaluated
    look_ahead: int = 2  # Num. of SNR points simulated after the 'min_error_prob' reached
    # Given the above scheduling strategy, 'errors_per_batch' makes a trade-off between how fast
    # the simulator will evaluate new points and how much statistics will be collected
    # before the simulator will include the next SNR point in the scheduling list
    errors_per_batch: float = 10
    # Maximum number of tasks in one scheduler request.
    max_tasks: int = int(1e6)
    # Specify the number of workers manually. For long codes, smaller number of workers may
    # result in higher simulated bit-rate due to CPU cache issues
    n_workers: int = multiprocessing.cpu_count()
    # Logging file. Note that crash events will be logged here.
    log_file: str = 'log_simulator.log'

    # Bitrate printing functionality
    bitrate_smooth_time: float = 120.0
    bitrate_print_period: float = 1.0

    def get_n_tasks(self, p_error, n_experiments, batch_size):
        """
        Get the number of tasks in accordance with the probability of error,
        the number of conducted experiments, and the experiment batch size.
        """
        n_tests = self.errors_per_batch / p_error
        n_remaining = max(self.max_experiments + 1 - n_experiments, 0)
        n_remaining = min(n_tests, n_remaining)
        return int(np.ceil(n_remaining / batch_size))

    def __post_init__(self):
        """
        Perform sanity checks
        """
        # Sort and merge SNR points close to each other and sort them
        snr_range = read_array(self.snr_range)
        self.snr_array = np.unique(np.sort(np.round(snr_range, self.snr_precision)))
        if len(snr_range) != len(self.snr_array):
            raise ValueError('SNR step is below SNR precision.')
        if self.snr_precision < 0 or self.snr_precision > 15:
            raise ValueError(f'Invalid value for SNR precision: {self.snr_precision}')
        if self.max_errors < 0:
            raise ValueError('Maximum number of errors must be a positive number')
        if self.min_error_prob <= 0 or self.min_error_prob >= 1:
            raise ValueError('Minimum probability threshold must be in (0, 1) interval')
        if self.look_ahead < 1:
            raise ValueError('Look ahead be a positive integer')
        dir_exists(self.log_file)

    def __str__(self):
        """
        Printing function for visual check
        """
        msg = f'SNR range:                     {self.snr_range} dB '
        msg += f'(total {len(self.snr_array)} points)\n'
        msg += f'SNR precision:                 {self.snr_precision} '
        msg += 'digits to distinguish unique points\n'
        msg += f'MAX number of errors:          {self.max_errors:1.4e}\n'
        msg += f'MAX number of experiments:     {self.max_experiments:1.4e}\n'
        msg += f'MIN probability of error:      {self.min_error_prob:1.4e}\n'
        msg += f'Scheduler \'look ahead\':        {self.look_ahead} SNR points\n'
        msg += f'Target num. of errors / batch: {self.errors_per_batch:1.3f}\n'
        msg += f'Maximum number of tasks:       {self.max_tasks}\n'
        msg += f'The number of workers:         {self.n_workers}\n'
        msg += f'Simulation log file:          \'{self.log_file}\''
        return msg


@dataclasses.dataclass
class PostprocessingSettings:
    """
    Postprocessing parameters like confidence intervals, maximum regression degree, etc
    """
    # Confidence level for error bars
    confidence_level: float = 0.95
    # Bernoulli's regression parameters
    # Ignore points above this probability of error
    # Starting from the lowest SNR, find the last point with probability of error
    # above this threshold and ignore all previous data
    pe_threshold: float = 0.95
    # Maximum regression degree
    max_degree: float = 15
    # Maximum degree should not exceed a fixed portion of points collected
    max_degree_ratio: float = 3
    # Regression type: 'polynomial' or 'spline'.
    # The second option may be preferable in the case of error-floor
    regression_type: str = 'polynomial'

    def __str__(self):
        """
        Printing function for visual check
        """
        msg = f'Confidence level for FER error bars:               {self.confidence_level:1.4f}\n'
        msg += f'Error probability upper threshold (Bernoulli fit): {self.pe_threshold:1.4f}\n'
        msg += 'Maximum regression degree:                         '
        msg += f'MIN({self.max_degree}, #points / {self.max_degree_ratio})\n'
        msg += f'Regression type (\'spline\' or \'polynomial\'):       \'{self.regression_type}\''
        return msg

    def __post_init__(self):
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError('Confidence level must be in (0, 1) interval')
        if self.pe_threshold <= 0 or self.pe_threshold >= 1:
            raise ValueError('Error probability threshold must be in (0, 1) interval')
        if self.regression_type not in ['polynomial', 'spline']:
            raise ValueError(f'Illegal value of regression type: {self.regression_type}')


@dataclasses.dataclass
class VisualizationSettings:
    """
    Plot server settings
    """
    # IP address
    ip_address: str = '127.0.0.1'
    # Starting port number. For multiple experiments, port will be allocated incrementally
    start_port: int = 8888
    # Live plot update period (ms)
    update_ms: int = 5000
    # Plot Server's stdout will be redirected to the following file:
    log_file: str = 'log_plotserver.log'

    def __post_init__(self):
        """
        Sanity checks
        """
        if self.start_port <= 0 or self.start_port > 65536:
            raise ValueError('Port number must be a positive integer below 65536')
        if self.update_ms <= 0:
            raise ValueError('Update period must be a positive integer')
        try:
            socket.inet_aton(self.ip_address)
            # legal
        except socket.error as exc:
            raise ValueError(f'Invalid IP address: {self.ip_address}') from exc
        self.url = self.get_url()
        dir_exists(self.log_file)

    def increment_port(self):
        """
        Increment port for the next experiment
        """
        self.start_port += 1
        self.url = self.get_url()

    def get_url(self):
        """
        Get URL to be printed
        """
        return f'http://{self.ip_address}:{self.start_port}'

    def __str__(self):
        msg = f'Plot server URL: {self.url} updated each {self.update_ms / 1e3:1.3f} seconds.\n'
        msg += f'Logfile:        \'{self.log_file}\''
        return msg


class Settings:
    """
    Aggregated settings
    """
    def __init__(self, filename, config_type):
        data = load_json(filename)
        # Config sections that contain non-default parameters
        required_sections = ['simulation', 'experiment']
        if not all(item in data for item in required_sections):
            msg = ''.join([f'{item} ' for item in required_sections])
            print(f'JSON does not contain one of {msg} sections')
            sys.exit(1)
        self.simulation = load_settings(SimulationSettings, data.get('simulation'))
        self.postproc = load_settings(PostprocessingSettings, data.get('postprocessing', {}))
        self.__visualization = load_settings(VisualizationSettings, data.get('visualization', {}))
        experiment_list = expand_experiment_parameters(data.get('experiment'))
        self.__all_experiments = [load_settings(config_type, config) for config in experiment_list]
        try:
            [check_experiment_config(experiment) for experiment in self.__all_experiments]
        except ValueError as exc:
            print('Illegal experiment config:', exc)
            sys.exit(1)

    def __str__(self):
        msg = '------------------------ Simulation parameters: ------------------------\n'
        msg += self.simulation.__str__()
        msg += '\n---------------------- Postprocessing parameters: ----------------------\n'
        msg += self.postproc.__str__()
        msg += '\n' + HLINE_STR + '\n'
        msg += f'Simulation plan contains {len(self.__all_experiments)} experiment(s).\n'
        msg += HLINE_STR
        return msg

    def next_experiment(self, print_exp=True, print_vis=True):
        """
        Get the next experiment from the experiment plan
        :param print_exp print experiment parameters
        :param print_vis print visualization parameters
        """
        if not self.__all_experiments:
            return None, None
        exp_settings = self.__all_experiments.pop(0)
        vis_settings = dataclasses.replace(self.__visualization)
        self.__visualization.increment_port()
        if print_exp:
            msg = '------------------------- Current experiment: --------------------------\n'
            msg += str(exp_settings)
            msg += '\n' + HLINE_STR
            if print_vis:
                msg += '\n'
                msg += str(vis_settings)
                msg += '\n' + HLINE_STR
            print(msg)
        return vis_settings, exp_settings

    def remaining(self):
        """
        Get the number of remaining experiments
        """
        return len(self.__all_experiments)


def check_experiment_config(config):
    """
    Check that experiment configuration has all required parameters:
     - title:      a string to be presented as a title of live-plot
     - filename:   a path to file where the simulated data should be saved
     - modulation: to derive theoretical channel BER (required by postprocessing)
     - inf_bits_count: to print simulated information bit-rate
    """
    required_parameters = ['filename', 'title', 'modulation', 'inf_bits_count']
    for param in required_parameters:
        if not hasattr(config, param):
            raise ValueError(f'Experiment config must have {param} attribute')
    # Check that filename has a proper extension
    # when writing a text file, this extension will be replaced with *.txt
    if len(os.path.splitext(config.filename)) <= 1:
        raise ValueError('Output files without extension are not supported')
    if os.path.splitext(config.filename)[-1] != '.pickle':
        raise ValueError('Output files must have *.pickle extension')


def expand_experiment_parameters(exp_params: dict):
    """
    If any parameter is a list, then run experiments iterating through this list
    If there are multiple list in parameters, the experiment setup is a product of this lists
    Total number of experiments will be a product of all lengths
    :param exp_params Experiment parameters
    """
    iterate_through = {}
    for param in exp_params:
        if isinstance(exp_params[param], list):
            iterate_through[param] = exp_params[param]
    if not iterate_through:
        return [exp_params]

    all_experiments = []
    param_names = list(iterate_through.keys())
    for param_tuple in itertools.product(*iterate_through.values()):
        for i, param in enumerate(param_names):
            exp_params[param] = param_tuple[i]
        all_experiments.append(exp_params.copy())
    return all_experiments
