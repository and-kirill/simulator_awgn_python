"""
This module represents tools to build and automate experiments
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


import os
import sys
import itertools
import json
import argparse
import signal
import logging
from functools import partial
import multiprocessing as mp
import numpy as np

from .simulator import DataEntry, DataStorage, Simulator
from .postprocessing import PostProcessing
from .live_plot import PlotServer


LOGGER = logging.getLogger(__name__)


def str2num(strnum):
    """
    Try to get integer or float from string
    """
    try:
        return int(strnum)
    except ValueError:
        return float(strnum)


def range_from_string(snr_range):
    """
    Read SNR range from different formats
    - Single number as a string
    - MATLAB-style string:
        '-10:10' -> [-10, -9, ..., 9, 10],
        '-10:0.1:10' -> [-10, -9.9, ..., 9.9., 10.0]
    - python list
    - np.array()
    """
    if isinstance(snr_range, str):
        split = snr_range.split(':')
        if len(split) == 1:
            return np.array(str2num(split[0]))
        if len(split) == 2:
            return np.arange(str2num(split[0]), str2num(split[1]) + 1)
        if len(split) == 3:
            return np.arange(
                str2num(split[0]),
                str2num(split[2]) + str2num(split[1]),
                str2num(split[1])
            )
        raise ValueError('Incorrect format of the SNR range')
    if isinstance(snr_range, list):
        snr_range = np.array(snr_range)
    if isinstance(snr_range, np.ndarray):
        if len(snr_range.shape) > 1:
            raise ValueError('SNR range must be an 1-D array')
        return snr_range
    raise ValueError('Incorrect format of the SNR range')


def enable_log(name, level=logging.DEBUG, filename=None):
    """
    Enable logging with proper formats
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if filename is None:
        handler = logging.StreamHandler(stream=sys.stdout)
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)s-%(levelname)s: %(message)s'))
    handler.setLevel(level)
    logger.addHandler(handler)


def load_config():
    """
    Parse command line arguments and load configuration file
    """
    parser = argparse.ArgumentParser(description='Simulate error-correcting codes')
    parser.add_argument('--config', help='Filename with simulation parameters')
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        print('Configuration file not found.')
        sys.exit(0)
    with open(args.config, 'r', encoding='utf-8') as file_desc:
        try:
            return json.load(file_desc)
        except json.decoder.JSONDecodeError:
            print('Invalid JSON format')
            sys.exit(1)


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
            print('Iterate through', param, ':\t', exp_params[param])
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


def simulate(experiment, snr_range, **kwargs):
    """
    Run simulations
    """
    sim_data = DataStorage(
        DataEntry,
        experiment.get_filename()
    )
    simulator = Simulator(storage=sim_data, experiment=experiment)
    # Add SNR range to run() method in addition to parameters loaded from config
    kwargs['snr_range'] = range_from_string(snr_range)
    simulator.run(**kwargs)
    simulator.close()


def run_plot_server(filename, modulation, **kwargs):
    """
    Target-function for live-plot server. Redirect stdout to log file
    """
    address = kwargs.get('address')
    update_ms = kwargs.get('update_ms')
    port = kwargs.get('port')
    title = kwargs.get('title')
    server_logfile = kwargs.get('server_logfile', 'plot_server.log')
    postproc_params = kwargs.get('postprocessing', {})

    with open(server_logfile, 'a', encoding='utf-8') as sys.stdout:
        postproc_instance = PostProcessing(
                filename=filename,
                modulation=modulation,
                **postproc_params
        )
        PlotServer(
            ip_address=address,
            port=port,
            update_ms=update_ms,
            title=title,
            postproc_instance=postproc_instance
        ).run()


def run_all_experiments(get_experiment_fcn, address='127.0.0.1', start_port=8888, update_ms=1000):
    """
    Main simulation script
    """
    # Load and expand configuration file
    sim_params = load_config()
    if 'experiment' not in sim_params:
        raise ValueError('Wrong configuration: missing experiment description')
    if 'simulation' not in sim_params:
        raise ValueError('Wrong configuration: missing simulation setup description')

    postproc_params = sim_params['postprocessing'] if 'postprocessing' in sim_params else {}
    all_experiments = expand_experiment_parameters(sim_params['experiment'])
    # Enable log for tool-scripts
    enable_log('simulator_awgn_python.tools')

    # Run all requested experiments
    server_processes = []

    # Run all experiments
    for i, exp_params in enumerate(all_experiments):
        exp_instance = get_experiment_fcn(**exp_params)
        print(f'Data: {exp_instance.get_filename()}\nURL: http://{address}:{start_port + i}')
        server_start_fcn = partial(
            run_plot_server,
            address=address,
            port=start_port + i,
            update_ms=update_ms,
            title=exp_instance.get_title(),
            postprocessing=postproc_params
        )
        plot_process = mp.Process(
            target=server_start_fcn,
            args=(exp_instance.get_filename(), exp_instance.modulation)
        )
        plot_process.start()
        server_processes.append(plot_process)
        simulate(exp_instance, **sim_params['simulation'])
    # When all experiments complete, plot server will continue working
    print('All experiments complete. Press Ctrl+C to stop plot servers.')
    # Keyboard interrupt can be captured by plot-server only
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Postprocessing (Ctrl + C pressed)
    for plot_process in server_processes:
        plot_process.join()
        print(f'Plot server {plot_process.pid} terminated.')
    for i, exp_params in enumerate(all_experiments):
        exp_instance = get_experiment_fcn(**exp_params)
        print(f'Postprocessing {exp_instance.get_filename()}')
        PostProcessing(
            filename=exp_instance.get_filename(),
            modulation=exp_instance.modulation
        ).get()
    print('Done.')
