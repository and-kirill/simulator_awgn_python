"""
Tool function to handle input/output data
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
import pickle
import json
import signal
import threading
import numpy as np

from contextlib import contextmanager
from filelock import FileLock


@contextmanager
def _defer_shutdown_signals():
    """
    Defer SIGINT/SIGTERM while a file lock is held.

    Otherwise an interrupt can arrive between acquiring and releasing the lock,
    leaving the process waiting for a lock it already owns during cleanup.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    deferred = []

    def defer(signum, _):
        deferred.append(signum)

    for sig in signals:
        signal.signal(sig, defer)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Re-deliver the first deferred signal with its original handler.
        if deferred:
            signal.raise_signal(deferred[0])


def load_pickle(filename):
    """
    Load pickle file using file lock
    """
    with _defer_shutdown_signals():
        with FileLock(filename + '.lock'):
            with open(filename, 'rb') as file_handle:
                return pickle.load(file_handle)


def save_pickle(filename, data):
    """
    Save pickle using file lock
    """
    with _defer_shutdown_signals():
        with FileLock(filename + '.lock'):
            with open(filename, 'wb') as file_handle:
                pickle.dump(data, file_handle, 2)


def load_json(config_file):
    """
    Load JSON config and handle errors
    """
    if not os.path.isfile(config_file):
        print(f'Configuration file {config_file} not found')
        sys.exit(1)
    try:
        with open(config_file, 'r', encoding='utf-8') as fdesc:
            return json.load(fdesc)
    except json.JSONDecodeError as exc:
        print(f'Can not load configuration file {config_file}:', exc)
        sys.exit(1)


def dir_exists(filename):
    """
    To avoid crash, directory of the file to be written must exist
    """
    dirname = os.path.split(filename)[0]
    if dirname != '' and not os.path.isdir(dirname):
        raise ValueError(f'File path contains non-existing \'{dirname}\' directory.')


def get_members(obj):
    """
    Tool function to check member attributes of the object provided by user
    :param obj: object to be checked
    :return: list of member attributes
    """
    return [a for a in dir(obj) if not callable(getattr(obj, a)) and not a.startswith("__")]


def dataclass_merge(result, other_list):
    """
    Merge a dataclass list to a single result
    """
    if result is None:
        result = other_list.pop(0)
    members = get_members(result)
    for member in members:
        val = getattr(result, member) + sum(getattr(other, member) for other in other_list)
        setattr(result, member, val)
    return result


def snr_db_str(snr_db, snr_precision):
    """
    Print SNR in accordance with selected precision
    """
    str_template = f'SNR %+2.{snr_precision}f dB'
    return str_template % snr_db


def str2num(strnum):
    """
    Try to get integer or float from string
    """
    try:
        return int(strnum)
    except ValueError:
        return float(strnum)


def read_array(snr_range):
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
        if snr_range == '':
            return None
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
        raise ValueError('Incorrect format of the range')
    if isinstance(snr_range, list):
        if not snr_range:
            return None
        snr_range = np.array(snr_range)
    if isinstance(snr_range, np.ndarray):
        if len(snr_range.shape) > 1:
            raise ValueError('Range must be an 1-D array')
        return snr_range
    raise ValueError('Incorrect format of the range')


def load_settings(typename, param_dict):
    """
    Load settings: dict->dataclass, provides error handling
    """
    try:
        settings = typename(**param_dict)
    except TypeError as exc:
        print(f'Can not load {typename.__name__}: ', exc)
        sys.exit(1)
    except ValueError as exc:
        print(f'Illegal value of parameter(s) of {typename.__name__}:', exc)
        sys.exit(1)
    return settings
