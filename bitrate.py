"""
This module implements simulated information bit-rate printing:
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

import time
import numpy as np


class BitratePrinter:
    """
    EWMA-filtered simulated bit-rate printer
    """
    def __init__(self, sim_settings, exp_settings, batch_size):
        self.last_update = time.time()
        self.last_print = 0
        self.bitrate = 0
        self.last_msg_len = 0

        self.sim_settings = sim_settings
        self.exp_settings = exp_settings
        self.batch_size = batch_size

    def print(self, n_tasks):
        """
        Main function
        """
        self.__update_bitrate(n_tasks)
        self.__print_bitrate()

    def __update_bitrate(self, n_tasks):
        """
        Derive instantaneous and save smooth bit-rate
        """
        # Take the codec-defined batch size into account
        tot_tsasks = n_tasks * self.batch_size
        # Calculate instantaneous simulated bitrate
        got_thr = tot_tsasks * self.exp_settings.inf_bits_count / (time.time() - self.last_update)

        elapsed = time.time() - self.last_update
        self.last_update = time.time()

        # Calculating smoothing coefficient (exponentially decaying with time)
        alpha = np.exp(-elapsed / self.sim_settings.bitrate_smooth_time)
        self.bitrate = alpha * self.bitrate + (1 - alpha) * got_thr

    @staticmethod
    def bitrate_to_str(bps):
        """
        Print simulated bitrate with a proper suffix
        """
        suffix_list = ['Bit/s', 'Kbit/s', 'Mbit/s', 'Gbit/s', 'Tbit/s']
        for suffix in suffix_list:
            if bps < 1024:
                return f'{bps:1.1f} {suffix}'
            bps /= 1024
        # Unlikely to drop here, keep code analyzer happy
        return f'{bps:1.1f} {suffix_list[-1]}'

    def __print_bitrate(self):
        """
        Print smoothed bit-rate
        """
        # Print simulated throughput
        if time.time() < self.last_print + self.sim_settings.bitrate_print_period:
            return
        print('\r' + ' ' * self.last_msg_len, end='')
        msg = f'\rInformation bitrate: {BitratePrinter.bitrate_to_str(self.bitrate)}'
        msg += f', {BitratePrinter.bitrate_to_str(self.bitrate / self.sim_settings.n_workers)}'
        msg += ' per process.'
        print(msg, end='')
        self.last_msg_len = len(msg)
        self.last_print = time.time()

    def stop(self):
        """
        Reset newline if any message has been printed
        """
        if self.last_msg_len > 0:
            print('')
        self.last_msg_len = 0
