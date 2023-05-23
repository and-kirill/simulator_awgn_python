"""
This module implements demo functions for the simulator.
Usage (from .. directory): python3 -m simulator_awgn_python.demo
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

import logging
import numpy as np

from .simulator import DataEntry, AwgnQAMChannel
from .tools import run_all_experiments, enable_log


# This is a per-process global variable
# that keeps all pre-initialized bulky data to reduce communication overhead in parallel pool
DEMO_CHANNEL = None


class DemoExperiment:
    """
    This class represents a single experiment run.
    The method run is further taken by Simulator
    """
    def __init__(self, block_len: int, correctable_errors: int, modulation: str):
        """
        Implements a simple experiment with a code that can correct some fixed number of errors
        :param block_len: Code block length
        :param correctable_errors: the number of errors that can be corrected
        :param modulation: Modulation, supported 'qpsk' and 'bpsk'
        """
        # Modulation parameter must be present in the experiment instance if tools.py are used
        self.modulation = modulation
        self.block_len = block_len  # Block length
        self.correctable_errors = correctable_errors  # The number of errors that can be corrected

    def run(self, snr_db, rng):
        """
        This method implements a single test
        :param snr_db: signal-to-noise ratio (dB)
        :param rng: Random number generator instance
        :return: DataEntry with results of single tet
        """
        cwd = (rng.random(size=self.block_len) < 0.5).astype(np.uint8)
        [llr_channel, in_ber, in_ser] = DEMO_CHANNEL.run(cwd, snr_db, rng)
        cwd_hat = llr_channel < 0
        if np.sum(cwd_hat != cwd) <= self.correctable_errors:
            cwd_hat = cwd
        out_ber = np.mean(cwd_hat != cwd)
        # Fill the output statistics
        chs = DataEntry()
        chs.in_ber = in_ber
        chs.in_ser = in_ser
        chs.out_ber = out_ber
        chs.out_fer = out_ber > 0
        chs.n_exp = 1
        return chs

    def init_worker(self):
        """
        This method instantiates all per-worker global variables and classes
        If the software uses dynamically linked libraries, they must be loaded
        within this method.
        :return: Initialized per-process global parameters
        """
        global DEMO_CHANNEL
        DEMO_CHANNEL = AwgnQAMChannel(self.modulation)

    # Functions required for postprocessing
    def get_filename(self):
        """
        Get filename for simulated data pickle file. As different experiments
        may have different parameters, filename generation is a responsibility
        of the experiment class.
        """
        return f'demo_n{self.block_len}_t{self.correctable_errors}_{self.modulation}.pickle'

    def get_title(self):
        """
        Return a human-readable string describing the experiment. To appear in dash plot
        """
        return f'Demo. n = {self.block_len}, t = {self.correctable_errors}, {self.modulation}'


def get_experiment(**kwargs):
    """
    Function generating experiment instance from kwargs. Required for automated experiment runner
    """
    return DemoExperiment(**kwargs)


if __name__ == '__main__':
    # Write simulator output to file to track the simulation process
    enable_log(
        'simulator_awgn_python.simulator', logging.DEBUG, 'simulator.log'
    )
    # To run experiments, provide a function that instantiates the experiment,
    # and provide address to generate the URL for the live-plot.
    # To interrupt the simulation, press Ctrl+C.
    # After the simulation ends, the live-plot server will continue working
    # until Ctrl+C is pressed.
    run_all_experiments(
        get_experiment,
        address='127.0.0.1', start_port=8888, update_ms=5000
    )
