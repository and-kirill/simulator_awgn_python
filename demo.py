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

import dataclasses
import numpy as np

from .data_storage import DataEntry
from .tools import dataclass_merge
from .worker import WorkerContext
from .channel import AwgnQAMChannel, output_ber, random_bits, lib_compile
from .simulator import run_all_experiments


@dataclasses.dataclass
class DemoExperimentConfig:
    """
    Demo experiment settings. These settings a passed to the experiment instance constructor
    """
    modulation: str  # Modulation. A string supported by AwgnQAMChannel
    block_length: int  # Code block length
    correctable_errors: int  # The number of errors that can be corrected

    def __post_init__(self):
        # Attributes below __must__ be represented by any type of experiment
        #  - title: Required by live-plot functionality to print a graph title
        self.title = self.get_title()
        #  - filename: Required by simulator to store simulation results
        self.filename = self.get_filename()
        #  - inf_its_count: The number of information bits required to print simulated bit-rate
        self.inf_bits_count = self.block_length

        # Sanity checks go here:
        if self.correctable_errors <= 0:
            raise ValueError('The number of correctable errors must be a positive integer')
        if self.block_length <= 0:
            raise ValueError('Block length must be a positive integer')

    def __str__(self):
        """
        Print to double-check the correctness of the configuration to be simulated
        """
        msg = f'Block length:          {self.block_length}\n'
        msg += f'Correctable errors:    {self.correctable_errors}\n'
        msg += f'Modulation:            {self.modulation}\n'
        msg += f'Output filename:       {self.filename}\n'
        msg += f'Plot title:            {self.title}\n'
        msg += f'Information bit count: {self.inf_bits_count} (for sim. bit-rate estimation)'
        return msg

    def get_title(self):
        """
        Get a title for live-plot tool
        """
        return f'Demo. n = {self.block_length}, t = {self.correctable_errors}, {self.modulation}'

    def get_filename(self):
        """
        Get output filename to store simulation results
        """
        return f'demo_n{self.block_length}_t{self.correctable_errors}_{self.modulation}.pickle'


class DemoExperimentInstance:
    """
    This class represents a single experiment run.
    The method run is further taken by Simulator
    """
    def __init__(self, settings):
        """
        Implements a simple experiment with a code that can correct some fixed number of errors
        :param settings are experiment settings described above
        """
        self.settings = settings
        # Prepare transmitted bits placeholder and LLR placeholder
        self.tx_bits = np.zeros((self.settings.block_length,), dtype=np.uint8)
        self.llr = np.zeros((self.settings.block_length,), dtype=np.float64)
        # Channel adapter is not required for randomly-generated codewords
        self.channel = AwgnQAMChannel(self.settings.modulation, self.tx_bits, self.llr, False)

    def prepare_workers(self, n_workers):
        return WorkerContext.spawn(n_workers)

    def init_worker(self, context):
        self.context = context

    def run(self, snr_db):
        """
        This method implements a single test
        :param snr_db: signal-to-noise ratio (dB)
        :param rng: Random number generator instance
        :return: DataEntry with results of single tet
        """
        rng = self.context.get_rng()
        random_bits(self.tx_bits, rng)
        in_ber, in_ser = self.channel.run(snr_db, rng)
        cwd_hat = self.llr < 0
        if np.sum(cwd_hat != self.tx_bits) <= self.settings.correctable_errors:
            cwd_hat = self.tx_bits
        out_ber = output_ber(1 - 2 * cwd_hat.astype(np.int32), self.tx_bits)
        # Fill the output statistics
        return DataEntry(
            in_be_cum=in_ber,
            in_se_cum=in_ser,
            be_cum=out_ber,
            fe_cum=out_ber > 0,
            tests=1
        )


    def run_batch(self, snr_db, batch_size):
        """
        Batch entry point for the simulator. The demo is cheap, so a plain
        loop is sufficient here; a vectorized experiment overrides this with
        its own batched implementation.
        """
        data = [self.run(snr_db) for _ in range(batch_size)]
        return dataclass_merge(None, data)


if __name__ == '__main__':
    # Usage: python3 -m simulator_awgn_python.demo --config=<json config>.json
    # To interrupt the simulation, press Ctrl+C.
    # After the simulation ends, the live-plot server will continue working
    # until Ctrl+C is pressed.
    lib_compile()
    run_all_experiments(DemoExperimentConfig, DemoExperimentInstance)
