"""
Compare the AwgnChannel implementation with commpy channel
https://pypi.org/project/scikit-commpy/
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
from commpy import QAMModem

from .simulator import AwgnQAMChannel


# To compare LLR values with MATLAB, run the following:
# MATLAB code below:
# >> snr_db = 8; sigma_noise = sqrt(1 / 10^(snr_db / 10)); ES = 10;
# >> qamdemod(-5:5, 16, 'OutputType', 'llr', 'NoiseVariance', ES * sigma_noise^2)
# Python code providing the same result:
# >>> from commpy import QAMModem
# >>> modem = QAMModem(16)
# >>> snr_db = 8; sigma_noise = np.sqrt(1 / 10 ** (snr_db / 10)) * np.sqrt(modem.Es)
# >>> print(-modem.demodulate(np.arange(-5, 6), 'soft', sigma_noise ** 2))

def run_commpy_channel(modulation, tx_bits, snr_db, rng):
    """
    Run channel from scikit-commpy module
    """
    if modulation == 'qam-16':
        modem = QAMModem(16)
    elif modulation == 'qpsk':
        modem = QAMModem(4)

    tx_symb = modem.modulate(tx_bits)
    sigma_noise = np.sqrt(1 / 10 ** (snr_db / 10)) * np.sqrt(modem.Es)

    n_noise_samples = 2 * len(tx_symb)
    noise = sigma_noise * rng.standard_normal(size=n_noise_samples, ) / np.sqrt(2)
    # Reshape the noise to match the result
    noise = -(noise.reshape(-1, 2).T).reshape(-1)
    noise = noise[:len(tx_symb)] + 1j * noise[len(tx_symb):]

    return -modem.demodulate(tx_symb + noise, 'soft', sigma_noise ** 2)


def run_awgn_channel(modulation, tx_bits, snr_db, rng):
    """
    Run channel from simulator
    """
    return AwgnQAMChannel(modulation).run(tx_bits, snr_db, rng)[0]


def compare(modulation, snr_db):
    """
    Compare results. Evaluate bitrate and the norm of LLR difference vector
    """
    print(f'Compare LLR values for {modulation.upper()}')
    n_bits = 400000
    bits = (np.random.random(size=n_bits) < 0.5).astype(np.uint8)

    t_start = time.time()
    llr_commpy = run_commpy_channel(modulation, bits, snr_db, np.random.default_rng(seed=1))
    print('Commpy:         ', len(bits) / (time.time() - t_start) / 1e6, 'Mbit/s')

    t_start = time.time()
    llr = run_awgn_channel(modulation, bits, snr_db, np.random.default_rng(seed=1))
    print('AWGN channel:   ', len(bits) / (time.time() - t_start) / 1e6, 'Mbit/s')

    print('LLR difference: ', np.linalg.norm(llr - llr_commpy))


if __name__ == '__main__':
    # Run comparison with commpy module
    compare('qam-16', 8.0)
    compare('qpsk', 5.0)
