"""
This module implements modulation and demodulation routines for AWGN channel
Supported modulations are: BPSK, QPSK, PAM-4, QAM-16 (Gray coded)
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

from functools import partial

import numpy as np
from scipy.special import erfc


class AwgnQAMChannel:
    """
    Initialize the AWGN QAM channel
    :param modulation: string constant representing the modulation
                       Supported: 'BPSK', 'QPSK', and 'QAM-16'
    """
    def __init__(self, modulation):
        if modulation.lower() == 'bpsk':
            self.bps = 1
            self.avg_energy = 1
            self.modulate = AwgnQAMChannel.modulate_bpsk
            self.demodulate = AwgnQAMChannel.demodulate_bpsk
            self.get_ber = AwgnQAMChannel.get_bpsk_ber
        elif modulation.lower() == 'qpsk':
            self.bps = 2
            self.avg_energy = 2
            self.modulate = AwgnQAMChannel.modulate_bpsk
            self.demodulate = AwgnQAMChannel.demodulate_bpsk
            self.get_ber = partial(AwgnQAMChannel.get_bpsk_ber, dof=2)
        elif modulation.lower() == 'pam-4':
            self.bps = 2
            self.avg_energy = 5
            self.modulate = AwgnQAMChannel.modulate_pam4
            self.demodulate = AwgnQAMChannel.demodulate_pam4
            self.get_ber = AwgnQAMChannel.get_pam4_ber
        elif modulation.lower() == 'qam-16':
            self.bps = 4
            self.avg_energy = 10
            self.modulate = AwgnQAMChannel.modulate_pam4
            self.demodulate = AwgnQAMChannel.demodulate_pam4
            self.get_ber = partial(AwgnQAMChannel.get_pam4_ber, dof=2)
        else:
            raise NotImplementedError(f'Modulation {modulation} is not supported')
        self.c_size = 2 ** self.bps  # Constellation size

    def run(self, tx_bits, snr_db, rng, use_adapter=False):
        """
        Perform the modulation and demodulation routines
        Transform transmitted bits to log-likelihood ratios
        :param tx_bits: transmitted bits (1D numpy array)
        :param snr_db: signal to noise ratio (dB)
        :param rng: Random number generator instance (required by correct multiprocess randomness)
        :param use_adapter: use XOR with random bit sequence. Required if transmitting
                            zero codewords using PAM/QAM modulation
        :return: Log likelihood ratio vector of the same size as transmitted bits
        """
        # Modulation
        if not use_adapter:
            # Default: moduate transmitted bits
            tx_symb = self.modulate(tx_bits)
            adapter = None  # Keep code checker happy
        else:
            # Adapter: generate random bits and modulate XOR of input sequence and adapter
            adapter = (rng.uniform(size=tx_bits.shape) < 0.5).astype(np.uint8)
            tx_symb = self.modulate(np.mod(adapter + tx_bits, 2))

        # Noise
        snr_lin = AwgnQAMChannel.db_to_linear(snr_db)
        sigma_noise = np.sqrt(self.avg_energy) / np.sqrt(snr_lin) / np.sqrt(2)
        noise_vec = sigma_noise * rng.standard_normal(size=tx_symb.shape)

        # Demodulation
        llr_channel = self.demodulate(tx_symb + noise_vec, sigma_noise ** 2)

        if use_adapter:
            # Make LLR values correct if adapter was used
            llr_channel[adapter == 1] = -llr_channel[adapter == 1]
        # Output statistics
        cwd_hat = (llr_channel < 0).reshape(-1, self.bps)
        ber = np.mean((llr_channel < 0) != tx_bits)
        ser = np.mean(np.sum(tx_bits.reshape(-1, self.bps) != cwd_hat, axis=1) != 0)

        return llr_channel, ber, ser

    @staticmethod
    def get_bpsk_ber(snr_db, dof=1):
        """
        Get BPSK theoretical BER
        :param snr_db: Signal-to_noise ratio (dB)
        :param dof: the number of degrees of freedom (1: BPSK, 2: QPSK)
        """
        ebno_linear = AwgnQAMChannel.db_to_linear(snr_db) / dof
        return erfc(np.sqrt(ebno_linear)) / 2

    @staticmethod
    def get_pam4_ber(snr_db, dof=1):
        """
        Get PAM-4 theoretical BER, see
        "Exact BEP Analysis for Coherent M-ary PAM and QAM over AWGN
        and Rayleigh Fading Channels", doi: 10.1109/VETECS.2008.93
        :param snr_db: Signal-to_noise ratio (dB)
        :param dof: the number of degrees of freedom (1: PAM-4, 2: QAM-16)
        """
        ebno_linear = AwgnQAMChannel.db_to_linear(snr_db) / 4 / dof
        # Calculate distance between constellation points
        dist = np.sqrt(ebno_linear * 4 / 5)
        return (3 * erfc(dist) / 4 + erfc(3 * dist) / 2 - erfc(5 * dist) / 4) / 2

    @staticmethod
    def modulate_bpsk(tx_bits: np.array) -> np.array:
        """
        BPSK modulation. Note that Gray-coded QPSK can be considered
        as two independent (over I and Q channels) BPSK modulations
        """
        # Convert uint8 bits to double
        return 1 - 2 * tx_bits.astype(np.double)

    @staticmethod
    def modulate_pam4(tx_bits: np.array) -> np.array:
        """
        Gray-coded PAM-4 modulation. Note that Gray-coded QAM-16 can be considered
        as two independent (over I and Q channels) PAM-4 modulations
        """
        # Convert uint8 bits to double
        tx_bits_stacked = tx_bits.reshape(-1, 2).astype(np.double)
        tx_seq = 2 * AwgnQAMChannel.modulate_bpsk(tx_bits_stacked[:, 0])
        tx_seq += np.sign(tx_seq) * AwgnQAMChannel.modulate_bpsk(tx_bits_stacked[:, 1])
        return tx_seq

    @staticmethod
    def demodulate_bpsk(rx_symb, noise_power):
        """
        Demodulate BPSK sequence
        """
        return 2 * rx_symb / noise_power

    @staticmethod
    def demodulate_pam4(rx_symb, noise_power):
        """
        Demodulate Gray-coded PAM-4 sequence
        """
        llr_bpsk = AwgnQAMChannel.demodulate_bpsk(rx_symb, noise_power)
        offset = -4 / noise_power
        return np.vstack([
            llr_bpsk + np.log((1 + np.exp(offset + llr_bpsk)) / (1 + np.exp(offset - llr_bpsk))),
            llr_bpsk + offset + np.log((1 + np.exp(-3 * llr_bpsk)) / (1 + np.exp(-llr_bpsk)))
        ]).T.reshape(-1)

    @staticmethod
    def db_to_linear(val_db):
        """
        Convert logarithmic to linear scale
        """
        return 10 ** (val_db / 10)


if __name__ == '__main__':
    import time
    from commpy import QAMModem

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
        else:
            raise ValueError('Unsupported modulation', modulation)

        tx_symb = modem.modulate(tx_bits)
        sigma_noise = np.sqrt(1 / 10 ** (snr_db / 10)) * np.sqrt(modem.Es)

        n_noise_samples = 2 * len(tx_symb)
        noise = sigma_noise * rng.standard_normal(size=n_noise_samples, ) / np.sqrt(2)
        # Reshape the noise to match the result
        noise = -noise.reshape(-1, 2).T.reshape(-1)
        noise = noise[:len(tx_symb)] + 1j * noise[len(tx_symb):]

        return -modem.demodulate(tx_symb + noise, 'soft', sigma_noise ** 2)


    def run_awgn_channel(modulation, tx_bits, snr_db, rng):
        """
        Run channel from simulator
        """
        return AwgnQAMChannel(modulation).run(tx_bits, snr_db, rng)[0]


    def compare(modulation, snr_db):
        """
        Compare results. Evaluate bit-rate and the norm of LLR difference vector
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


    compare('qam-16', 8.0)
    compare('qpsk', 5.0)
