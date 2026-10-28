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

import os
import ctypes

import numpy as np
from scipy.special import erfc

from .compile_tools import compile_shared

# C++ implementation: compilation, linking, and execution routines
LIB_PATH = 'channel_impl.so'


def lib_compile():
    """
    Compile channel implementation
    """
    compile_shared(['channel_impl'], LIB_PATH)


def lib_args(dtype):
    """
    Get channel implementation argument list to load shared library
    """
    return [
        np.ctypeslib.ndpointer(dtype=np.uint8),  # Transmitted bits
        # Standard normal gaussian samples
        np.ctypeslib.ndpointer(dtype=dtype),
        ctypes.c_uint,                          # Block length (bits)
        ctypes.c_double,                        # Noise stddev
        np.ctypeslib.ndpointer(dtype=dtype),    # Channel LLR
        np.ctypeslib.ndpointer(dtype=dtype)     # Output statistics [BER, SER]
    ]


def load_shared_object():
    """
    Load shared library
    """
    wdir = os.path.abspath(os.path.dirname(__file__))
    lib = ctypes.CDLL(os.path.join(wdir, LIB_PATH))
    # Tanner graph initializer
    # C++ channel implementation arguments:
    impl_args = lib_args(np.float32)
    lib.run_bpsk_channel_f32.argtypes = impl_args
    lib.run_pam4_channel_f32.argtypes = impl_args
    lib.run_qpsk_channel_f32.argtypes = impl_args
    lib.run_qam16_channel_f32.argtypes = impl_args

    impl_args = lib_args(np.float64)
    lib.run_bpsk_channel_f64.argtypes = impl_args
    lib.run_pam4_channel_f64.argtypes = impl_args
    lib.run_qpsk_channel_f64.argtypes = impl_args
    lib.run_qam16_channel_f64.argtypes = impl_args
    return lib


def load_channel_impl(modulation_str, dtype):
    """
    Load channel implementation function from the shared library
    """
    chan_method = 'run_' + modulation_str.lower().replace('-', '') + '_channel_f'
    chan_method += ('64' if dtype == np.float64 else '32')
    lib = load_shared_object()
    return getattr(lib, chan_method)


class Modulation:
    """
    Main parameters of the modulation:
     - bits per symbol, bits per degree of freedom, average symbol energy
     - a function to evaluate theoretical BER values
    """

    def __init__(self, name):
        self.name = name
        if name.lower() == 'bpsk':
            self.bps        = 1  # Bits per symbol
            self.bpdof      = 1  # Bits per degree of freedom
            self.avg_energy = 1 # Avg. symbol energy assuming integer-valued QAM points
            self.get_ber    = Modulation.get_bpsk_ber
        elif name.lower() == 'qpsk':
            self.bps        = 2
            self.bpdof      = 1
            self.avg_energy = 2
            self.get_ber    = partial(Modulation.get_bpsk_ber, dof=2)
        elif name.lower() == 'pam-4':
            self.bps        = 2
            self.bpdof      = 2
            self.avg_energy = 5
            self.get_ber    = Modulation.get_pam4_ber
        elif name.lower() == 'qam-16':
            self.bps        = 4
            self.bpdof      = 2
            self.avg_energy = 10
            self.get_ber    = partial(Modulation.get_pam4_ber, dof=2)
        else:
            raise ValueError(f'Modulation {name} is not supported')

    def __str__(self):
        return self.name


    def sigma_noise(self, snr_db):
        """
        Define the noise variance based on the modulation
        """
        snr_lin = Modulation.db_to_linear(snr_db)
        return np.sqrt(self.avg_energy) / np.sqrt(snr_lin) / np.sqrt(2)

    @staticmethod
    def get_bpsk_ber(snr_db, dof=1):
        """
        Get BPSK theoretical BER
        :param snr_db: Signal-to_noise ratio (dB)
        :param dof: the number of degrees of freedom (1: BPSK, 2: QPSK)
        """
        ebno_linear = Modulation.db_to_linear(snr_db) / dof
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
        ebno_linear = Modulation.db_to_linear(snr_db) / 4 / dof
        # Calculate distance between constellation points
        dist = np.sqrt(ebno_linear * 4 / 5)
        return (3 * erfc(dist) / 4 + erfc(3 * dist) / 2 - erfc(5 * dist) / 4) / 2

    @staticmethod
    def db_to_linear(val_db):
        """
        Convert logarithmic to linear scale
        """
        return 10 ** (val_db / 10)


class AwgnQAMChannel:
    """
    Initialize the AWGN QAM channel
    :param modulation: string constant representing the modulation
                       Supported: 'BPSK', 'QPSK', 'PAM-4' 'QAM-16'
    NOTE: for PAM-4 and QAM-16 mpodulations, the use of adapter may be required,
    NOTE: especially when running simulations with all-zero codewords
    """

    def __init__(self, modulation, tx_bits_placeholder, llr_placeholder, use_adapter):
        # Load modulation parameters
        self.modulation = Modulation(modulation)
        # Check placeholders: types and length
        self.check_placeholder(tx_bits_placeholder, [np.uint8])
        self.check_placeholder(llr_placeholder, [np.float32, np.float64])
        self.tx_bits = tx_bits_placeholder
        self.llr = llr_placeholder
        self.dtype = llr_placeholder.dtype
        # Load channel implementtion function
        self.impl = load_channel_impl(modulation, self.dtype)
        # Create placeholder for Gaussian samples:
        self.gn_samples = np.zeros(
            len(self.tx_bits) // self.modulation.bpdof,
            dtype=self.dtype
        )
        # Avoid the use of adapter for modulations with symmetric error stats
        use_adapter = use_adapter and modulation.lower() in ['pam-4', 'qam-16']
        self.run = self.__run_adapter if use_adapter else self.__run


    def __run(self, snr_db, rng):
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
        self.generate_noise(rng)
        stats = np.zeros((2,), dtype=self.dtype)
        self.impl(
            self.tx_bits,
            self.gn_samples,
            len(self.tx_bits),
            self.modulation.sigma_noise(snr_db),
            self.llr,
            stats
        )
        return stats[0], stats[1]

    def __run_adapter(self, snr_db, rng):
        """
        The same as __run, with random adapter added
        """
        self.generate_noise(rng)
        # Apply adapter:
        adapter = rng.integers(2, size=self.tx_bits.shape, dtype=np.uint8)
        tx_bits = (self.tx_bits + adapter) % 2

        stats = np.zeros((2,), dtype=self.dtype)
        self.impl(
            tx_bits,  # Use local bit buffer
            self.gn_samples,
            len(self.tx_bits),
            self.modulation.sigma_noise(snr_db),
            self.llr,
            stats
        )
        self.llr[:] = (1 - 2.0 * adapter) * self.llr
        return stats[0], stats[1]

    def generate_noise(self, rng):
        """
        Fill noise placeholder with new samples
        """
        rng.standard_normal(
            size=self.gn_samples.shape,
            dtype=self.dtype,
            out=self.gn_samples)

    def check_placeholder(self, x, expected_types):
        """
        Check placeholders for bits and LLRs. Both ust be an 1-D numpy arrays of proper type
        """
        if not isinstance(x, np.ndarray):
            raise ValueError('Channel operates with np.ndarray')
        if len(x.shape) > 1:
            raise ValueError(
                'Channel operates with one-dimentional arrays only')
        if len(x) % self.modulation.bps:
            raise ValueError(
                'Transmitted sequence must contain integer number of symbols')
        if x.dtype not in expected_types:
            print(x.dtype, expected_types)
            raise ValueError(
                'Unsupported placeholder type. Got:',
                x.dtype,
                'Expected:',
                expected_types)


def random_bits(bits_placeholder, rng):
    """
    Generate a vector of random bits
    """
    bits_placeholder[:] = rng.integers(
        2, size=bits_placeholder.shape, dtype=np.uint8)


def output_ber(llr, tx_bits):
    """Fraction of hard decisions that differ from transmitted bits.

    A negative LLR is interpreted as bit 1; LLR == 0 is counted as an error.
    """
    llr = np.asarray(llr)
    tx_bits = np.asarray(tx_bits)
    if llr.shape != tx_bits.shape:
        raise ValueError('LLR and transmitted bits must have the same shape')
    if llr.size == 0:
        raise ValueError('Empty LLR vector')
    errors = (llr == 0) | ((llr < 0) != tx_bits)
    return float(np.count_nonzero(errors) / llr.size)
