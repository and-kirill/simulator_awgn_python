"""
This module implements all link-level simulation routines:
 - AWGN channel with QAM modulation
 - Parallel execution of experiments
 - Saving and printing the output data
 - Advanced postprocessing with smooth BER curves and confidence intervals
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
import os
import time
import sys
import signal
import pickle
import traceback
import logging
import dataclasses
import pandas as pd
import numpy as np
from numpy.random import SeedSequence, default_rng

from scipy.optimize import minimize
from scipy.stats import binomtest
from scipy.special import erfc

from filelock import FileLock


# A global random state to be used in different processes of the multiprocessing pool
RNG_INSTANCE = None


LOGGER = logging.getLogger(__name__)


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
        elif modulation.lower() == 'qpsk':
            self.bps = 2
            self.avg_energy = 2
            self.modulate = AwgnQAMChannel.modulate_bpsk
            self.demodulate = AwgnQAMChannel.demodulate_bpsk
        elif modulation.lower() == 'qam-16':
            self.bps = 4
            self.avg_energy = 10
            self.modulate = AwgnQAMChannel.modulate_pam4
            self.demodulate = AwgnQAMChannel.demodulate_pam4
        else:
            raise NotImplementedError(f'Modulation {modulation} is not supported')
        self.c_size = 2 ** self.bps  # Constellation size

    def run(self, tx_bits, snr_db, rng):
        """
        Perform the modulation and demodulation routines
        Transform transmitted bits to log-likelihood ratios
        :param tx_bits: transmitted bits (1D numpy array)
        :param snr_db: signal to noise ratio (dB)
        :param rng: Random number generator instance (required by correct multiprocess randomness)
        :return: Log likelihood ratio vector of the same size as transmitted bits
        """
        # Modulation
        tx_symb = self.modulate(tx_bits)

        # Noise
        snr_lin = AwgnQAMChannel.db_to_linear(snr_db)
        sigma_noise = np.sqrt(self.avg_energy) / np.sqrt(snr_lin) / np.sqrt(2)
        noise_vec = sigma_noise * rng.standard_normal(size=tx_symb.shape)

        # Demodulation
        llr_channel = self.demodulate(tx_symb + noise_vec, sigma_noise ** 2)

        # Output statistics
        cwd_hat = (llr_channel < 0).reshape(-1, self.bps)
        ber = np.mean((llr_channel < 0) != tx_bits)
        ser = np.mean(np.sum(tx_bits.reshape(-1, self.bps) != cwd_hat, axis=1) != 0)

        return llr_channel, ber, ser

    def get_ber(self, snr_db):
        """
        Get theoretical BER value
        """
        ebno_linear = AwgnQAMChannel.db_to_linear(snr_db) / self.bps
        if self.bps == 1:  # BPSK
            return erfc(np.sqrt(ebno_linear)) / 2

        erfc_arg = np.sqrt(3 * self.bps * ebno_linear / (self.c_size - 1) / 2)
        erfc_scale = 2 * (1 - 1 / np.sqrt(self.c_size)) / self.bps
        return erfc_scale * erfc(erfc_arg)

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


class DataEntry:
    """
    This class keeps the simulation results for each single run of the simulator.
    To implement custom data collection technique, implement a class with the following methods
    - merge: to merge results from multiple tests
    - error_count() and error_prob() to provide stop criterion to simulator
    - print() to provide the information during the simulation process
    - output_names() and output() to save the data to text file
    """
    def __init__(self):
        # These fields keep the cumulative statistics:
        self.in_ber = 0   # Input bit error rate
        self.in_ser = 0   # Input symbol error rate
        self.out_ber = 0  # Output bit error rate
        self.out_fer = 0  # Output frame error rate
        # The final result should be divided by the experiment count:
        self.n_exp = 0

    def merge(self, other_list):
        """
        Merge results from multiple independent tests
        :param other_list: List of results
        :return: None, just updates own state
        """
        if not other_list:
            return
        members = get_members(self)
        for member in members:
            val = getattr(self, member) + sum(getattr(other, member) for other in other_list)
            setattr(self, member, val)

    def print(self):
        """
        Provide information during the simulation process
        :return: Information string with current results
        """
        data_str = f'#{self.n_exp:1.3e}, '
        if not self.n_exp:
            return data_str
        data_template = 'IN BER: %1.3e, IN SER: %1.3e, OUT BER: %1.3e, OUT FER: %1.3e'
        return data_str + data_template % tuple(self.output())

    def error_count(self):
        """
        Required for simulation stop criterion
        :return: the number of block errors
        """
        return self.out_fer

    def error_prob(self):
        """
        Required for simulation stop criterion
        :return: block error probability
        """
        if not self.n_exp:
            return 0.0
        return self.out_fer / self.n_exp

    def experiment_count(self):
        """
        Required for simulation stop criterion
        :return: the number of independent experiments conducted
        """
        return self.n_exp

    @staticmethod
    def output_names():
        """
        :return: The header string for text data
        """
        return ['INBER', 'INSER', 'BER', 'FER']

    def output(self):
        """
        :return: output variables in accordance with output_names() static method
        """
        if not self.n_exp:
            return [0] * len(DataEntry.output_names())
        # Divide cumulative statistics by the experiment count
        return [
            self.in_ber / self.n_exp,
            self.in_ser / self.n_exp,
            self.out_ber / self.n_exp,
            self.out_fer / self.n_exp
        ]


class DataStorage:
    """
    This class implements simulation results and keeps the following data:
     - Simulation parameters (which SNRs to test and how many experiments to conduct
     - Captured statistics for each simulated SNR
     - Postprocessing capabilities
     - Serialization capabilities (save and get methods)
    """
    def __init__(self, data_type, filename=None):
        """
        Initialize the data storage.
        :param data_type -- type of the DataEntry (see implementation above)
        :param filename -- Filename to store results. If None, data will not be saved
        """
        # Initialize data entry type and check that all required methods are present
        self.data_type = data_type

        # The parameters below must be set by simulator
        self.max_errors = None
        self.max_experiments = None
        self.min_error_prob = None

        # Initialize with empty data until 'link_file' method is not called
        self.filename = filename
        self.entries = {}
        if filename is None:
            LOGGER.info('Data will not be saved!')
        elif os.path.isfile(filename):
            LOGGER.info('Loading data from file.')
            self.load()
        else:
            LOGGER.info('File does not exist. Empty simulation results.')

    def set_sim_params(self, max_errors, max_experiments, min_error_prob):
        """
        Set simulation parameters to determine stop/terminate criteria
        """
        self.max_errors = max_errors
        self.max_experiments = max_experiments
        self.min_error_prob = min_error_prob

    def get_entry(self, snr_db):
        """
        Get an entry by the SNR. If does not exist, create a new one
        Note that SNR must be a somehow rounded number to avoid
        multiple closely located SNR points use np.round(SNR, #digits).
        """
        if snr_db not in self.entries:
            self.entries[snr_db] = self.data_type()
            self.entries = dict(sorted(self.entries.items()))
        return self.entries[snr_db]

    def update(self, results, snr_db):
        """
        Update corresponding entry with new data
        :param results: DataEntry() list
        :param snr_db: entry key in this storage
        :return: None, updates self
        """
        self.get_entry(snr_db).merge(results)

    def experiment_count(self, snr_db):
        """
        Get experiment count for corresponding data entry (single SNR point)
        """
        return self.get_entry(snr_db).experiment_count()

    def stop_criterion(self, snr_id):
        """
        Check simulation stop criterion for a given data entry (single SNR point)
        :param snr_id: data entry index
        :return: True is can stop simulations
        """
        data_entry = self.entries[snr_id]
        hit_experiment_count = data_entry.experiment_count() > self.max_experiments
        hit_error_count = data_entry.error_count() > self.max_errors
        return hit_experiment_count or hit_error_count

    def terminate_criterion(self, snr_id):
        """
        Check whether to stop simulations
        :return: True if can stop simulations
        """
        assert snr_id < len(self.entries)
        data_entry = self.entries[snr_id]
        if data_entry.experiment_count() == 0:
            return False
        if not self.stop_criterion(snr_id):
            return False
        if data_entry.error_prob() < self.min_error_prob:
            return True
        return False

    def print(self, snr_db):
        """
        Print the corresponding data entry statistics during simulations
        :param snr_db: SNR
        :return: human-readable string
        """
        entry = self.get_entry(snr_db)
        n_err = entry.error_count()
        data_str = entry.print()
        return f'SNR: {snr_db:+2.2f} dB, ' + data_str + f' {n_err:1.2e}/{self.max_errors} Errors'

    def load(self):
        """
        Try to get data from file. If any data mismatch, it raises a runtime error
        :return: None, just update itself or raise runtime error
        """
        data = load_pickle(self.filename)
        # Check the SNR range consistency
        for snr_db, entry_dict in data.items():
            self.entries[snr_db] = self.__entry_from_dict(entry_dict)

    def save(self):
        """
        Save data to pickle file with proper KeyboardInterrupt handling
        """
        # Avoid data corruption at save stage. Handle KeyboardInterrupt correctly
        try:
            self.__save()
        except KeyboardInterrupt:
            LOGGER.debug('Catch interrupt at save. Save again.')
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.__save()
            sys.exit(0)

    def __save(self):
        """
        Save data to file (wrapped by save public method)
        """
        data = {}
        for snr_db, entry in self.entries.items():
            if entry.experiment_count() == 0:  # Skip empty entries
                continue
            data[snr_db] = vars(entry)

        save_pickle(self.filename, data)

    def __entry_from_dict(self, entry_dict):
        """
        Create data entry instance from the dictionary
        """
        entry = self.data_type()
        for k, val in entry_dict.items():
            setattr(entry, k, val)
        return entry


class Simulator:
    """
    Implements parallel execution of multiple independent tests
    Simulator runs through the user-defined SNR range until the stop criterion is met:
     - Go to the next SNR point if the maximum error count or the maximum experiment count reached
     - Stop the simulation if hit minimum error probability or go outside the custom SNR range
    """
    def __init__(self, storage, experiment, **kwargs):
        """
        Initialize simulator with DataStorage instance and experiment instance
        Experiment is any class supporting run() and init_worker() methods
        """
        # Simulation data storage. Initialize this storage before passing to simulator
        self.storage = storage

        # Class handling single test execution. Methods run() and init_worker() must be implemented
        self.experiment = experiment

        # Initialize pool of workers
        # Triggers pylint R1732: Consider using 'with' for resource-allocating operations
        # Note that decoder initializer may take a long time (especially for long codes).
        # It is reasonable to create the pool one and close it when
        # the simulation session terminates.
        self.pool = Simulator.init_pool(experiment)
        if self.pool is None:
            LOGGER.critical('Failed to initialize parallel pool.')
            sys.exit(1)

        # SNR values will be rounded up <snr-precision> digits after floating point
        self.snr_precision = kwargs.get('snr_precision', 3)

    def __del__(self):
        try:
            self.close()
        # If called before init_pool or if called at worker initialization failure
        except AttributeError:
            pass
        LOGGER.info('Pool closed.')

    @staticmethod
    def init_pool(experiment):
        """
        Initialize the parallel pool
        """
        n_workers = multiprocessing.cpu_count()
        LOGGER.info('Initializing pool of %d workers.', n_workers)
        # Initialize the seed sequence for child processes:
        rng_queue = multiprocessing.Queue()
        report_queue = multiprocessing.Queue()
        for rng in SeedSequence().spawn(n_workers):
            rng_queue.put(rng)

        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=Simulator.init_worker,
            initargs=(rng_queue, report_queue, experiment)
        )
        rng_queue.close()
        # Try not returning non-initialized pool.
        # If any worker initialization returned an error, terminate the pool
        LOGGER.info('Waiting for initialization routines to complete...')
        for _ in range(n_workers):
            status = report_queue.get()
            if status:
                continue
            pool.terminate()
            pool.join()
            pool = None
            LOGGER.critical('Failed to initialize pool.')
            break
        report_queue.close()
        LOGGER.info('Pool created.')
        return pool

    def run(self, **kwargs):
        """
        Main loop
        """
        # Parse input arguments
        snr_range = np.sort(  # SNR values must be in the increasing order
            np.round(
                get_required_argument('snr_range', **kwargs),
                self.snr_precision  # Merge SNR points close to each other and sort them
            )
        )
        batch_size = get_required_argument('batch_size', **kwargs)
        max_errors = get_required_argument('max_errors', **kwargs)
        max_experiments = get_required_argument('max_experiments', **kwargs)
        min_error_prob = get_required_argument('min_error_prob', **kwargs)

        # Error rate is assumed to decrease with the SNR increase
        self.storage.set_sim_params(max_errors, max_experiments, min_error_prob)
        for snr_db in snr_range:
            self.simulate_snr(snr_db, batch_size)
            # Check the termination condition
            if self.storage.terminate_criterion(snr_db):
                break

    def simulate_snr(self, snr_db, batch_size):
        """
        Simulate single SNR point
        """
        # Run batches if needed
        LOGGER.debug('Simulating SNR %1.3f.', snr_db)
        LOGGER.debug(self.storage.print(snr_db))
        n_experiments = self.storage.experiment_count(snr_db)
        while not self.storage.stop_criterion(snr_db):
            LOGGER.debug('Requested batch size %d', batch_size)
            self.run_batch(snr_db, batch_size)
            self.storage.save()
        # Save data after each SNR point (only if experiment count updated)
        if n_experiments < self.storage.experiment_count(snr_db):
            self.storage.save()

    def run_batch(self, snr_db, batch_size):
        """
        Run single batch for given SNR index
        """
        try:
            t_start = time.time()
            # Use map_async. Otherwise, merging the results may become a bottleneck
            stats_array = self.pool.map_async(
                Simulator.single_run,  # Function to execute
                [(snr_db, self.experiment)] * batch_size  # Arguments repeated batch_size times
            )
            self.storage.update(stats_array.get(), snr_db)
            elapsed = time.time() - t_start
            LOGGER.info(
                self.storage.print(snr_db) + f', {elapsed:1.3f}s/{batch_size:1.4e} tests.'
            )
        except KeyboardInterrupt:
            self.interrupted()
        except:  # Pylint W0702: bare-except. Note that the trace is logged here
            self.crashed()

    # Below are pool termination routines
    def crashed(self):
        """
        Routines at pool crash: write the stacktrace and terminate the pool
        """
        write_stacktrace(f'pool_{os.getpid()}_crash.txt')
        self.interrupted()

    def interrupted(self):
        """
        Routines to handle KeyboardInterrupt during pool.map execution
        """
        try:
            LOGGER.critical('Try to terminate pool. Press Ctrl+C if stuck.')
            self.pool.terminate()
        except KeyboardInterrupt:
            self.terminate_agressive()
        self.pool.join()
        LOGGER.critical('Pool terminated.')
        sys.exit(1)

    def terminate_agressive(self):
        """
        To stop the pool immediately, one can use pool.terminate()
        Unfortunately, pool.terminate() may hang the main process:
        https://github.com/python/cpython/issues/78178
        Now terminate all sub-processes manually and use hard exit
        """
        LOGGER.critical('Terminate manually...')
        pids = [worker.pid for worker in self.pool._pool]
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        LOGGER.critical('Forced exit')
        # Exit without garbage collection and resource freeing/
        os._exit(0)

    def close(self):
        """
        Close pool after simulations
        """
        self.pool.close()
        self.pool.join()

    # Below are worker-related methods (All methods are static to avoid serialization)
    @staticmethod
    def single_run(args):
        """
        This is a single test wrapper
        In the case of crash, stacktrace will be written to file
        :param args: required arguments tuple
        """
        snr_db, experiment = args
        # Use per-process global random state here (see init_pool)
        return experiment.run(snr_db, RNG_INSTANCE)

    @staticmethod
    def init_worker(rng_queue, report_queue, experiment):
        """
        Initialize pool worker
        :param rng_queue: Multiprocessing queue with seed list
        :param report_queue: Put init worker status (success or failure)
        :param experiment: Experiment class instance, see Simulator.init_pool()
        :return: None, just updates all global per-process variables
        """
        # Disable response to keyboard interrupt within sub-processes:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Initialize random number generator instance:
        global RNG_INSTANCE
        try:
            RNG_INSTANCE = default_rng(rng_queue.get())
            rng_queue.close()
        except OSError:
            # If some worker crashed, multiprocessing.Pool() will try to re-initialize a worker
            # This re-created worker will not be able to get a proper seed value
            # Silently exit in this case
            sys.exit(1)
        try:
            experiment.init_worker()
            report_queue.put(True)
        except:  # PEP-8: bare exception. Note that at his moment the exception is logged
            write_stacktrace(f'worker_{os.getpid()}_init_crash.txt')
            report_queue.put(False)
        report_queue.close()


@dataclasses.dataclass
class PostprocessingParameters:
    """
    Postprocessing parameters like confidence intervlas, maximum regression degree, etc
    """
    # Confidence level for error bars
    confidence_level: float
    # Bernoulli's regression parameters
    # Ignore points above this probability of error
    # Starting from the lowest SNR, find the last point with probability of error
    # above this threshold and ignore all previous data
    pe_threshold: float
    # Maximum regression degree
    max_degree: float
    # Maximum degree should not exceed a fixed portion of points collected
    max_degree_ratio: float


class PostProcessing:
    """
    Generate text/pandas data frame outputs from raw simulation results
    """
    def __init__(self, **kwargs):
        # Parameters without default values
        self.modulation = kwargs.get('modulation')  # Channel modulation
        self.filename = kwargs.get('filename')

        # Parameters not subject to change
        # Confidence level for error bars
        self.params = PostprocessingParameters(
            confidence_level=kwargs.get('confidence_level', 0.95),
            pe_threshold=kwargs.get('pe_threshold', 0.96),
            max_degree=kwargs.get('max_degree', 15),
            max_degree_ratio=kwargs.get('max_degree_ratio', 3)
        )
        # Data update lock
        self.update_lock = multiprocessing.Lock()
        # Cached data access lock
        self.cache_lock = multiprocessing.Lock()

        # Cached data
        self.pickle_cache = {}
        self.data = None
        # Try to load data immediately
        self.get()

    def get(self):
        """
        Get the up-to-date pandas data frame.
        This data-frame will also be saved to txt file.
        """
        if not os.path.isfile(self.filename):
            self.__reset()
            return None

        txt_file = os.path.splitext(self.filename)[0] + '.txt'
        if os.path.isfile(txt_file):
            pickle_updated = os.path.getmtime(txt_file) < os.path.getmtime(self.filename)
            if not (pickle_updated or self.data is None):
                return self.__data_get()

        if self.update_lock.acquire(block=False):
            # Update cached data
            self.__update_pickle_cache()
            # Generate dataframe from the cached data
            data = self.__get_dataframe()
            self.update_lock.release()
            self.__data_set(data)
            return data

        # Lock was not acquired. Parallel postprocessing in progress. Return data as is.
        return self.__data_get()

    def __reset(self):
        """
        Reset cache and remove post-processed text file
        """
        with self.update_lock:
            self.pickle_cache = {}

        txt_file = os.path.splitext(self.filename)[0] + '.txt'
        with self.cache_lock:
            os.system(f'rm -f {txt_file}')
            self.data = None

    def __data_get(self) -> pd.DataFrame:
        """
        Get data with lock
        """
        with self.cache_lock:
            return self.data

    def __data_set(self, data):
        """
        Save data to file (with lock)
        """
        txt_file = os.path.splitext(self.filename)[0] + '.txt'
        with self.cache_lock:
            data.to_csv(txt_file, sep=' ', float_format='%1.6e', index=False)
            # Save dataframe for fast-return if pickle file was not changed
            self.data = data

    def __update_pickle_cache(self):
        """
        Load pickle and update cache
        """
        data_pickle = load_pickle(self.filename)
        # Update pickle cache: re-evaluate confidence intervals for updated entries
        for snr, entry in data_pickle.items():
            if entry['n_exp'] == 0:
                continue
            if snr not in self.pickle_cache or entry['n_exp'] != self.pickle_cache[snr]['n_exp']:
                self.pickle_cache[snr] = data_pickle[snr]
                # Error bars it the most expensive postprocessing procedure.
                # Do it only if the data was updated
                pe_minus, pe_plus = self.bernoulli_confidence(
                    entry['out_fer'],  # Number of errors
                    entry['n_exp']  # Number of tests
                )
                self.pickle_cache[snr]['fer_e_minus'] = pe_minus
                self.pickle_cache[snr]['fer_e_plus'] = pe_plus
        # Sort pickle cache by SNR
        self.pickle_cache = dict(sorted(self.pickle_cache.items()))

    def __get_dataframe(self):
        """
        Generate pandas Data Frame from cached data
        """
        entries = list(self.pickle_cache.values())
        snr_range = np.array([float(k) for k in list(self.pickle_cache.keys())])
        n_tests = np.array([e['n_exp'] for e in entries])
        fer_cum = np.array([e['out_fer'] for e in entries])
        ber_cum = np.array([e['out_ber'] for e in entries])

        # Create Pandas dataframe
        data = pd.DataFrame()
        # Signal-to-noise ratio range
        data['snr'] = snr_range
        # Generate frame error rate outputs with error bars
        data['fer'] = np.array([e['out_fer'] / e['n_exp'] for e in entries])
        data['fer_e_minus'] = np.array([e['fer_e_minus'] for e in entries])
        data['fer_e_plus'] = np.array([e['fer_e_plus'] for e in entries])
        data['fer_fit'] = self.get_bernoulli_fit(snr_range, fer_cum, n_tests)

        data['ber'] = np.array([e['out_ber'] / e['n_exp'] for e in entries])
        # Typically, errors in individual bits may be dependent.
        # Thus:
        # 1. One can generate fit without exact number of bits and
        #    use down-scaled values by the number of bits per single test
        # 2. Bit error rate confidence intervals may be useless
        data['ber_fit'] = self.get_bernoulli_fit(snr_range, ber_cum, n_tests)

        data['in_ber'] = np.array([e['in_ber'] / e['n_exp'] for e in entries])
        # Add theoretical BER values to immediately detect any bug from plots
        data['in_ber_ref'] = AwgnQAMChannel(self.modulation).get_ber(data['snr'].to_numpy())

        return data

    def bernoulli_confidence(self, n_errors: float, n_tests: int):
        """
        Calculate upper and lower confidence intervals for Bernoulli random variable
        """
        p_err = n_errors / n_tests
        result = binomtest(k=np.int64(np.round(n_errors)), n=n_tests, p=p_err)
        conf_interval = result.proportion_ci(
            confidence_level=self.params.confidence_level,
            method='wilson'  # Do not use 'exact' method when the number of tests is large
        )
        return p_err - conf_interval.low, conf_interval.high - p_err  # Error 'minus', error 'plus'

    def get_bernoulli_fit(self, snr_range: np.array, n_errors: np.array, n_tests: np.array):
        """
        Try multiple get_bernoulli_fit attempts with different degrees of freedom and
        output the best fit in terms of resulting loss
        :param snr_range        SNR points
        :param n_errors         The number of errors
        :param n_tests          The number of tests conducted
        """
        pe_raw = n_errors / n_tests

        # Skip points with error probabilities close to one

        idx_skip = np.argwhere(pe_raw > self.params.pe_threshold)
        skip_indx = 0
        if len(idx_skip):
            skip_indx = np.max(idx_skip) + 1

        if len(snr_range[skip_indx:]) < self.params.max_degree_ratio:
            return pe_raw
        # Estimate degree: minimum between requested degrees and a fraction of collected points
        degree = min(
            self.params.max_degree + 1,
            np.int32(np.round(len(snr_range) / self.params.max_degree_ratio))
        )
        # Perform fit
        pe_raw[skip_indx:] = BerFit(
            snr_range[skip_indx:],  # SNR points
            n_errors[skip_indx:],  # The number of errors
            n_tests[skip_indx:],  # The number of experiments
            degree  # Regression degree
        ).fit()[0]
        return pe_raw


class BerFit:
    """
    Generate smooth bit error rate curves based on the linear regression with
    Bernoulli likelihood loss function for selected regression degree
    """
    def __init__(self, snr_range, n_errors, n_tests, fit_degree):
        # Normalize the SNR range
        snr_span = np.max(snr_range) - np.min(snr_range)
        self.snr_normalized = (snr_range - np.min(snr_range)) / snr_span
        # Save number of success and errors
        self.errors = n_errors
        self.success = n_tests - n_errors
        self.n_tests = n_tests
        # Generate feature matrix
        self.feature_matrix = np.vstack([self.snr_normalized ** i for i in range(fit_degree + 1)]).T

    def fit(self, init_variance=0.01):
        """
        Apply a linear regression on log(p) with Bernoulli loss function
        """
        # Suppress 'divide by zero encountered in _binom_cdf'
        np.seterr(divide='ignore')
        features = np.hstack([
            -1,  # Zero-order coefficient must be negative to avoid infinite loss value
            init_variance * np.random.normal(size=self.feature_matrix.shape[1] - 1)
        ])
        sol = minimize(self.loss, features, jac=self.jac, method='Newton-CG')
        return np.exp(self.feature_matrix @ sol.x), sol.fun

    def loss(self, features):
        """
        Bernoulli-loss function is p^k * (1 - p) ^ (n - k) * binom(n, k)
        Note that binom (n, k) does not change as the dataset is fixed, one can ignore it
        return: negative log-likelihood
        """
        log_pe = self.feature_matrix @ features
        if np.sum(log_pe > 0):
            return np.inf
        # Calculate log(1 - p) given log(p)
        log_p1 = np.log1p(-np.exp(log_pe))
        return -np.sum(self.errors * log_pe + self.success * log_p1)

    def jac(self, features):
        """
        Loss function gradient
        """
        p_err = np.exp(self.feature_matrix @ features)
        return ((self.n_tests * p_err - self.errors) / (1 - p_err)) @ self.feature_matrix


# Below are tool-functions
def get_required_argument(arg_name, **kwargs):
    """
    Get required argument from kwargs
    """
    val = kwargs.get(arg_name)
    if val is None:
        raise TypeError(f'Required argument {arg_name}')
    return val


def get_members(obj):
    """
    Tool function to check member attributes of the object provided by user
    :param obj: object to be checked
    :return: list of member attributes
    """
    return [a for a in dir(obj) if not callable(getattr(obj, a)) and not a.startswith("__")]


def write_stacktrace(filename):
    """
    Write stacktrace to file
    """
    LOGGER.critical('Crashed. Created trace %s', filename)
    with open(filename, 'w', encoding='utf-8') as file_handle:
        file_handle.write(traceback.format_exc())


# Pickle save/load routines using FileLock

def load_pickle(filename):
    """
    Load pickle file using file lock
    """
    with FileLock(filename + '.lock'):
        with open(filename, 'rb') as file_handle:
            return pickle.load(file_handle)


def save_pickle(filename, data):
    """
    Save pickle using file lock
    """
    with FileLock(filename + '.lock'):
        with open(filename, 'wb') as file_handle:
            pickle.dump(data, file_handle, 2)
