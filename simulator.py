"""
This module implements all link-level simulation routines:
 - Parallel execution of experiments
 - Saving and printing the output data
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
import sys
import signal
import pickle
import traceback
import logging
from enum import IntEnum

import numpy as np
from numpy.random import SeedSequence, default_rng

from filelock import FileLock


# A global random state to be used in different processes of the multiprocessing pool
RNG_INSTANCE = None


LOGGER = logging.getLogger(__name__)


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

    def scheduling_info(self, snr_db):
        """
        Interface to scheduler
        """
        if snr_db not in self.entries:
            entry = self.data_type()
        else:
            entry = self.get_entry(snr_db)
        hit_experiment_count = entry.experiment_count() > self.max_experiments
        hit_error_count = entry.error_count() > self.max_errors
        return {
            # Entry us complete if required number of errors has been captured
            'is_complete': hit_experiment_count or hit_error_count,
            # Entry hits error probability criterion if this probability is below a required value
            'error_prob_criterion': entry.error_prob() < self.min_error_prob,
            # Error count (required for scheduler to choose an entry with minimum error count)
            'error_count': entry.error_count(),
            # Error probability (required to estimate the batch size)
            'error_prob': entry.error_prob()
        }

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


class Scheduler:
    """
    Implements a scheduling algorithm:
     - selects a batch size
     - Selects a set of points requiring additional experiments
     - Suggests a next batch ( SNR point and the number of tests) to simulate
     Scheduler assumes that the requested SNR points are sorted in increasing order, and
     the probability of error to be estimated does not increase when the SNR increases
    """

    class State(IntEnum):
        """
        Each requested SNR point has one of the following states:
        """
        # IDLE: The SNR point is not considered by scheduler. The reasons are
        #  - error probability is below a minimum requested value
        #  - no experiments were conducted for this point
        #  - the SNR point is too far from points that are currently evaluated
        IDLE = 0
        # PENDING: SNR point requites more experiments if
        #  - some number of experiments has been conducted
        #  - error probability is higher than a threshold
        #  - the number of errors is smaller than required
        #  - SNR point is not further than 'look_ahead' points (SNR points are sorted) from
        #    the last SNR point for which the probability of error is above the threshold
        #  - There are no IDLE points before any PENDING/COMPLETE point
        PENDING = 1    # SNR point is considered by scheduler and requires more experiments
        SCHEDULED = 2  # The SNR point is being evaluated by parallel pool
        COMPLETE = 3   # Sufficient number of errors has been collected

    class Entry:
        """
        Scheduling entry. Tracks the state of each SNR point
        """
        def __init__(self, errors_per_batch, chunk_size, **kwargs):
            is_complete = get_required_argument('is_complete', **kwargs)
            error_prob_criterion = get_required_argument('error_prob_criterion', **kwargs)
            if is_complete:
                self.state = Scheduler.State.COMPLETE
            elif error_prob_criterion:
                self.state = Scheduler.State.IDLE
            else:
                self.state = Scheduler.State.PENDING
            # Indicator whether a minimum probability of error was achieved
            self.hit_minimum_pe = error_prob_criterion
            # Error count required to make a scheduling decision.
            # The entry with a minimum error count is scheduled first
            self.error_count = get_required_argument('error_count', **kwargs)
            # Batch size (estimated in accordance with probability of error)
            n_cpu = multiprocessing.cpu_count()
            self.batch_size = n_cpu
            if self.error_count > 0:
                error_prob = get_required_argument('error_prob', **kwargs)
                scale = int(np.ceil(errors_per_batch / error_prob / chunk_size / n_cpu))
                self.batch_size = scale * n_cpu

        def print(self):
            """
            Nice print
            """
            if self.state == Scheduler.State.IDLE:
                state_str = 'IDLE'
            elif self.state == Scheduler.State.PENDING:
                state_str = 'PEND'
            else:
                state_str = 'DONE'
            min_error_prob_str = 'yes' if self.hit_minimum_pe else 'no '
            return f'State: {state_str}, ' +\
                f'Achieved min_error_prob: {min_error_prob_str}, ' +\
                f'{self.error_count} errors, batch size = {self.batch_size}'

        def __str__(self):
            return self.print()

    def __init__(self, **kwargs):
        self.errors_per_batch = get_required_argument('errors_per_batch', **kwargs)
        self.chunk_size = get_required_argument('chunk_size', **kwargs)
        self.look_ahead = get_required_argument('look_ahead', **kwargs)

        self.snr_range = []
        self.entries = []

    def init(self, storage, snr_range):
        """
        Initialize the schedule
        """
        self.snr_range = snr_range
        for snr_db in snr_range:
            self.entries.append(self.Entry(
                self.errors_per_batch,
                self.chunk_size,
                **storage.scheduling_info(snr_db)
            ))
        self.__update_batch_size()
        self.__update_pending()

    def pending_count(self):
        """
        Get the number of entries having PENDING state
        """
        states = np.array([entry.state for entry in self.entries])
        return np.sum(states == self.State.PENDING)

    def incomplete(self):
        """
        Return true if more experiments required
        """
        states = np.array([entry.state for entry in self.entries])
        return np.sum(states == self.State.SCHEDULED) + self.pending_count() > 0

    def notify_batch(self, snr_index, entry_kwargs):
        """
        Notify that batch has been evaluated.
        """
        # Notify batch started:
        self.entries[snr_index] = self.Entry(
            self.errors_per_batch,
            self.chunk_size,
            **entry_kwargs
        )
        self.__update_batch_size()
        self.__update_pending()

    def request_batch(self):
        """
        A batch has been req
        """
        states = np.array([entry.state for entry in self.entries])
        error_count = np.array([entry.error_count for entry in self.entries])
        idx_pending = np.argwhere(states == self.State.PENDING).reshape(-1)
        if len(idx_pending) == 0:
            return None, 0
        id_schedule = idx_pending[np.argmin(error_count[idx_pending])]
        self.entries[id_schedule].state = self.State.SCHEDULED
        return id_schedule, int(self.entries[id_schedule].batch_size)

    def __update_batch_size(self):
        """
        Batch sizes is a non-decreasing sequence.
        Set batch size as a maximum among all previous values and a current vlaue
        """
        max_batch_size = 0
        for i, _ in enumerate(self.snr_range):
            max_batch_size = max(self.entries[i].batch_size, max_batch_size)
            self.entries[i].batch_size = max_batch_size

    def __update_pending(self):
        """
        Enforces the following rules:
         - SNR point is PENDING if it is not further than 'look_ahead' points from
           the last SNR point for which the probability of error is above the threshold
         - There are no IDLE points before any PENDING/COMPLETE point
        """
        states = np.array([entry.state for entry in self.entries])
        hit_error_prob = np.array([entry.hit_minimum_pe for entry in self.entries])
        last_incomplete = np.max(np.argwhere(hit_error_prob == 0), initial=0) + self.look_ahead
        idx_idle = np.argwhere(states == self.State.IDLE)
        idx_idle = idx_idle[idx_idle <= last_incomplete]
        for i in idx_idle:
            self.entries[i].state = self.State.PENDING


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
        LOGGER.info('Creating simulator of %s.', self.experiment.get_title())

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
        # Scheduler-related variables

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
        if hasattr(experiment, 'init_worker_args'):
            LOGGER.info('Experiment requires initial arguments.')
            initargs = rng_queue, report_queue, experiment, experiment.init_worker_args()
        else:
            initargs = rng_queue, report_queue, experiment
        pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=Simulator.init_worker,
            initargs=initargs
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
        Main loop wrapped into KeybardInterrupt catcher.
        If any other exception happens, dump file will be created.
        """
        try:
            self.__run(**kwargs)
        except KeyboardInterrupt:
            LOGGER.info('Pool interrupted.')
            self.interrupted()
        except:  # Pylint W0702: bare-except. Note that the trace is logged here
            self.crashed()

    def __run(self, **kwargs):
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
        # Error rate is assumed to decrease with the SNR increase
        min_error_prob = get_required_argument('min_error_prob', **kwargs)
        self.storage.set_sim_params(
            get_required_argument('max_errors', **kwargs),
            get_required_argument('max_experiments', **kwargs),
            min_error_prob
        )
        # Arguments allowing default values:
        chunk_size = kwargs.get('chunk_size', 1)
        look_ahead = kwargs.get('look_ahead', 2)
        errors_per_batch = kwargs.get('errors_per_batch', 10)

        scheduler = Scheduler(
            max_errors=get_required_argument('max_errors', **kwargs),
            errors_per_batch=errors_per_batch,
            chunk_size=chunk_size,
            min_error_prob=min_error_prob,
            look_ahead=look_ahead
        )
        # Run simulations
        scheduler.init(self.storage, snr_range)
        scheduled_items = []
        while scheduler.incomplete():
            if len(scheduled_items) > look_ahead or scheduler.pending_count() == 0:
                # Collect data
                snr_index, async_result = scheduled_items.pop(0)
                snr_db = snr_range[snr_index]
                self.storage.update(async_result.get(), snr_db)
                LOGGER.info(self.storage.print(snr_db))
                self.storage.save()
                # Notify scheduler
                scheduler.notify_batch(snr_index, self.storage.scheduling_info(snr_db))
            snr_index, batch_size = scheduler.request_batch()
            if snr_index is None:
                continue
            LOGGER.info(
                'Schedule SNR %2.2f. Batch size %1.4e (%1.4e X %1.4e)',
                snr_range[snr_index],
                batch_size * chunk_size, batch_size, chunk_size
            )
            scheduled_items.append((
                snr_index,
                self.schedule_batch(snr_range[snr_index], int(batch_size), chunk_size)
            ))

    def schedule_batch(self, snr_db, batch_size, chunk_size):
        """
        Run single batch for given SNR value. Return multiprocessing async result
        """
        return self.pool.map_async(
            Simulator.single_run,
            [(snr_db, self.experiment, chunk_size)] * batch_size
        )

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
        snr_db, experiment, chunk_size = args
        # Use per-process global random state here (see init_pool)
        entry = experiment.run(snr_db, RNG_INSTANCE)
        if chunk_size > 1:
            data = [experiment.run(snr_db, RNG_INSTANCE) for _ in range(chunk_size - 1)]
            entry.merge(data)
        return entry

    @staticmethod
    def init_worker(rng_queue, report_queue, experiment, experiment_args=None):
        """
        Initialize pool worker
        :param rng_queue: Multiprocessing queue with seed list
        :param report_queue: Put init worker status (success or failure)
        :param experiment: Experiment class instance, see Simulator.init_pool()
        :param experiment_args: Experiment init_worker() arguments (e.g., a shared memory)
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
            if experiment_args is not None:
                experiment.init_worker(experiment_args)
            else:
                experiment.init_worker()
            report_queue.put(True)
        except:  # PEP-8: bare exception. Note that at his moment the exception is logged
            write_stacktrace(f'worker_{os.getpid()}_init_crash.txt')
            report_queue.put(False)
        report_queue.close()


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
