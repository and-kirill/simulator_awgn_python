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

import os
import logging
import dataclasses
from enum import IntEnum
import numpy as np

from .tools import load_pickle, save_pickle, snr_db_str, dataclass_merge


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class DataEntry:
    """
    Cumulative simulation statistics for each SNR point, this is a default data entry type.
    To use postprocessing module with another data type,
    the following fields presented below are obligatory:
    """
    in_be_cum: float  # Cumulative input bit errors
    in_se_cum: float  # Cumulative input symbol errors
    be_cum:    float  # Cumulative output bit errors
    fe_cum:    int    # Cumulative output frame errors
    tests:     int    # The number of experiments

    def __str__(self):
        """
        Provide information during the simulation process
        :return: Information string with current results
        """
        data_str = f'#{self.tests:1.3e}, '
        if not self.tests:
            return data_str
        if self.tests:
            output = np.array([
                self.in_be_cum, self.in_se_cum,
                self.be_cum, self.fe_cum
            ]) / self.tests
        else:
            output = np.zeros(4,)
        data_template = 'IN BER: %1.3e, IN SER: %1.3e, OUT BER: %1.3e, OUT FER: %1.3e'
        return data_str + data_template % tuple(output.tolist())


class DataStorage:
    """
    This class implements simulation results and keeps the following data:
     - Simulation parameters (which SNRs to test and how many experiments to conduct
     - Captured statistics for each simulated SNR
     - Scheduling capabilities (via request_batch and update methods)
    """
    class SchedulingState(IntEnum):
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
        COMPLETE = 3   # Sufficient number of errors collected / max experiments conducted

    class SchedulingEntry:
        """
        Scheduling entry. Tracks the state of each SNR point
        """
        def __init__(self, settings, n_errors, n_tests, batch_size):
            """
            Initialize the scheduling entry using simulation settings and corresponding data entry
            """
            self.state = self.get_initial_state(settings, n_errors, n_tests)
            p_error = n_errors / n_tests if n_tests > 0 else 0
            # Required to include SNR points in acordance with 'look_ahead' parameter
            self.hit_minimum_pe = p_error < settings.min_error_prob
            # Error count required to make a scheduling decision
            self.error_count = n_errors
            # Batch size (estimated in accordance with probability of error)
            if n_errors > 0:
                # Set batch size in accordance wirh errors-per-batch
                self.n_tasks = settings.get_n_tasks(
                    p_error, n_tests, batch_size)
            else:
                # Otherwise, exponentially increase the batch size to load pool
                self.n_tasks = n_tests + 1

        @staticmethod
        def get_initial_state(settings, n_errors, n_tests):
            """
            Derive the state given simulation settings, the number of tests and errors
            """
            if n_tests > settings.max_experiments or n_errors > settings.max_errors:
                # Simulation is_simulation_complete
                return DataStorage.SchedulingState.COMPLETE
            if n_tests == 0 or n_errors / n_tests < settings.min_error_prob:
                # If no tests conducted or the error probability is below threshold -> IDLE
                # This state can be changed after checking the look-ahead condition
                return DataStorage.SchedulingState.IDLE
            # Need more experiments
            return DataStorage.SchedulingState.PENDING

        def is_pending(self):
            """
            Check whether the state is PENDING
            """
            return self.state == DataStorage.SchedulingState.PENDING

        def scheduled(self):
            """
            Mark entry as scheduled
            """
            self.state = DataStorage.SchedulingState.SCHEDULED

    def __init__(self, data_type, sim_settings, filename=None):
        """
        Initialize the data storage.
        :param data_type -- type of the DataEntry (see implementation above)
        :param sim_settings -- simulation settings
        :param filename -- Filename to store results. If None, data will not be saved
        """
        # Data type required to load data from pickle correctly
        self.data_type = data_type
        # The parameters below must be set by simulator
        self.sim_settings = sim_settings
        self.filename = filename
        # Codec-defined number of trials per task. It is set before scheduling.
        self.batch_size = 1

        # Initialize data entries
        self.d_entries = {}
        if filename is None:
            LOGGER.info('Data will not be saved!')
        elif os.path.isfile(filename):
            LOGGER.info('Loading data from file.')
            self.__load()
        else:
            LOGGER.info('File does not exist. Empty simulation results.')

        # Initialize scheduling entries
        self.s_entries = [self.init_s_entry(snr_db) for snr_db in self.sim_settings.snr_array]

        self.__update_n_tasks()
        self.__update_pending()

    def set_batch_size(self, batch_size):
        """
        Set the codec-defined number of trials per task and rebuild the
        scheduling state accordingly.
        """
        if batch_size < 1:
            raise ValueError('Batch size must be a positive integer')
        self.batch_size = batch_size
        self.s_entries = [
            self.init_s_entry(snr_db)
            for snr_db in self.sim_settings.snr_array
        ]
        self.__update_n_tasks()
        self.__update_pending()

    def update(self, results, snr_db):
        """
        Update corresponding entry with new data
        :param results: DataEntry() list
        :param snr_db: entry key in this storage
        :return: None, updates self
        """
        self.d_entries[snr_db] = dataclass_merge(self.d_entries.get(snr_db, None), results)
        self.save()
        # Log update event:
        updated_entry = self.d_entries[snr_db]
        LOGGER.info(
            'Collected %s. %s %1.2e/%1.2e Errors.',
            snr_db_str(snr_db, self.sim_settings.snr_precision),
            str(updated_entry),
            updated_entry.fe_cum,
            self.sim_settings.max_errors
        )

        # Update the scheduling entry:
        snr_index = np.argwhere(self.sim_settings.snr_array == snr_db).reshape(-1)[0]
        self.s_entries[snr_index] = self.init_s_entry(snr_db)

        # Re-estimate batch sizes and scheduling entries states
        self.__update_n_tasks()
        self.__update_pending()

    def is_simulation_complete(self):
        """
        Return true if more experiments required
        """
        states = np.array([entry.state for entry in self.s_entries])
        n_scheduled = np.sum(states == self.SchedulingState.SCHEDULED)
        return n_scheduled + self.get_pending_count() == 0

    def request_batch(self):
        """
        A batch has been req
        """
        idx_pending = np.argwhere([e.is_pending() for e in self.s_entries]).reshape(-1)
        if not len(idx_pending):
            raise ValueError('Check pending count before requesting the batch!')
        error_count = np.array([entry.error_count for entry in self.s_entries])
        id_schedule = idx_pending[np.argmin(error_count[idx_pending])]
        self.s_entries[id_schedule].scheduled()
        snr_db = self.sim_settings.snr_array[id_schedule]
        n_tasks = int(self.s_entries[id_schedule].n_tasks)
        n_tasks = min(n_tasks, self.sim_settings.max_tasks)

        # Log request event:
        LOGGER.info(
            'Scheduled %s. Batch size %1.4e (%1.4e X %1.4e)',
            snr_db_str(snr_db, self.sim_settings.snr_precision),
            n_tasks * self.batch_size,
            n_tasks,
            self.batch_size
        )

        return snr_db, n_tasks

    def get_pending_count(self):
        """
        Get the number of entries having PENDING state
        """
        return np.sum([e.is_pending() for e in self.s_entries])

    def __load(self):
        """
        Try to get data from file. If any data mismatch, it raises a runtime error
        :return: None, just update itself or raise runtime error
        """
        data = load_pickle(self.filename)

        # Check the SNR range consistency
        snr_array = np.array(list(data.keys()))
        if len(np.unique(np.round(snr_array, self.sim_settings.snr_precision))) != len(snr_array):
            raise ValueError('SNR precision does not match the data stored in *.pickle')

        # Fill the data
        for snr_db, entry_dict in data.items():
            self.d_entries[snr_db] = self.data_type(**entry_dict)

    def save(self):
        """
        Save data to pickle file
        """
        self.d_entries = dict(sorted(self.d_entries.items()))

        data = {}
        for snr_db, item in self.d_entries.items():
            data[snr_db] = vars(item)
        save_pickle(self.filename, data)

    def init_s_entry(self, snr_db):
        """
        Initialize the scheduling entry given the snr point
        """
        if snr_db not in self.d_entries:
            return DataStorage.SchedulingEntry(
                self.sim_settings, 0, 0, self.batch_size)
        d_entry = self.d_entries[snr_db]
        return DataStorage.SchedulingEntry(
            self.sim_settings, d_entry.fe_cum, d_entry.tests,
            self.batch_size)

    def __update_n_tasks(self):
        """
        The number of tasks is a non-decreasing sequence. Set it to the maximum
        among all previous values and the current value.
        """
        max_n_tasks = 0
        for i, _ in enumerate(self.sim_settings.snr_array):
            max_n_tasks = max(self.s_entries[i].n_tasks, max_n_tasks)
            self.s_entries[i].n_tasks = max_n_tasks

    def __update_pending(self):
        """
        Enforces the following rules:
         - SNR point is PENDING if it is not further than 'look_ahead' points from
           the last SNR point for which the probability of error is above the threshold
         - There are no IDLE points before any PENDING/COMPLETE point
        """
        states = np.array([entry.state for entry in self.s_entries])
        hit_error_prob = np.array([entry.hit_minimum_pe for entry in self.s_entries])
        last_incomplete = np.max(np.argwhere(hit_error_prob == 0), initial=0)
        idx_idle = np.argwhere(states == self.SchedulingState.IDLE)
        idx_idle = idx_idle[idx_idle <= last_incomplete + self.sim_settings.look_ahead]
        for i in idx_idle:
            self.s_entries[i].state = self.SchedulingState.PENDING
