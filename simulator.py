"""
This module represents simulation tools to build and automate experiments
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

import pickle
import sys
import argparse
import signal
import logging
import time
import multiprocessing as mp

from functools import partial
from rich import print

from .postprocessing import PostProcessing
from .live_plot import PlotServer
from .settings import Settings
from .parallel import main_loop, schedule_tasks
from .data_storage import DataStorage, DataEntry
from .bitrate import BitratePrinter


def run_all_experiments(config_type, experiment_type, data_type=DataEntry):
    """
    Main simulation script
    :param config_type: a type of experiment configuration class (see demo)
    :param experiment_type: a type of experiment instance
    :param data_type: type of data entry. Required to restore data from pickle files correctly
    Experiment must take a config instance as constructor input
    """
    SimulatorApp(config_type, experiment_type, data_type).run()


class SimulatorApp:
    """
    SimulatorApp keeps all live plot server processes, postprocessing instances,
    and runs all scheduled experiments.
    There are three main options captured from the command line arguments:
    - General JSON configuration file
    - Option to start live plot servers
    - Option to stop live plot servers after the simulation ends
    """
    def __init__(self, config_type, experiment_type, data_type=DataEntry):
        self.args = cmdline_argparser() # Command line arguments
        self.data_type = data_type  # Output data type
        self.experiment_type = experiment_type  # Experiment instance type
        self.settings = Settings(self.args.config, config_type)  # Settings

        # Postprocessing instances and live plots: one instance per experiment
        self.postproc_instances = []
        self.plot_processes = []

        # Output information (data storage logs and settings)
        print(self.settings)
        enable_log(
            'simulator_awgn_python.data_storage',
            logging.DEBUG,
            self.settings.simulation.log_file
        )

    def __del__(self):
        try:
            self.__stop_plot_servers()
        except AttributeError:
            # If initialization failed at settings creation, silently exit
            pass

    def run(self):
        """
        Main function to run all experiments.

        The plot processes are always stopped and the text output is always
        written, including on Ctrl+C, SIGTERM, and unexpected exceptions.
        """
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.__raise_keyboard_interrupt)
        try:
            while self.settings.remaining():
                self.__run_experiment()
            self.__terminate()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.__stop_plot_servers()
                self.__write_txt()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)

    @staticmethod
    def __raise_keyboard_interrupt(_, __):
        """
        Convert SIGTERM into the same cleanup path as Ctrl+C.
        """
        raise KeyboardInterrupt

    def __run_experiment(self):
        """
        Run single experiment
        """
        vis_params, exp_params = self.settings.next_experiment(print_vis=self.args.run_plots)
        # Connect data storage to the specified filename
        storage = self.__connect_storage(exp_params.filename)
        # Start live plotting
        self.__start_live_postprocessing(vis_params, exp_params)
        try:
            self.__run_main_loop(exp_params, storage)
        finally:
            self.__stop_live_postprocessing()

    def __connect_storage(self, filename):
        """
        Connect data storage. If some data is already stored, load it first.
        If data can not be loaded - report an error and exit.
        """
        try:
            return DataStorage(
                self.data_type,
                self.settings.simulation,
                filename
            )
        except pickle.UnpicklingError as exc:
            print('Can not read data from pickle:', exc)
            sys.exit(1)
        except ValueError as exc:
            print('Can not load data:', exc)
            sys.exit(1)

    def __run_main_loop(self, exp_params, storage):
        """
        Run main loop given data storage and experiment parameters.
        Parallel pool is created only if some experiments required
        """
        # Do not create a pool if no simulations required
        if storage.is_simulation_complete():
            return
        try:
            main_loop(
                self.experiment_type, exp_params, self.settings.simulation.n_workers,
                partial(do_simulate_awgn_fec, storage=storage, exp_settings=exp_params)
            )
        except KeyboardInterrupt:
            print('Simulation interrupted by user. Stopping experiment.')
            raise
        except ValueError as exc:
            print('Simulation failed.', exc)
            self.__stop_plot_servers()
            sys.exit(1)
        finally:
            storage.save()

    def __start_live_postprocessing(self, vis_params, exp_params):
        """
        Start live postprocessing (depending on selected options)
        """
        postproc_instance = PostProcessing(
            exp_params.filename,
            exp_params.modulation,
            self.settings.postproc
        )
        self.postproc_instances.append(postproc_instance)
        if not self.args.run_plots:
            return
        plot_process = mp.Process(
            target=run_plot_server,
            args=(postproc_instance, vis_params, exp_params),
            daemon=True
        )
        # Fork must not inherit the parent's SIGTERM->KeyboardInterrupt handler:
        # terminate() should stop the plot server immediately.
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            plot_process.start()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
        self.plot_processes.append(plot_process)

    def __stop_live_postprocessing(self):
        """
        Stop live postprocessing (depending on selected options)
        """
        if not self.args.run_plots:
            assert not self.plot_processes
            self.__write_txt()
            return
        if not self.args.stop_plots:
            print('To generate text files immediately, click on the link above.')
            return
        self.__stop_plot_servers()
        self.__write_txt()

    def __write_txt(self):
        """
        Run postprocessing: write text files from pickle files
        """
        for postproc in self.postproc_instances:
            print(f'Postprocessing: {postproc.filename} -> {postproc.txt_filename()}')
            postproc.get()
        self.postproc_instances = []

    def __stop_plot_servers(self):
        """
        Terminate all plot server processes and wait until they are gone.
        """
        processes = self.plot_processes
        self.plot_processes = []
        for plot_process in processes:
            if plot_process.is_alive():
                plot_process.terminate()
        for plot_process in processes:
            plot_process.join(timeout=5)
            if plot_process.is_alive():
                plot_process.kill()
                plot_process.join()

    def __terminate(self):
        """
        Wait for Ctrl+C before stopping persistent plot servers.
        If option 'stop_plots=no' has been selected, plot servers run until
        interrupted by user. Cleanup itself is handled by run().
        """
        if not self.args.run_plots:
            return
        # When all experiments are complete, plot server will continue working
        if not self.args.stop_plots:
            wait_for_ctrlc()


def do_simulate_awgn_fec(pool, storage, exp_settings, batch_size=1):
    """
    Main loop function consists of schedule-collect chain, with a maximum number
    of simultaneously scheduled tasks not exeeding 'look_ahead' parameter.
    batch_size is the codec-defined experiment batch size (default 1).
    """
    storage.set_batch_size(batch_size)
    task_queue = []
    bitrate_printer = BitratePrinter(
        storage.sim_settings, exp_settings, batch_size)
    look_ahead = storage.sim_settings.look_ahead  # Maximum tasks that can be scheduled
    while not storage.is_simulation_complete():
        # Collect data (if the number of scheduled task reached maximum):
        if len(task_queue) > look_ahead or storage.get_pending_count() == 0:
            got_snr_db, async_result = task_queue.pop(0)
            batch_data = async_result.get()
            bitrate_printer.print(len(batch_data))
            storage.update(batch_data, got_snr_db)
        # Schedule data:
        if storage.get_pending_count():  # May change afer storage.update()
            scheduled_snr_db, n_tasks = storage.request_batch()
            async_result = schedule_tasks(scheduled_snr_db, int(n_tasks), batch_size, pool)
            task_queue.append((scheduled_snr_db, async_result))
    # Reset bit-rate printing
    bitrate_printer.stop()


def wait_for_ctrlc():
    """
    Wait for Ctrl+C event triggered by user
    """
    print('All experiments completed. ', end='')
    print('Press Ctrl+C to stop plot servers and run postprocessing.')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def cmdline_argparser():
    """
    Command line arguments parser
    """
    parser = argparse.ArgumentParser(description='FEC simulation tool')
    parser.add_argument('-c', '--config', required=True, help='Filename with simulation parameters')
    parser.add_argument(
        '-p', '--run-plots', default=True, action=argparse.BooleanOptionalAction,
        help='Run live-plots visualization'
    )
    parser.add_argument(
        '-s', '--stop-plots', default=False, action=argparse.BooleanOptionalAction,
        help='Terminate live-plots after each experiment'
    )
    return parser.parse_args()


def enable_log(name, level=logging.DEBUG, filename=None):
    """
    Enable logging with proper formats
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if filename is None:
        handler = logging.StreamHandler(stream=sys.stdout)
    elif filename == '':
        print('Empty logging file. Logging disabled')
        return
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter('%(asctime)s %(name)s-%(levelname)s: %(message)s'))
    handler.setLevel(level)
    logger.addHandler(handler)


def run_plot_server(postproc_instance, vis_settings, exp_settings):
    """
    Target-function for live-plot server. Redirect stdout to log file
    """
    # Redirect flask stdout to file. The parent stops this process with
    # terminate() (SIGTERM), so keep the default SIGTERM behavior here.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    with open(vis_settings.log_file, 'a', encoding='utf-8') as sys.stdout:
        PlotServer(
            exp_settings.title,
            vis_settings,
            postproc_instance=postproc_instance
        ).run()
