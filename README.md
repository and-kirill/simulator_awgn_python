# Parallel link-level simulator for error-correcting codes evaluation
## Description

This module provides simulation routines for error-correcting codes. It provides the following functionality:
* AWGN channel routines (BPSK, QPSK, QAM-16 modulation, additive white Gaussian noise simulation, soft demodulation)
* Parallel execution of multiple tests for different signal-to-noise ratios
* [Proper initialization](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html) of the random number generator to provide independent RNGs within multiple processes. By default, the [numpy](https://pypi.org/project/numpy/) `default_rng` [random number generator](https://numpy.org/doc/stable/reference/random/generator.html) is used

## Module structure

### Simulator

The structure of the `Simulator` is as follows. It starts from the `DataEntry` class, which keeps decoding statistics for a single SNR point. These statistics can be incrementally updated.

The `DataStorage` accumulates multiple SNR points. It also provides data-saving routines and informs the `Simulator` whether to proceed to the next SNR or stop the simulations.

The `Simulator` runs user-specified decoding function in parallel until the simulation reaches the requested number of errors, or the number of experiments, or reaches the minimum probability of error specified by user.

To instantiate the `DataStorage`, the user must provide a `DataEntry` type (any class that implements methods like print, merge, calculates error count and error probability).

For more details, refer to the [demo](demo.py) script.
### Simulation scripts
Some automated routines are presented in [tools.py](tools.py), which allow:
* Gradually increase the batch size to reduce communication overhead
* Gradually increase the maximum number of errors to be collected such that a user has the simulated results that gradually improve the collected statistics (default for automated experiments)
* To automate experiments (see `run_all_experiments` function). Experiments assume a JSON configuration file to be parsed. If some parameters are lists, multiple experiments with the iteration over elements are conducted.
### Demo
[demo](demo.py) scripts runs a dummy decoder that is able to correct some fixed number of errors. Default `DataEntry` class provides input/output BER (bit error rate), input SER (symbol error rate), and the output FER (frame error rate).
One can implement any arbitrary experiment. To avoid communication overhead, avoid keeping bulkey structures within this class.

### Live plot
To check the simulation result in realtime, [demo](demo.py) starts a process that periodically loads the pickle file with the simulated data, performs postprocessing, and generates a [plotly](https://plotly.com) figure. One can view this figure using `dash` (URL will be printed to [demo.py](demo.py) output).
The following plots are available:
1. Uncoded bit error rate and its theoretical values
2. Output frame error rate (FER) with error bars corresponding to [confidence intervals for Bernoulli trials](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
3. A smooth FER generated by a linear regression over the logarithm of the error probability and bernoulli likelihood function

These outputs are saved to the text file that can further be captured by latex (See corresponding [standalone tex-file](berfit_plot.tex))

## Usage
The code is assumed to be used as a module. To run demo experiment, proceed through the following steps:

* Create a JSON file specifying the experiment. Demo-example is shown below:
  - JSON file consists of two sections: "experiment" and "simulations". The first section keeps all data required to instantiate the experiment. The second section keeps all simulation-related data. Below is an example with comments.
  - To instantiate the demo experiment, specify the following parameters:
    - `block_len` specifies the codeword length
    - `correctable_errors` specifies the number of errors that demo-decoder can correct (see [demo.py](demo.py))
    - `modulation` a string constant specifying the modulation (upper or lower case). supported values are 'BPSK', 'QPSK', and 'QAM-16'
  - To run simulations, specify the following parameters:
    - `snr_range` a string that represents a MATLAB-style array '`min`:`step`:`max`'
    - `max_errors` is a maximum number of errors to be collected. If the specified number of errors has happened, simulator stops evaluating the corresponding SNR point and proceeds to the next one.
    - `max_experiments` is a maximum number of experiments to conduct. If this number is hit before the condition above, simulator will proceed to the next SNR point
    - `min_error_prob` is a minimum probability of error to be simulated. If (after the conditions above satisfied) the probability of error goes below the specified value, simulation stops. Simulator assumes that the probability of error decreases with the SNR increase, and it iterates through the sorted SNR values. Note that some requested SNR points may not be evaluated.
  - The summarized JSON example is presented below:
```json
{
  "experiment": {
    "block_len": 32,
    "correctable_errors": 4,
    "modulation": "qpsk"
  },
  "simulation": {
    "snr_range": "-5:0.02:10",
    "max_errors": 50,
    "max_experiments": 1e7,
    "min_error_prob": 1e-4
  }
}
```


* Run the simulation script (not from the module directory)
```console
python3 -m simulator_awgn_python.demo --config=<json_file>.json
```

## Requirements
This module was tested in Python 3.8 under UNIX OS with the following packages installed:

* Numerical python [numpy](https://pypi.org/project/numpy/)
* Scientific python [scipy](https://scipy.org) for postprocessing and AWGN channel
* [plotly](https://plotly.com) and [dash](https://pypi.org/project/dash/) for live-plots
* [pandas](https://pandas.pydata.org) for postprocessing
To test the AWGN channel, run the test [script](test_awgn_channel.py) that requires [scikit-commpy](https://pypi.org/project/scikit-commpy/). Note that the latter is considerably slower.
