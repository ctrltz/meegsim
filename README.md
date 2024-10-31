# MEEGsim

## Overview

**MEEGsim** is a Python package that provides template waveforms for simulating M/EEG data with known ground truth source activity. In addition, it aims to simplify the manipulation of relevant simulation parameters (e.g., signal-to-noise ratio and source connectivity). As a result, the users can focus on what to simulate, not on how to implement the simulation. The package is designed to be compatible with MNE-Python and re-uses the forward and inverse modeling functionality provided by MNE.

Find more details about the package in the [documentation](https://meegsim.readthedocs.io/en/latest/). For a brief overview of the functionality, feel free to have a look at the [poster](https://drive.google.com/file/d/14KVjHdnnEdUFOrbRWb59Rqsj_cwjElHV/view?usp=sharing) about MEEGsim that was presented at the [CuttingEEGX](https://cuttingeegx.org/) conference (28-31.10.2024, Nijmegen, The Netherlands, and online).

## Development

### Creating a Local Copy of the Project

1. Clone the repository.

2. Create an environment (conda/mamba/virtualenv).

3. Switch to the project folder and install the package and all dependencies: 

```bash
cd meegsim
pip install -e .[dev]
```

4. You're ready to start now!

### Running Tests

```
pytest
```

### Building the Documentation

1. Install the required packages.

```bash
pip install -e .[docs]
```

2. Build the documentation.

```bash
make html
```

3. Open it in the web browser.

```bash
make show
```
