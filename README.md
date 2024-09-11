# MEEGsim

M/EEG simulation framework

## Creating a Local Copy of the Project

1. Clone the repository.

2. Create an environment (conda/mamba/virtualenv).

3. Switch to the project folder and install the package and all dependencies: 

```bash
cd meegsim
pip install -e .[dev]
```

4. You're ready to start now!

## Running Tests

```
pytest
```

## Building the Documentation

1. Install the required packages.

```bash
pip install -e.[docs]
```

2. Build the documentation.

```bash
make html
```

3. Open it in the web browser.

```bash
make show
```

