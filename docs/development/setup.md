# Setting up the local environment

## Creating a local copy of the project

1. Clone the [repository](https://www.github.com/ctrltz/meegsim).

2. Create an environment (conda/mamba/virtualenv/etc).

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
