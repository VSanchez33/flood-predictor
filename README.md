# CSCI 4405/5405 Project

Flash flood predictor.

## Install

Firstly, install pipx in order to install UV. Once pipx is installed run the
following:

```
pipx install uv
```

Secondly, create a virtual environment through UV and source it:

```
uv venv
source .venv/bin/activate
```

Lastly, build the application like so:

```
uv pip install -e .
```

Now the `flood-predictor` CLI app is available in the virtual environment,
and be run like so:

```
flood-predictor /path/to/gage-height.csv /path/to/discharge.csv
```

## License

This project uses the MIT license.
