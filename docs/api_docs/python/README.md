# Python API Docs

Documentation for the Python API is built with Sphinx under this directory.
When new versions are released, the documentation is generated and published
to the [Read the Docs](https://readthedocs.org/projects/iree-python-api/)
project and is served at
[readthedocs.io](https://iree-python-api.readthedocs.io/en/latest/).

## Building Docs

### Install IREE binaries

Either install python packages or, from the build directory:

```shell
source .env && export PYTHONPATH
```

(See
<https://iree.dev/building-from-source/getting-started/#using-the-python-bindings>)

### Setup virtual environment with requirements

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Build docs

```shell
sphinx-build -b html . _build
```

### Serve locally locally with autoreload

```shell
sphinx-autobuild . _build
```

Then open http://127.0.0.1:8000 as instructed by the logs and make changes to
the files in this directory as needed to update the documentation.
