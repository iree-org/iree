# Python API Docs

Documentation for the Python API is built with Sphinx under this directory.
When new versions are released, the documentation is generated and published
to the [Read the Docs](https://app.readthedocs.org/projects/iree-python-api/)
project and is served at
[readthedocs.io](https://iree-python-api.readthedocs.io/en/latest/).

## Building the API documentation locally

### Setup virtual environment with requirements

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### (Optional) Using locally built IREE packages

You can build the site using either released Python packages or local packages,
whichever appear first on the `PYTHONPATH` environment variable. The
`requirements.txt` file used in the previous step downloads the latest
pre-release (nightly) `iree-base-compiler` and `iree-base-runtime` packages.

To use local packages, such as when changing docstrings and wanting to see how
they appear in the generated documentation, follow the instructions for building
the Python bindings from source at
<https://iree.dev/building-from-source/getting-started/#using-the-python-bindings>.

In particular, after building with `-DIREE_BUILD_PYTHON_BINDINGS=ON`, you will
need to extend your `PYTHONPATH` to include the relevant build directories. The
the generated `.env` files can help with this:

```shell
source ../../../iree-build/.env && export PYTHONPATH
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

### Clean to show all warnings

A clean rebuild will show all warnings again:

```shell
make clean
sphinx-build -b html . _build
```
