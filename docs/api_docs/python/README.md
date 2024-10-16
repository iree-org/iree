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
export PYTHONPATH=$PWD/bindings/python:$PWD/compiler-api/python_package
```

### Install dependencies

```shell
python -m pip install -r requirements.txt
```


### Build docs

```shell
sphinx-build -b html . _build
```
