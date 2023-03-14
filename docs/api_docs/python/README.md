# Python API Docs

Documentation for the Python API is built with Sphinx under this directory.
When new versions are released, the documentation is generated and published
to the website.

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
