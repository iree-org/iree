# PyDM compiler framework

PyDM ("Python DataModel") is an IREE dialect which models the Python
language in a way suitable for compilation and offline execution. While the
core compiler itself is encapsulated in the `iree_pydm` dialect, this
directory contains "the rest of the story", consisting of:

* User-level APIs
* Example integrations
* Test suite

This is only loosely coupled to IREE and should be seen as an incubation
staging area vs a signal that the end result will be tightly bound to IREE
itself.

## Development

This package is pure Python and can be installed via:

```
pip install -e experimental/pydm_compiler/
```

The usual cautions about using a virtual environment, etc are encouraged.

The above presumes that the `iree-compiler` and `iree-runtime` dependencies
can be resolved. Typically, this is done by installing from nightlies:

```
pip install -e experimental/pydm_compiler/ \
  -f https://github.com/google/iree/releases
```

If you are already using IREE, setup for Python development, then you should
install without deps:

```
pip install --no-deps -e experimental/pydm_compiler/
```

## Running tests

From this directory:

```
pytest
```
