# IREE Core Python Package

`iree.core` holds the pure-Python components shared by the IREE compiler
(`iree-base-compiler`) and runtime (`iree-base-runtime`) packages. It is
distributed as the `iree-base-core` wheel.

`iree` is a namespace package that `iree.core` shares with `iree.compiler` and
`iree.runtime`, so there must be no `iree/__init__.py`.

## Dependencies

`pyproject.toml` lists the dependencies of the wheel as version ranges.
`requirements.txt` pins them with hashes for builds and CI. To regenerate it
after a change, run the command at its top in this directory.

## Testing

With Bazel:

```shell
bazel test //core/python/...
```

With CMake, the tests are registered if the Python interpreter of the build has
the dependencies installed. They are run by pytest. The Python bindings are not
needed:

```shell
python -m pip install pytest
python -m pip install --require-hashes -r core/python/requirements.txt
ctest --test-dir ${BUILD_DIR} -R iree/core/python
```

Against the installed package:

```shell
python -m pip install pytest core/python/
python -m pytest core/python/test
```
