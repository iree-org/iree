# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

There are multiple development workflows, ranked from easiest to hardest (but
most powerful).

## Install a compatible version of Jax and the IREE compiler

```
pip install -r requirements.txt

# Assume that you have the Jax repo checked out at JAX_REPO from
# https://github.com/google/jax (must be paired with nightly jaxlib).
pip install -e $JAX_REPO
```

Verify that your Jax install is functional like:

```shell
python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Install the plugin of your choice (in this example 'cpu')

pip install -v --no-deps -e python_packages/iree_cpu_plugin

## Verify basic functionality

```shell
JAX_PLATFORMS=iree_cpu python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Incrementally developing

If you did an editable install (`-e`) above, then you should be able to incrementally
make changes and build the native component with no further interaction needed.

```shell
cd python_packages/iree_cpu_plugin/build/cmake
ninja
```

## Running the Jax test suite

The JAX test suite can be run with pytest. We recommend using `pytest-xdist`
as it spawns tests in workers which can be restarted in the event of individual
test case crashes.

Setup:

```
# Install pytest
pip install pytest pytest-xdist

# Install the ctstools package from this repo (`-e` makes it editable).
pip install -e ctstools
```

Example of running tests:

```
JAX_PLATFORMS=iree_cuda pytest -n4 --max-worker-restart=9999 \
  -p openxla_pjrt_artifacts --openxla-pjrt-artifact-dir=/tmp/foobar \
  ~/src/jax/tests/nn_test.py
```

Note that you will typically want a small number of workers (`-n4` above) for
CUDA and a larger number can be tolerated for cpu.

The plugin `openxla_pjrt_artifacts` is in the `ctstools` directory and
performs additional manipulation of the environment in order to save
compilation artifacts, reproducers, etc.

## Contacts

* [GitHub issues](https://github.com/openxla/openxla-pjrt-plugin/issues):
  Feature requests, bugs, and other work tracking
* [OpenXLA discord](https://discord.gg/pvuUmVQa): Daily development discussions
  with the core team and collaborators

## License

OpenXLA PJRT plugin is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.
