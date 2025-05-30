# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

There are multiple development workflows, ranked from easiest to hardest (but
most powerful).

## Install a compatible version of Jax and the IREE compiler

```shell
pip install -r requirements.txt

pip install jax==0.6.1
```

Verify that your Jax install is functional like:

```shell
python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Install the plugin of your choice (in this example 'cpu')

```shell
pip install -v --no-deps -e python_packages/iree_cpu_plugin
```

## Verify basic functionality

```shell
JAX_PLATFORMS=iree_cpu python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Advanced settings

To pass additional compile options to IREE during JIT compilation, you can use
the `IREE_PJRT_IREE_COMPILER_OPTIONS` environment variable. This variable can
be set to a space-delimited list of flags that would be passed to the
`iree-compile` command-line tool.

For example:
```shell
export IREE_PJRT_IREE_COMPILER_OPTIONS=--iree-scheduling-dump-statistics-format=csv
JAX_PLATFORMS=iree_cpu python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

Besides, to control logging levels in the IREE PJRT plugin,
you can set `IREE_PJRT_LOG_LEVEL` to `debug` or `error` (default: `debug`).

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

## Communication channels

* Please submit feature requests and bug reports about the plugin in [GitHub Issues](https://github.com/iree-org/iree/issues).
* Discuss the development of the plugin at `#jax` or `#pjrt-plugin` channel of [IREE Discord server](https://discord.gg/wEWh6Z9nMU).
* Check the [OpenXLA/XLA](https://github.com/openxla/xla) repo and [its communication channels](https://github.com/openxla/community?tab=readme-ov-file#communication-channels) for PJRT APIs and clients.

## License

IREE PJRT plugin is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](../../LICENSE) for more information.

[PJRT C API](./third_party/pjrt_c_api) comes from
[OpenXLA/XLA](https://github.com/openxla/xla) and is licensed under
the Apache 2.0 License. See its own [LICENSE](./third_party/pjrt_c_api/LICENSE) for more information.
