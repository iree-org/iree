# OpenXLA PJRT Plugin

This repository contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to OpenXLA/IREE.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

It is recommended to checkout `jax`, `iree`, `iree-samples`, and `tensorflow`
side by side, as we will be overriding them all to build at head.

Note that although Jax is emitting serialized stablehlo bytecode, which should
be compatible across versions eventually, it is still early days and things
are not stable yet. It is recommended to use a `tensorflow` repo at the same
commit that IREE uses.

## Build and install custom jaxlib

From a Jax checkout:

```
# Currently pluggable PJRT is commingled with TPU support... folks are
# working on it :/
pip install -e .
python build/build.py \
  --bazel_options=--override_repository=org_tensorflow=$PWD/../tensorflow \
  --enable_tpu
pip install dist/*.whl --force-reinstall
```

## Build this project and look at a plugin

Currently, enabling Bazel build support for IREE is manual and requires
setting an environment variable in a way that Bazel uses. In the IREE source
directory, open `configured.bazelrc` and add a line like this (replacing
`PATH_TO_IREE_BUILD_DIR` with an actual path to a build dir or set the
location to a manually installed SDK):

```
build --action_env IREE_CUDA_TOOLKIT_ROOT="PATH_TO_IREE_BUILD_DIR/build_tools/third_party/cuda/11.6.2/linux-x86_64"
```

```
bazel build ...
IREE_PLUGIN_PATH="$PWD/bazel-bin/iree/integrations/pjrt/cpu/lib_pjrt_plugin_iree_cpu.so"
ls -lh $IREE_PLUGIN_PATH
```

## Run a Jax test program.

Note that the JAX plugin initialization sequence needs a patch:
https://github.com/google/jax/pull/14011

```
# Tells the IREE plugin where to find the compiler. Only needed for now.
export IREE_PJRT_COMPILER_LIB_PATH=$IREE_BUILD_DIR/lib/libIREECompiler.so
export PJRT_NAMES_AND_LIBRARY_PATHS="iree_cpu:$PWD/bazel-bin/iree/integrations/pjrt/cpu/pjrt_plugin_iree_cpu.so,iree_cuda:$PWD/bazel-bin/iree/integrations/pjrt/cuda/pjrt_plugin_iree_cuda.so"
# Jax only enable the plugin path if TPUs enabled for the moment.
export JAX_USE_PJRT_C_API_ON_TPU=1

# Optional: path to libcuda.so
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib

JAX_PLATFORMS=iree_cpu python test/test_simple.py
JAX_PLATFORMS=iree_cuda python test/test_simple.py
```

## Generating runtime traces

The plugins can be build with tracing enabled by adding the bazel build flag
`--iree_enable_runtime_tracing`. With this flag, if a profiler is running,
instrumentation will be sent to it. It can be useful to set the environment
variable `TRACY_NO_EXIT=1` in order to block termination of one-shot programs
that exit too quickly to stream all events.

## ASAN

Developing with ASAN is recommended but requires some special steps because
of we need to arrange for the plugin to be able to link with undefined
symbols and load the ASAN runtime library.

* Edit out the `"-Wl,--no-undefined"` from `build_defs.bzl`
* Set env var `LD_PRELOAD=$(clang-12 -print-file-name=libclang_rt.asan-x86_64.so)`
  (assuming compiling with `clang-12`. See configured.bazelrc in the IREE repo).
* Set env var `ASAN_OPTIONS=detect_leaks=0` (Python allocates a bunch of stuff
  that it never frees. TODO: Make this more fine-grained so we can detect leaks in
  plugin code).
* `--config=asan`

This can be improved and made more systematic but should work.

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

* (GitHub issues)[https://github.com/openxla/openxla-pjrt-plugin/issues]:
  Feature requests, bugs, and other work tracking
* (OpenXLA discord)[https://discord.gg/pvuUmVQa]: Daily development discussions
  with the core team and collaborators

## License

OpenXLA PJRT plugin is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.

