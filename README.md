# OpenXLA PJRT Plugin

This repository contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to OpenXLA/IREE.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

There are multiple development workflows, ranked from easiest to hardest (but
most powerful).

## Setup options

The below presumes that you have a compatible Jax/Jaxlib installed. Since 
PJRT plugin support is moving fast, it is rare that released versions are 
appropriate. See ["Building Jax from Source"](#building-jax-from-source) below.

### Option 1: Synchronize to a nightly IREE release

This will install compiler binaries from IREE's nightly releases and will then
sync a local clone of IREE's runtime to the same commit used to build the
compiler. This is the easiest way to work on pure-runtime/plugin features and
involves building the least.

```
# Run at any time to sync to the then-current nightly.
python build_tools/sync.py nightly

# Source environment variables to run interactively.
# The above generates a .env and .env.sh file with key setup vars.
source .env.sh

# Replace with actual compiler binaries if multiple.
CC=clang CXX=clang++ python external/iree/configure_bazel.py

# Configure path to CUDA SDK (for building CUDA plugin).
# Replace $CUDA_SDK_DIR as appropriate.
echo "build --action_env IREE_CUDA_DEPS_DIR=$CUDA_SDK_DIR" > user.bazelrc

# Build.

bazel build iree/integrations/pjrt/...

# Run a sample.

JAX_PLATFORMS=iree_cpu python test/test_simple.py
JAX_PLATFORMS=iree_cuda python test/test_simple.py
```

### Option 2: Set up for a full at-head dev rig

TODO: Document how to symlink existing repos and manually sync


## Building Jax from Source

Install Jax with Python sources:

```
pip install -e external/jax
```

Build a compatible jaxlib:

```
# Currently pluggable PJRT is commingled with TPU support... folks are
# working on it :/
cd external/jax
python build/build.py \
  --bazel_options=--override_repository=org_tensorflow=$PWD/../tensorflow \
  --enable_tpu
pip install dist/*.whl --force-reinstall
```

## Generating runtime traces

The plugins can be build with tracing enabled by adding the bazel build flag
`--iree_enable_runtime_tracing`. With this flag, if a profiler is running,
instrumentation will be sent to it. It can be useful to set the environment
variable `TRACY_NO_EXIT=1` in order to block termination of one-shot programs
that exit too quickly to stream all events.

## ASAN

Developing with ASAN is recommended but requires some special steps because
we need to arrange for the plugin to be able to link with undefined
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

* [GitHub issues](https://github.com/openxla/openxla-pjrt-plugin/issues):
  Feature requests, bugs, and other work tracking
* [OpenXLA discord](https://discord.gg/pvuUmVQa): Daily development discussions
  with the core team and collaborators

## License

OpenXLA PJRT plugin is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](LICENSE) for more information.

