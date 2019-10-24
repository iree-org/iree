# Getting Started on Linux

There are many Linux variants. This document provides project-specific
requirements for setting up your environment and should be relatively
translatable. It was written for a relatively recent, Debian-derived
distribution. If you have issues on other systems, we accept issues or additions
to the documentation.

## Pre-requisites

### Install Bazel >= 1.0

*Optional if you will only be using the CMake build for runtime deps.*

Follow the
[install instructions](https://docs.bazel.build/versions/master/install-ubuntu.html)
and verify with: `bazel --version`.

### Install clang

```
sudo apt install clang
```

Verify version with `clang++ -v`. We have verified with the following versions:

*   6.0.1

### Install python, pip and dependencies

Install python (note that if, on your distribution, python3 is the only option,
you may be able to drop the "3").

```
sudo apt install python3 python3-pip
```

Verify python version with `python3 -V`. We have tested with >= 3.6. Some
optional integrations (such as TensorFlow/Python may have more stringent
requirements).

Install packages:

```
sudo pip3 install numpy
```

## Building with Bazel

We support both Bazel and CMake, however, the Bazel build covers more parts of
the system (tests, integrations, bindings, etc) whereas (currently) CMake is
maintained for the runtime components (which will be consumed by other
projects). This section covers building with Bazel.

### Environment variables

The following environment variables must be set.

```shell
export CXX=clang++
export CC=clang
export PYTHON_BIN="$(which python3)"
```

### Optional: Setup user config aliases

You can create a `user.bazelrc` in the repository root with extra bazel configs
that may be useful. We usually have something like this (make sure to make
replacements as needed):

```
build --disk_cache=/REPLACE/WITH/CACHE/DIR --experimental_guard_against_concurrent_changes
build:debug --compilation_mode=dbg --copt=-O2 --per_file_copt=iree@-O0 --strip=never
```

### Build

```shell
# Run all tests (as of Oct-23-2019, all tests build but some still fail in the
# OSS version).
bazel test -k iree/... test/... bindings/python/...

# Or build with optimizations disabled (just for IREE, not for deps) and
# debug symbols retained. This assumes you have an alias like above setup):
bazel test --config=debug bindings/python/...
```

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Building with CMake

TODO
