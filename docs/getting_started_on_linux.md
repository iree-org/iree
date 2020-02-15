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

If you are planning on using TensorFlow/Colab, make sure that the Bazel version
you install is
[supported by TensorFlow](https://www.tensorflow.org/install/source#install_bazel).
A .bazelversion file is also provided if you want to use
[Bazelisk](https://github.com/bazelbuild/bazelisk) to manage Bazel versions.

### Install clang

```
sudo apt install clang
```

Verify version with `clang++ --version`. We have verified with the following
versions:

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

If using Colab, you may also want to install TensorFlow:

```shell
sudo pip3 install tf-nightly
```

### Install the Vulkan SDK

Some parts of the project link against the Vulkan SDK and require it be
installed on your system. If you are planning on building these, or see linker
errors about undefined references to `vk` symbols, download and install the
Vulkan SDK from https://vulkan.lunarg.com/, and check that the `VULKAN_SDK`
environment variable is set when you are building. You may find it useful to add
`source [PATH TO VULKAN SDK]/setup-env.sh` to your `~/.bashrc` file to simplify
environment variable setup.

## Optional: Configure Git

### Git SSH

*   Generate SSH Key: `ssh-keygen -t rsa -b 4096 -C "EMAIL@email.com"`
*   Add `~/.ssh.id_rsa.pub` key to GitHub
*   Try a test connection `ssh git@github.com`

### Other git config options

```shell
git config --global user.name "MY NAME"
git config --global user.email "MY EMAIL"
```

## Clone

This assumes that we are cloning into `$HOME/ireepub`. Update accordingly for
your use.

Note that if you will be cloning frequently, it can be sped up significantly by
creating a reference repo and setting
`IREE_CLONE_ARGS="--reference=/path/to/reference/repo"`. See
`scripts/git/populate_reference_repo.sh` for further details.

```shell
IREE_CLONE_ARGS=""
mkdir -p $HOME/ireepub
cd $HOME/ireepub
git clone $IREE_CLONE_ARGS https://github.com/google/iree.git iree
cd iree
git submodule init
git submodule update $IREE_CLONE_ARGS --recursive
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
build --disk_cache=/REPLACE/WITH/CACHE/DIR
build:debug --compilation_mode=dbg --copt=-O2 --per_file_copt=iree@-O0 --strip=never
```

### Build

```shell
# Run all core tests
bazel test -k iree/...

# Or build with optimizations disabled (just for IREE, not for deps) and
# debug symbols retained. This assumes you have an alias like above setup):
bazel test --config=debug bindings/python/...
```

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Building with CMake

### Environment variables

The following environment variables must be set.

```shell
export CXX=clang++
```

### Build

```shell
mkdir build && cd build

cmake -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=ON -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_DEBUGGER=OFF ..

cmake --build .
```
