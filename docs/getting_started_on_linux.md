<!--
  Copyright 2019 Google LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

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

### Install the Vulkan SDK

Some parts of the project link against the Vulkan SDK and require it be
installed on your system. If you are planning on building these, or see linker
errors about undefined references to `vk` symbols, download and install the
Vulkan SDK from https://vulkan.lunarg.com/, and check that the `VULKAN_SDK`
environment variable is set when you are building. You may find it useful to add
`source [PATH TO VULKAN SDK]/setup-env.sh` to your `~/.bashrc` file to simplify
environment variable setup.

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
