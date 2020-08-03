# Getting Started on Linux with Cross-Compilation

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.
-->

This guide walks through cross-compiling IREE core runtime towards the embedded
Linux platform. Cross-compiling IREE compilers is not supported at the moment.

Cross-compilation involves both a *host* platform and a *target* platform. One
invokes compiler toolchains on the host platform to generate libraries and
executables that can be run on the target platform.

The following steps will use aarch64 as the embedded platform. These steps has
been tested on Ubuntu 18.04.

## Prerequisites

### Set up host development environment

The host platform should have been set up for developing IREE. Please make sure
you have followed the steps for
[Linux](./getting_started_linux_cmake.md).

### Install GCC/G++-10 Cross Toolchain

If Ubuntu version is eariler than Ubuntu 20.04, we need to add apt-repository
entry for newer gcc/g++.

```shell
# Again, this is not required for Ubuntu 20.04
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo vi /etc/apt/sources.list
    # add this line
     deb http://de.archive.ubuntu.com/ubuntu groovy main
```

Install gcc/g++-10

```shell
$ sudo apt-get update
$ sudo apt install gcc-10 g++-10
$ sudo apt install gcc-10-aarch64-linux-gnu g++-10-aarch64-linux-gnu
```

### Create CMake Cross-Toolchain File

Create cmake cross-toolchain file. We use aarch64 as an example.

```shell
$ vi aarch64-toolchain.cmake
```

Then copy-paste the following:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   "aarch64-linux-gnu-gcc-10")
set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++-10")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_C_FLAGS "-march=armv8-a")
set(CMAKE_CXX_FLAGS "-march=armv8-a")
```

## Configure and Build

### Configure on Linux

```shell
$ cmake -G Ninja /path/to/iree \
    -DCMAKE_TOOLCHAIN_FILE=aarch64-toolchain.cmake \
    -B build-aarch64 \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HOST_C_COMPILER=/usr/bin/gcc-10 \
    -DIREE_HOST_CXX_COMPILER=/usr/bin/g++-10
```

### Build

```shell
$ cmake --build build-aarch64/
```

## Test on Embedded Device

Translate a source MLIR into IREE module:

```shell
# Assuming in IREE source root
$ build-aarch64/host/bin/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-hal-target-backends=vmla \
    iree/tools/test/simple.mlir \
    -o /tmp/simple-vmla.vmfb
```

Then copy the IREE runtime executable and module to the device:

```shell
$ scp build-aarch64/iree/tools/iree-run-module ${device}:~
$ scp /tmp/simple-vulkan.vmfb ${device}:~
```

Log into device:

```shell
$ ssh ${device}

aarch64-ubuntu $ ./iree-run-module -driver=vmla \
                 -input_file=simple-vmla.vmfb \
                 -entry_function=abs \
                 -inputs="i32=-5"

EXEC @abs
i32=5
```