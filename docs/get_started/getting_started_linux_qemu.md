# Getting Started on Linux with Qemu-User

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.
-->

This guide walks through using qemu-user to build the core compiler and runtime
parts of IREE towards the embedded linux platform.

The following steps will use aarch64 as the embedded platform. These steps has
been tested on Ubuntu 18.04. It may take around 7 hours to build everything on
the x86 machines with 8 hardware threads running on 3.4 GHz + 16 GB RAM.

## Prerequisites

### Install Qemu User

Install qemu-user to run aarch64 based rootfs (root file system).

```shell
$ sudo apt install qemu-user-static
```

### Prepare AArch64 Based Ubuntu Rootfs

Download aarch64 ubuntu-base rootfs from official site and use sudo to untar it
in ubuntu-18.04-arm64 foler. `sudo` is required here since there are special
device node files which need to be created under root permission.

```shell
$ export ROOTFS=$PWD
$ wget http://cdimage.ubuntu.com/ubuntu-base/releases/18.04/release/ubuntu-base-18.04-base-arm64.tar.gz
$ mkdir ubuntu-18.04-arm64 && cd ubuntu-18.04-arm64
$ sudo tar -xf ../ubuntu-base-18.04-base-arm64.tar.gz -C ./
```

Copy `qemu-aarch64-static` to the aarch64 rootfs for executing aarch64
binaries on the host x86. Copy `resolv.conf` to the aarch64 rootfs for network
access.

```shell
$ sudo cp /usr/bin/qemu-aarch64-static $ROOTFS/usr/bin/
$ sudo cp /etc/resolv.conf $ROOTFS/etc
```

### Prepare Folders Shared between Host and Rootfs

It is very convenient to have some shared folders between the host and the
aarch64 rootfs, this example creates iree source and build folder on the host,
then bind it to the aarch64 rootfs.

```shell
$ sudo mkdir -p $ROOTFS/path/to/iree
$ sudo mkdir -p $ROOTFS/path/to/iree-build
$ sudo mount --bind /path/to/iree $ROOTFS/path/to/iree
$ sudo mount --bind /path/to/iree-build $ROOTFS/path/to/iree-build
```

### Bind System Nodes to Aarch64 Rootfs, then Chroot

Bind system nodes and start emulating aarch64 on host.

```shell
$ sudo mount --bind /proc proc
$ sudo mount --bind /sys sys
$ sudo mount --bind /dev dev
$ sudo chroot $ROOTFS
```

## Setup Build Environment for Aarch64 Rootfs

```shell
# apt-get update
# apt-get install build-essential wget python3 git ninja-build
```

### Install CMake

IREE uses CMake version `>= 3.13`, default installed cmake for ubuntu 18.04 is
older than 3.13, so build and install newer version cmake for the aarch64
rootfs. This example uses cmake 3.15.2.

```shell
# wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz
# tar -zxf cmake-3.15.2.tar.gz
# cd cmake-3.15.2
# ./bootstrap
# make -j8 && sudo make install
```

### Install a Compiler

We recommend Clang. GCC version `>= 9` is also supported.

```shell
# wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-aarch64-linux-gnu.tar.xz
# tar -xf clang+llvm-10.0.0-aarch64-linux-gnu.tar.xz
```

### [Optional] Prepare Vulkan

Assume vulkan include and lib are available, then create a folder and copy
header and lib into this folder.

```shell
# mkdir vulkan-sdk
# cp -r /path/to/vulkan/include vulkan-sdk
# cp -r /path/to/vulkan/lib vulkan-sdk
```

The vulkan-sdk folder struct
```shell
vulkan-sdk/
├── include
│   └── vulkan
│       ├── vk_android_native_buffer
...
│       ├── vk_android_native_buffer.h
...
└── lib
    └── libvulkan.so
```

### Build

Build IREE with lld linker and experimental on.

```shell
# export IREE_LLVMAOT_LINKER_PATH="$(which ld)"
# export VULKAN_SDK=/path/to/vulkan-sdk
# cmake -G Ninja /path/to/iree \
    -B /path/to/iree-build/aarch64 \
    -DIREE_BUILD_EXPERIMENTAL=ON \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER=./clang+llvm-10.0.0-aarch64-linux-gnu/bin/clang \
    -DCMAKE_CXX_COMPILER=./clang+llvm-10.0.0-aarch64-linux-gnu/bin/clang++
# cmake --build /path/to/iree-build/aarch64
```

## What's next?

### Take a Look Around

Check out the contents of the 'experimental test' build directory:

```shell
# cd /path/to/iree-build/aarch64
# ls experimental/ModelBuilder/test
```

Scp these tests and `third_party/llvm-project/llvm/lib/libvulkan-runtime-wrappers.so.12git`
to real device, then run it.

```shell
$ ./bench-matmul-vector-jit
$ ./test-matmul-vulkan
```
