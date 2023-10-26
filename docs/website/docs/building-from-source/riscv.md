---
hide:
  - tags
tags:
  - CPU
icon: octicons/cpu-16
---

# RISC-V cross-compilation

Running on a platform like RISC-V involves cross-compiling from a _host_
platform (e.g. Linux) to a _target_ platform (a specific RISC-V CPU architecture
and operating system):

* IREE's _compiler_ is built on the host and is used there to generate modules
  for the target
* IREE's _runtime_ is built on the host for the target. The runtime is then
  pushed to the target to run natively.

## :octicons-download-16: Prerequisites

### Host environment setup

You should already be able to build IREE from source on your host platform.
Please make sure you have followed the [getting started](./getting-started.md)
steps.

### Install RISC-V cross-compile toolchain and emulator

You'll need a RISC-V LLVM compilation toolchain and a RISC-V enabled QEMU
emulator.

See instructions in the following links

* [Clang getting started](https://clang.llvm.org/get_started.html)
* [RISC-V GNU toolchain](https://github.com/riscv/riscv-gnu-toolchain)
* [QEMU](https://gitlab.com/qemu-project/qemu)
* [RISC-V Linux QEMU](https://risc-v-getting-started-guide.readthedocs.io/en/latest/linux-qemu.html)

!!! note
    The `RISCV_TOOLCHAIN_ROOT` environment variable needs
    to be set to the root directory of the installed GNU toolchain when building
    the RISC-V compiler target and the runtime library.

#### Install prebuilt RISC-V tools (RISC-V 64-bit Linux toolchain)

Execute the following script to download the prebuilt RISC-V toolchain and QEMU
from the IREE root directory:

```shell
./build_tools/riscv/riscv_bootstrap.sh
```

!!! note
    The prebuilt toolchain is built with AlmaLinux release 8.8
    [docker](https://quay.io/pypa/manylinux_2_28_x86_64)
    It requires glibc >= 2.28 for your host machine.

#### Support vector extension

For RISC-V vector extensions support, see
[additional instructions](#optional-configuration)

## :octicons-sliders-16: Configure and build

### Host configuration

Build and install on your host machine:

``` shell
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install
```

### Target configuration

The following instruction shows how to build for a RISC-V 64-bit Linux machine.
For other RISC-V targets, please refer to
[riscv.toolchain.cmake](https://github.com/openxla/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
as a reference of how to set up the cmake configuration.

#### RISC-V 64-bit Linux target

```shell
cmake -GNinja -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin) \
  -DRISCV_CPU=linux-riscv_64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT} \
  -DIREE_ENABLE_CPUINFO=OFF \
  .
cmake --build ../iree-build-riscv/
```

## :octicons-code-16: Running IREE bytecode modules on the RISC-V system

!!! note
    The following instructions are meant for the RISC-V 64-bit Linux
    target. For the bare-metal target, please refer to
    [simple_embedding](https://github.com/openxla/iree/blob/main/samples/simple_embedding)
    to see how to build a ML workload for a bare-metal machine.

Set the path to qemu-riscv64 Linux emulator binary in the `QEMU_BIN` environment
variable. If it is installed with `riscv_bootstrap.sh`, the path is default at
${HOME}/riscv/qemu/linux/RISCV/bin/qemu-riscv64.

```shell
export QEMU_BIN=<path to qemu-riscv64 binary>
```

Invoke the host compiler tools to produce a bytecode module FlatBuffer:

``` shell
../iree-build/install/bin/iree-compile \
  --iree-hal-target-backends=vmvx \
  samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

Run the RISC-V emulation:

```shell
${QEMU_BIN} \
  -cpu rv64 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/tools/iree-run-module \
  --device=local-task \
  --module=/tmp/simple_abs_vmvx.vmfb \
  --function=abs \
  --input=f32=-5
```

## Optional configuration

[RISC-V Vector extensions](https://github.com/riscv/riscv-v-spec) allows SIMD
 code to run more efficiently. To enable the vector extension for the compiler
 toolchain and the emulator, build the tools from the following sources:

* RISC-V toolchain is built from
<https://github.com/llvm/llvm-project> (main branch).
    * Currently, the LLVM compiler is built on GNU toolchain, including libgcc,
      GNU linker, and C libraries. You need to build GNU toolchain first.
    * Clone GNU toolchain from:
      <https://github.com/riscv/riscv-gnu-toolchain>
      (master branch). Switch the "riscv-binutils" submodule to
      `git://sourceware.org/git/binutils-gdb.git` (master branch) manually.
* RISC-V QEMU is built from
<https://gitlab.com/qemu-project/qemu/tree/v8.1.2>.

The SIMD code can be generated following the
[IREE CPU flow](../guides/deployment-configurations/cpu.md)
with the additional command-line flags

```shell hl_lines="3 4 5 6 7 8"
tools/iree-compile \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-cpu=generic-rv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+zvl512b,+v" \
  --riscv-v-fixed-length-vector-lmul-max=8 \
  iree_input.mlir -o mobilenet_cpu.vmfb
```

Then run on the RISC-V QEMU:

```shell hl_lines="2 5"
${QEMU_BIN} \
  -cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/tools/iree-run-module \
  --device=local-task \
  --module=mobilenet_cpu.vmfb \
  --function=predict \
  --input="1x224x224x3xf32=0"
```
