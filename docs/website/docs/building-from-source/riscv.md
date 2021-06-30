# RISC-V cross-compilation

Running on a platform like RISC-V involves cross-compiling from a _host_
platform (e.g. Linux) to a _target_ platform (a specific RISC-V CPU architecture
and operating system):

* IREE's _compiler_ is built on the host and is used there to generate modules
  for the target
* IREE's _runtime_ is built on the host for the target. The runtime is then
  pushed to the target to run natively.

## Prerequisites

### Host environment setup

You should already be able to build IREE from source on your host platform.
Please make sure you have followed the [getting started](./getting-started.md)
steps.

### Install RISC-V cross-compile toolchain and emulator

You'll need a RISC-V LLVM compilation toolchain and a RISC-V enabled QEMU
emulator.

* RISC-V toolchain is built from https://github.com/llvm/llvm-project (main branch).<br>
  * Currently, the LLVM compiler is built on GNU toolchain, including libgcc,
    GNU linker, and C libraries. You need to build GNU toolchain first.<br>
  * Clone GNU toolchain from: https://github.com/riscv/riscv-gnu-toolchain
    (master branch). Switch the "riscv-binutils" submodule to `rvv-1.0.x-zfh`
    branch manually.
* RISC-V QEMU is built from https://github.com/sifive/qemu/tree/v5.2.0-rvv-rvb-zfh

An environment variable `RISCV_TOOLCHAIN_ROOT` needs
to be set to the root directory of the installed GNU toolchain. The variable can
be used in building the RISCV target and a LLVM AOT module.

### Install Prebuilt RISC-V Tools (RISC-V 64-bit Linux toolchain)

Execute the following script to download the prebuilt RISC-V toolchain and QEMU:

```shell
# In IREE source root
$ ./build_tools/riscv/riscv_bootstrap.sh
```
**NOTE**:
* You also need to set `RISCV_TOOLCHAIN_ROOT`
(default at ${HOME}/riscv/toolchain/clang/linux/RISCV).

## Configure and build

### Host configuration

Build and install on your host machine:

``` shell
cmake -B ../iree-build/ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install
```

### Target configuration

Currently IREE verifies (and tests in CI) two RISC-V targets:
* RISC-V 64-bit CPU with Linux OS
* RISC-V 32-bit CPU with "bare-metal" config -- a limited subset of IREE runtime
libraries that _can_ run on the bare-metal machine mode (with the machine BSP
linker script), but can also run on larger machines without the system library
support.

For other RISC-V targets, please refer to
[riscv.toolchain.cmake](https://github.com/google/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
as a reference of how to set up the cmake configuration.

#### RISC-V 64-bit Linux target
```shell
$ cmake -G Ninja -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath ../iree-build-host/install) \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DIREE_BUILD_SAMPLES=ON \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}
  .
```

For the RISC-V 32-bit bare-metal config, append the following CMake options
```shell
-DRISCV_CPU=rv32-baremetal \
-DIREE_BUILD_TESTS=OFF
```

## Running IREE bytecode modules on the RISC-V system

**NOTE**:The following instructions are meant for the RISC-V 64-bit Linux
target. For the bare-metal target, please refer to
[simple_embedding](https://github.com/google/iree/blob/main/iree/samples/simple_embedding)
to see how to build a ML workload for a bare-metal machine.

Set the environment variable `RISCV_TOOLCHAIN_ROOT` if it is not set yet:

```shell
$ export RISCV_TOOLCHAIN_ROOT=<root directory of the RISC-V GNU toolchain>
```

Set the path to qemu-riscv64 Linux emulator binary in the `QEMU_BIN` environment
variable. If it is installed with `riscv_bootstrap.sh`, the path is default at
${HOME}/riscv/qemu/linux/RISCV/bin/qemu-riscv64.

```shell
$ export QEMU_BIN=<path to qemu-riscv64 binary>
```

Invoke the host compiler tools produce input files:

``` shell
$ ../iree-build/install/bin/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=vmvx \
  iree/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

Run the RISC-V emulation:

```shell
$ ${QEMU_BIN} \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/iree/tools/iree-run-module \
  --driver=vmvx \
  --module_file=/tmp/simple_abs_vmvx.vmfb \
  --entry_function=abs \
  --function_input=f32=-5
```
