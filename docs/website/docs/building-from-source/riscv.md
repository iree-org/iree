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

See instructions in the following links

* [Clang getting started](https://clang.llvm.org/get_started.html)
* [RISC-V GNU toolchain](https://github.com/riscv/riscv-gnu-toolchain)
* [QEMU](https://github.com/qemu/qemu)
* [RISC-V Linux QEMU](https://risc-v-getting-started-guide.readthedocs.io/en/latest/linux-qemu.html)

!!! note
    An environment variable `RISCV_TOOLCHAIN_ROOT` needs
    to be set to the root directory of the installed GNU toolchain. The variable
    can be used in building the RISCV target and a LLVM AOT module.

### Install Prebuilt RISC-V Tools (RISC-V 64-bit Linux toolchain)

Execute the following script to download the prebuilt RISC-V toolchain and QEMU
from the IREE root directory:

```shell
./build_tools/riscv/riscv_bootstrap.sh
```
!!! note
    You also need to set `RISCV_TOOLCHAIN_ROOT`
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

The following instruction shows how to build for a RISC-V 64-bit Linux machine.
For other RISC-V targets, please refer to
[riscv.toolchain.cmake](https://github.com/google/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
as a reference of how to set up the cmake configuration.

#### RISC-V 64-bit Linux target
```shell
cmake -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath ../iree-build-host/install) \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_MLIR=OFF \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT}
  .
```

## Running IREE bytecode modules on the RISC-V system

!!! note
    The following instructions are meant for the RISC-V 64-bit Linux
    target. For the bare-metal target, please refer to
    [simple_embedding](https://github.com/google/iree/blob/main/iree/samples/simple_embedding)
    to see how to build a ML workload for a bare-metal machine.

Set the environment variable `RISCV_TOOLCHAIN_ROOT` if it is not set yet:

```shell
export RISCV_TOOLCHAIN_ROOT=<root directory of the RISC-V GNU toolchain>
```

Set the path to qemu-riscv64 Linux emulator binary in the `QEMU_BIN` environment
variable. If it is installed with `riscv_bootstrap.sh`, the path is default at
${HOME}/riscv/qemu/linux/RISCV/bin/qemu-riscv64.

```shell
export QEMU_BIN=<path to qemu-riscv64 binary>
```

Invoke the host compiler tools produce input files:

``` shell
../iree-build/install/bin/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=vmvx \
  iree/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

Run the RISC-V emulation:

```shell
${QEMU_BIN} \
  -cpu rv64 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/iree/tools/iree-run-module \
  --driver=vmvx \
  --module_file=/tmp/simple_abs_vmvx.vmfb \
  --entry_function=abs \
  --function_input=f32=-5
```
