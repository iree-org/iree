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
    The `RISCV_TOOLCHAIN_ROOT` environment variable needs
    to be set to the root directory of the installed GNU toolchain when building
    the RISC-V compiler target and the runtime library.

#### Install prebuilt RISC-V tools (RISC-V 64-bit Linux toolchain)

Execute the following script to download the prebuilt RISC-V toolchain and QEMU
from the IREE root directory:

```shell
./build_tools/riscv/riscv_bootstrap.sh
```

#### Support vector extension

For RISC-V vector extensions support, see
[additional instructions](#optional-configuration)

## Configure and build

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
[riscv.toolchain.cmake](https://github.com/google/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
as a reference of how to set up the cmake configuration.

#### RISC-V 64-bit Linux target

```shell
cmake -GNinja -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath ../iree-build-host/install) \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT} \
  .
```

## Running IREE bytecode modules on the RISC-V system

!!! note
    The following instructions are meant for the RISC-V 64-bit Linux
    target. For the bare-metal target, please refer to
    [simple_embedding](https://github.com/google/iree/blob/main/iree/samples/simple_embedding)
    to see how to build a ML workload for a bare-metal machine.

Set the path to qemu-riscv64 Linux emulator binary in the `QEMU_BIN` environment
variable. If it is installed with `riscv_bootstrap.sh`, the path is default at
${HOME}/riscv/qemu/linux/RISCV/bin/qemu-riscv64.

```shell
export QEMU_BIN=<path to qemu-riscv64 binary>
```

Invoke the host compiler tools to produce a bytecode module flatbuffer:

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

## Optional configuration

[RISC-V Vector extensions](https://github.com/riscv/riscv-v-spec) allows SIMD
 code to run more efficiently. To enable the vector extension for the compiler
 toolchain and the emulator, build the tools from the following sources:

* RISC-V toolchain is built from
[https://github.com/llvm/llvm-project](https://github.com/llvm/llvm-project) (main branch).
    * Currently, the LLVM compiler is built on GNU toolchain, including libgcc,
      GNU linker, and C libraries. You need to build GNU toolchain first.
    * Clone GNU toolchain from:
      [https://github.com/riscv/riscv-gnu-toolchain](https://github.com/riscv/riscv-gnu-toolchain)
      (master branch). Switch the "riscv-binutils" submodule to `rvv-1.0.x-zfh`
      branch manually.
* RISC-V QEMU is built from
[https://github.com/sifive/qemu/tree/v5.2.0-rvv-rvb-zfh](https://github.com/sifive/qemu/tree/v5.2.0-rvv-rvb-zfh).

The SIMD code can be generated following the
[IREE dynamic library CPU HAL driver flow](../deployment-configurations/cpu-dylib.md)
with the additional command-line flags

```shell hl_lines="3 4 5 6 7 8"
iree/tools/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-target-triple=riscv64 \
  -iree-llvm-target-cpu=generic-rv64 \
  -iree-llvm-target-abi=lp64d \
  -iree-llvm-target-cpu-features="+m,+a,+f,+d,+experimental-v" \
  -riscv-v-vector-bits-min=256 -riscv-v-fixed-length-vector-lmul-max=8 \
  iree_input.mlir -o mobilenet-dylib.vmfb
```

Then run on the RISC-V QEMU:

```shell hl_lines="2 5"
${QEMU_BIN} \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/iree/tools/iree-run-module \
  --driver=dylib \
  --module_file=mobilenet-dylib.vmfb \
  --entry_function=predict \
  --function_input="1x224x224x3xf32=0"
```
