# Getting Started on RISC-V with CMake

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.
-->

This guide walks through cross-compiling IREE core runtime towards the RISC-V
Linux platform. Cross-compiling IREE compilers towards RISC-V is not supported
at the moment.

Cross-compilation involves both a *host* platform and a *target* platform. One
invokes compiler toolchains on the host platform to generate libraries and
executables that can be run on the target platform.

## Prerequisites

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

Build and install at least the compiler tools on your host machine, or install
them from a binary distribution:

```shell
$ cmake -G Ninja -B ../iree-build-host/ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=../iree-build-host/install \
    .
$ cmake --build ../iree-build-host/ --target install
```

### Target configuration

The following instruction shows how to build for the RISC-V 64-bit Linux machine
and 32-bit bare-metal machine. For other RISC-V targets, please refer to
[riscv.toolchain.cmake](https://github.com/google/iree/blob/main/build_tools/cmake/riscv.toolchain.cmake)
as a reference of how to set up the cmake configuration.

#### RISC-V 64-bit Linux target
```shell
$ cmake -G Ninja -B ../iree-build-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath ../iree-build-host/install) \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_SAMPLES=ON \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT} \
  .
```
#### RISC-V 32-bit bare-metal target

For the RISC-V 32-bit bare-metal config, append the following CMake options
```shell
-DRISCV_CPU=rv32-baremetal \
-DIREE_BUILD_TESTS=OFF
```

#### Build target

```shell
$ cmake --build ../iree-build-riscv/
```

## Test on RISC-V QEMU

**NOTE**:The following instructions are meant for the RISC-V 64-bit Linux
target. For the bare-metal target, please refer to
[simple_embedding](https://github.com/google/iree/blob/main/iree/samples/simple_embedding)
to see how to build a ML workload for a bare-metal machine.

Set the environment variable `RISCV_TOOLCHAIN_ROOT` if it is not set yet:

```shell
$ export RISCV_TOOLCHAIN_ROOT=<root directory of the RISC-V GNU toolchain>
```

### VMVX HAL backend

Translate a source MLIR into IREE module:

```shell
$ ../iree-build-host/install/bin/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=vmvx \
  ${PWD}/iree/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

Then run on the RISC-V QEMU:

Set the path to qemu-riscv64 emulator binary in the `QEMU_BIN` environment
variable. If it is installed with `riscv_bootstrap.sh`, the path is default at
${HOME}/riscv/qemu/linux/RISCV/bin/qemu-riscv64.

```shell
$ export QEMU_BIN=<path to qemu-riscv64 binary>
```

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

Output:

```
I ../iree/tools/utils/vm_util.cc:227] Creating driver and device for 'vmvx'...
EXEC @abs
f32=5
```

### Dylib LLVM AOT backend
To compile an IREE module using the Dylib LLVM ahead-of-time (AOT) backend for
a RISC-V target we need to use the corresponding cross-compile toolchain.

Translate a source MLIR into an IREE module:

```shell
$ ../iree-build-host/install/bin/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-target-triple=riscv64 \
  -iree-llvm-target-cpu=sifive-u74 \
  -iree-llvm-target-abi=lp64d \
  ${PWD}/iree/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_dylib.vmfb
```

Then run on the RISC-V QEMU:

```shell
$ ${QEMU_BIN} \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/iree/tools/iree-run-module \
  --driver=dylib \
  --module_file=/tmp/simple_abs_dylib.vmfb \
  --entry_function=abs \
  --function_input=f32=-5
```

Output:

```
I ../iree/tools/utils/vm_util.cc:227] Creating driver and device for 'dylib'...
EXEC @abs
f32=5
```

#### Enable RVV code-gen [Experimental]
Through IREE's vectorization pass and LLVM backend, we can generate RVV
VLS(Vector Length Specific) style codes.

```shell
$ ../iree-build-host/install/bin/iree-translate \
-iree-mlir-to-vm-bytecode-module \
-iree-hal-target-backends=dylib-llvm-aot \
-iree-input-type=mhlo \
-iree-llvm-target-triple=riscv64 \
-iree-llvm-target-cpu=sifive-7-rv64 \
-iree-llvm-target-abi=lp64d \
-iree-llvm-target-cpu-features="+m,+a,+d,+experimental-v" \
-riscv-v-vector-bits-min=256 -riscv-v-fixed-length-vector-lmul-max=8 \
${PWD}/iree/samples/simple_embedding/simple_embedding_test.mlir -o /tmp/output-rvv.vmfb
```

Then run on the RISC-V QEMU:

```shell
$ ${QEMU_BIN} \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/iree/tools/iree-run-module --driver=dylib \
  --driver=dylib \
  --module_file=/tmp/output-rvv.vmfb \
  --entry_function=simple_mul \
  --function_input="4xf32=[1 2 3 4]" \
  --function_input="4xf32=[2 4 6 8]"
```
