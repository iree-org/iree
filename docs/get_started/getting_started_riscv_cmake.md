# Getting Started on RISC-V with CMake

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.
-->

This guide walks through cross-compiling IREE core runtime towards the RISC-V Linux platform. Cross-compiling IREE compilers towards RISC-V is not supported at the moment.

Cross-compilation involves both a *host* platform and a *target* platform. One
invokes compiler toolchains on the host platform to generate libraries and
executables that can be run on the target platform.

## Prerequisites

### Set up host development environment

The host platform should have been set up for developing IREE. Right now only
Linux is supported. Please make sure you have followed the steps for
[Linux](./getting_started_linux_cmake.md).

### Install RISC-V Tools

Execute the following script to download RISC-V toolchain and QEMU:

```shell
# In IREE source root
$ ./build_tools/riscv/riscv_bootstrap.sh
```

* RISC-V toolchain is built from https://github.com/llvm/llvm-project
* RISC-V QEMU is built from https://github.com/sifive/qemu/tree/v5.1.0-rvv-zfh-pmp

## Configure and build

### Configure on Linux

```shell
$ mkdir build-riscv ; cd build-riscv
$ cmake -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="../build_tools/cmake/riscv.toolchain.cmake" \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HOST_C_COMPILER=`which clang` -DIREE_HOST_CXX_COMPILER=`which clang++` ..
```

*   The above configures IREE to cross-compile towards `rv64` cpu platform.
*   If user specify different download path (default in `${HOME}/riscv`) in `riscv_bootscrap.sh`, please append `-DRISCV_TOOL_PATH="/path/to/the/downloaded/folder"` in cmake command.

### Build all targets

```shell
$ cmake --build .
```

## Test on RISC-V QEMU

### VMLA HAL backend

Translate a source MLIR into IREE module:

```shell
# Still in "build-riscv" folder.
$ ./host/iree/tools/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=vmla \
  ../iree/tools/test/simple.mlir \
  -o /tmp/simple-vmla.vmfb
```

Then run on the RISC-V QEMU:

```shell
$ $HOME/riscv/qemu/linux/RISCV/bin/qemu-riscv64 \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L $HOME/riscv/toolchain/clang/linux/RISCV/sysroot/ \
  ./iree/tools/iree-run-module -driver=vmla \
  -module_file=/tmp/simple-vmla.vmfb \
  -entry_function=abs \
  -function_inputs="i32=-5"
```

Output:

```
I ../iree/tools/utils/vm_util.cc:227] Creating driver and device for 'vmla'...
EXEC @abs
I ../iree/tools/utils/vm_util.cc:172] result[0]: Buffer<sint32[]>
i32=5
```

### Dylib LLVM AOT backend
To compile an IREE module using the Dylib LLVM ahead-of-time (AOT) backend for
a target RISC-V we need to use the corresponding toolchain which we have downloaded at the `$HOME/riscv` folder.
Set the AOT linker path environment variable:

```shell
# Still in "build-riscv" folder
$ export IREE_LLVMAOT_LINKER_PATH="$HOME/riscv/toolchain/clang/linux/RISCV/bin/clang++"
```

Translate a source MLIR into an IREE module:

```shell
$ ./host/iree/tools/iree-translate \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-target-triple=riscv64 \
  -iree-llvm-target-cpu=sifive-u74 \
  -iree-llvm-target-abi=lp64d \
  ../iree/tools/test/simple.mlir \
  -o /tmp/simple-llvm_aot.vmfb
```

Then run on the RISC-V QEMU:

```shell
$ $HOME/riscv/qemu/linux/RISCV/bin/qemu-riscv64 \
  -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 \
  -L $HOME/riscv/toolchain/clang/linux/RISCV/sysroot/ \
  ./iree/tools/iree-run-module -driver=dylib \
  -module_file=/tmp/simple-llvm_aot.vmfb \
  -entry_function=abs \
  -function_inputs="i32=-5"
```

Output:

```
I ../iree/tools/utils/vm_util.cc:227] Creating driver and device for 'dylib'...
EXEC @abs
I ../iree/tools/utils/vm_util.cc:172] result[0]: Buffer<sint32[]>
i32=5
```
