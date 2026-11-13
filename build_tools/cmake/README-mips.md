# MIPS64 (VMVX) cross-compilation

Cross-compile the IREE **runtime** for MIPS64 little-endian (`mips64el`) on an
x86-64 host and run a model under the QEMU emulator. The model is compiled to
architecture-independent VMVX bytecode on the host; only the runtime is
cross-compiled - there is **no native MIPS code generation** (that is future work;
see the end).

This is the MIPS analog of IREE's
[RISC-V cross-compilation guide](https://iree.dev/building-from-source/riscv/).
Run all commands from the repository root.

## Prerequisites

### Host build

You should already be able to
[build IREE from source](https://iree.dev/building-from-source/getting-started/).
The host build provides `iree-compile`, which turns a model into a VMVX bytecode
module - no MIPS backend is needed on the host.

### MIPS toolchain and emulator

Unlike RISC-V, IREE ships no prebuilt MIPS toolchain. On Debian/Ubuntu, install
the distro GNU cross toolchain, the target sysroot, and QEMU user-mode:

```bash
sudo apt install \
  gcc-mips64el-linux-gnuabi64 \
  g++-mips64el-linux-gnuabi64 \
  libc6-dev-mips64el-cross \
  binutils-mips64el-linux-gnuabi64 \
  qemu-user-static
```

The sysroot lands at `/usr/mips64el-linux-gnuabi64`. Verify end to end with a
non-IREE program:

```bash
echo 'int main(){return 42;}' > /tmp/t.c
mips64el-linux-gnuabi64-gcc -static /tmp/t.c -o /tmp/t
qemu-mips64el-static /tmp/t; echo "exit=$?"   # expect exit=42
file /tmp/t                                    # expect: ELF 64-bit LSB, MIPS, N64
```

## Configure and build

### Host

```bash
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo .
cmake --build ../iree-build/ --target install
```

### Target (MIPS64)

The simplest path is the `build_mips.sh` wrapper, which configures, builds, and
installs the runtime with the correct flags (VMVX only, GNU linker, samples off):

```bash
export IREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin)
./build_tools/cmake/build_mips.sh build-mips
```

Equivalent direct CMake invocation, if you prefer:

```bash
cmake -GNinja -B build-mips/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/linux_mips64_gcc.cmake" \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin) \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_ENABLE_LLD=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=OFF \
  .
cmake --build build-mips/
```

Notable flags (differ from RISC-V): `IREE_ENABLE_LLD=OFF` (use `ld.bfd`; LLD's
MIPS support is weak), `IREE_BUILD_SAMPLES=OFF` (the `simple_embedding` samples
have no MIPS branch), and the driver/backend flags selecting VMVX only.

Confirm the result is a MIPS binary:

```bash
file build-mips/tools/iree-run-module   # expect: ELF 64-bit LSB, MIPS, N64
```

## Running a model under QEMU

Compile a sample model **on the host** to VMVX bytecode, then run it under QEMU:

```bash
../iree-build/install/bin/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  samples/models/simple_abs.mlir -o /tmp/simple_abs_vmvx.vmfb

qemu-mips64el -cpu MIPS64R2-generic -L /usr/mips64el-linux-gnuabi64/ \
  build-mips/tools/iree-run-module \
  --device=local-task --module=/tmp/simple_abs_vmvx.vmfb \
  --function=abs --input=f32=-5
# expect: f32=5
```

`-L` points QEMU's loader at the MIPS sysroot so it can find `libc`/`libm`.

> If QEMU prints a flood of syscall traces, `unset QEMU_STRACE` - QEMU enables
> strace whenever that variable is set to *anything*, including `QEMU_STRACE=0`.

### Real model, cross-checked against the host

To confirm real compute (matmul, activation, softmax), compile MNIST and compare
the MIPS output against the host output for the same input - they must match:

```bash
../iree-build/install/bin/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  samples/models/mnist.mlir -o /tmp/mnist_vmvx.vmfb

ARGS="--device=local-task --module=/tmp/mnist_vmvx.vmfb --function=predict --input=1x28x28x1xf32=0.3"

../iree-build/install/bin/iree-run-module $ARGS                            # host
qemu-mips64el -cpu MIPS64R2-generic -L /usr/mips64el-linux-gnuabi64/ \
  build-mips/tools/iree-run-module $ARGS                                   # MIPS
```

Both print the same `1x10xf32` distribution (finite, sums to ~1.0).

### Runtime test suite

`test_mips.sh` runs the runtime CTest suite under QEMU. It auto-detects `QEMU_BIN`
and the sysroot, so no prefix is needed:

```bash
./build_tools/cmake/test_mips.sh build-mips
```

It excludes a known set of tests that fail under QEMU with SIGBUS (unaligned
access in binary/data parsers unrelated to VMVX execution) and prints them each
run.

> **Host-tools path.** This guide uses the out-of-tree layout
> (`../iree-build/install/bin/…`). An in-tree build with no install step puts the
> host tools in `build/tools/` instead - adjust the paths accordingly (they may
> already be on your `PATH`).

## Future work (not implemented)

This build is VMVX only. The following are intentionally left for a future effort:

- **Native MIPS code generation** - enabling the LLVM `Mips` backend so
  `iree-compile` emits native machine code (`--iree-hal-local-target-device-backends=llvm-cpu`
  with a `mips64el` triple), including the real ELF relocation implementation.
- **MSA vectorization** and **microkernels / data tiling** for performance.
- A website `building-from-source/mips.md` page (CI already runs the mips64el
  runtime suite under QEMU; see the `pkgci_test_mips64` workflow).
