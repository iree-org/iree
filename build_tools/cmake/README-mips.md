# MIPS64 (VMVX) cross-compilation

Cross-compile the IREE **runtime** for MIPS64 - little-endian (`mips64el`) or
big-endian (`mips64`) - on an x86-64 host and run a model under the QEMU
emulator. The model is compiled to architecture-independent VMVX bytecode on the
host; only the runtime is cross-compiled - there is **no native MIPS code
generation** (that is future work; see the end).

This is the MIPS analog of IREE's
[RISC-V cross-compilation guide](https://iree.dev/building-from-source/riscv/).
Run all commands from the repository root. The little-endian flow is described
first; see [Big-endian (mips64)](#big-endian-mips64) for the big-endian deltas.

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

## Big-endian (mips64)

The runtime also cross-compiles for big-endian MIPS64 (`mips64`, n64). The flow
mirrors the little-endian steps above with a different toolchain and QEMU binary,
plus one extra consideration: embedded constants.

IREE's serialized formats are little-endian-canonical. The runtime byte-swaps
little-endian metadata/bytecode when running on a big-endian host, but bulk
constant data (rodata) is read with native typed loads, so the **compiler** must
emit it in the target's byte order. The `--iree-vm-bytecode-rodata-big-endian`
flag does this; `build_mips.sh` sets it automatically when `MIPS_ENDIAN=big`.

### Toolchain and emulator

```bash
sudo apt install \
  gcc-mips64-linux-gnuabi64 \
  g++-mips64-linux-gnuabi64 \
  qemu-user
```

The sysroot lands at `/usr/mips64-linux-gnuabi64` and the emulator is
`qemu-mips64` (big-endian; the little-endian flow uses `qemu-mips64el`).

### Build and test

```bash
export IREE_HOST_BIN_DIR=$(realpath ../iree-build/install/bin)

MIPS_ENDIAN=big \
CMAKE_TOOLCHAIN_FILE="$(realpath build_tools/cmake/linux_mips64_be_gcc.cmake)" \
MIPS_TOOLCHAIN_PREFIX=mips64-linux-gnuabi64- \
  ./build_tools/cmake/build_mips.sh build-mips-be

file build-mips-be/tools/iree-run-module   # expect: ELF 64-bit MSB, MIPS, N64

MIPS_ENDIAN=big QEMU_BIN=qemu-mips64 QEMU_LD_PREFIX=/usr/mips64-linux-gnuabi64 \
  ./build_tools/cmake/test_mips.sh build-mips-be
```

To run a model, compile it on the host **with the big-endian rodata flag** and
run it under `qemu-mips64`:

```bash
../iree-build/install/bin/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  --iree-vm-bytecode-rodata-big-endian \
  samples/models/mnist.mlir -o /tmp/mnist_vmvx_be.vmfb

qemu-mips64 -cpu MIPS64R2-generic -L /usr/mips64-linux-gnuabi64/ \
  build-mips-be/tools/iree-run-module \
  --device=local-task --module=/tmp/mnist_vmvx_be.vmfb \
  --function=predict --input=1x28x28x1xf32=0.3
```

The big-endian output matches the host (and little-endian) output bit for bit.

### Known limitations on big-endian

- **LE-only serialization features return `UNIMPLEMENTED`.** The HAL replay and
  profile file formats and the FatELF writer are defined as little-endian and
  are not supported on big-endian hosts. All are off the VMVX inference path.
- **Little-endian-hardcoded test fixtures are skipped on big-endian.** A handful
  of unit tests assert specific little-endian byte layouts (e.g. buffer fill
  patterns, opaque hex formatting). The runtime behaves correctly (native byte
  order); only those fixtures' expectations assume little-endian, so
  `test_mips.sh` skips them when `MIPS_ENDIAN=big`.

## Future work (not implemented)

This build is VMVX only. The following are intentionally left for a future effort:

- **Native MIPS code generation** - enabling the LLVM `Mips` backend so
  `iree-compile` emits native machine code (`--iree-hal-local-target-device-backends=llvm-cpu`
  with a `mips64el` triple), including the real ELF relocation implementation.
- **MSA vectorization** and **microkernels / data tiling** for performance.
- A website `building-from-source/mips.md` page (CI already runs the
  little- and big-endian runtime suites under QEMU; see the `pkgci_test_mips64`
  and `pkgci_test_mips64_be` workflows).
