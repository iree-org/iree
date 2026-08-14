# Bare-metal RISC-V 64 sample

Runs a single-op IREE workload (elementwise `arith.mulf` on `4xf32`) on
bare-metal riscv64 under `qemu-system-riscv64` (`-machine virt`, no OS,
semihosted I/O), using the inline HAL. Two runners are built:

- `runner_llvmcpu`: kernels compiled to RISC-V machine code by the `llvm-cpu`
  backend (`--iree-execution-model=inline-dynamic`), loaded at runtime with
  the embedded-ELF loader via the `hal_loader` VM module.
- `runner_vmvx`: kernels as portable VM bytecode (`vmvx-inline`,
  `--iree-execution-model=inline-static`); no loader involved.

The full-HAL path used by `samples/simple_embedding` is not used here because
its synchronous device creation currently requires a proactor, which
`IREE_PLATFORM_GENERIC` does not provide. The inline HAL runs the module
without a HAL device, threads, or a filesystem.

## Building and running

Requires a `riscv-none-elf` GCC toolchain that provides `semihost.specs` and
uses a newlib built with `-mcmodel=medany`, such as
[xPack riscv-none-elf-gcc](https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack),
along with prebuilt IREE host tools and `qemu-system-riscv64`. From the repository
root:

```sh
RISCV_TOOLCHAIN_ROOT=/path/to/xpack-gcc \
IREE_HOST_BIN_DIR=/path/to/host/tools \
  ./build_tools/cmake/build_riscv_baremetal.sh

./samples/baremetal_riscv64/run_qemu.sh
```

## Boot and memory layout

QEMU's `virt` machine places RAM at `0x80000000`, requiring
`-mcmodel=medany` throughout, including for newlib. QEMU's generic loader
loads the ELF and sets CPU 0's PC to its entry point, but does not initialize
a stack. `start.S` sets `sp` to `__stack_top`—128 MiB above the RAM base—and
enables the FPU before tailing into newlib's `crt0`. Semihosting provides
console output and process exit.
