# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Bare-metal riscv64 single-op inference runners for QEMU's virt machine.
# Requires a riscv-none-elf GCC cross build (build_tools/cmake/generic_riscv64_gcc.cmake)
# and host tools; see build_tools/cmake/build_riscv_baremetal.sh

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Generic" OR
   NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64" OR
   NOT IREE_HOST_BIN_DIR)
  return()
endif()

# The link options below rely on GCC spec files (semihost.specs).
if(NOT CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(STATUS
      "Skipping samples/baremetal_riscv64: requires a riscv-none-elf GCC "
      "toolchain with semihosting support (CMAKE_C_COMPILER_ID is "
      "${CMAKE_C_COMPILER_ID})")
  return()
endif()

# QEMU virt: RAM starts at 0x80000000. start.S sets the stack pointer to
# 0x88000000 (128 MiB above the RAM base) and the stack grows downward.
set(_BAREMETAL_QEMU_VIRT_LINKOPTS
  "--specs=semihost.specs"
  "-Wl,-Ttext-segment=0x80000000"
  "-Wl,--defsym=__stack_top=0x88000000"
  "-Wl,-e,_enter"
)

iree_bytecode_module(
  NAME
    simple_mul_module_llvmcpu
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_samples_baremetal_riscv64_module_llvmcpu"
  FLAGS
    "--iree-execution-model=inline-dynamic"
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
    "--iree-llvmcpu-target-triple=riscv64-unknown-unknown-elf"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-vm-bytecode-module-strip-source-map=true"
    "--iree-vm-emit-polyglot-zip=false"
  PUBLIC
)

iree_bytecode_module(
  NAME
    simple_mul_module_vmvx
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_samples_baremetal_riscv64_module_vmvx"
  FLAGS
    "--iree-execution-model=inline-static"
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=vmvx-inline"
    "--iree-llvmcpu-debug-symbols=false"
    "--iree-vm-bytecode-module-strip-source-map=true"
    "--iree-vm-emit-polyglot-zip=false"
  PUBLIC
)

iree_cc_binary(
  NAME
    runner_llvmcpu
  SRCS
    "runner.c"
    "start.S"
  DEFINES
    "USE_LLVMCPU=1"
  DEPS
    ::simple_mul_module_llvmcpu_c
    iree::base
    iree::hal
    iree::hal::local::loaders::embedded_elf_loader
    iree::modules::hal::inline
    iree::modules::hal::loader
    iree::modules::hal::types
    iree::vm
    iree::vm::bytecode::module
  LINKOPTS
    ${_BAREMETAL_QEMU_VIRT_LINKOPTS}
)

iree_cc_binary(
  NAME
    runner_vmvx
  SRCS
    "runner.c"
    "start.S"
  DEPS
    ::simple_mul_module_vmvx_c
    iree::base
    iree::hal
    iree::modules::hal::inline
    iree::modules::hal::types
    iree::vm
    iree::vm::bytecode::module
  LINKOPTS
    ${_BAREMETAL_QEMU_VIRT_LINKOPTS}
)
