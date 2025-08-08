#!/bin/bash
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Compile test kernels for multiple targets (AMDGPU, SPIR-V, CPU)

set -e

KERNEL_DIR="$(dirname "$0")"
OUTPUT_DIR="${KERNEL_DIR}/compiled"

echo "Compiling streaming test kernels..."
mkdir -p "${OUTPUT_DIR}"

# List of kernels to compile
KERNELS=("vector_add" "matrix_multiply")

# Compile for AMDGPU (multiple architectures)
compile_amdgpu() {
  local kernel=$1
  local archs=("gfx900" "gfx906" "gfx1030" "gfx1100")

  for arch in "${archs[@]}"; do
    echo "  Compiling ${kernel} for AMDGPU ${arch}..."
    clang -x c -std=c23 \
      -target amdgcn-amd-amdhsa \
      -march="${arch}" \
      -nogpulib \
      -fgpu-rdc \
      -fno-short-wchar \
      -fno-ident \
      -Xclang -finclude-default-header \
      -fvisibility=hidden \
      -O3 \
      -c "${KERNEL_DIR}/${kernel}.c" \
      -o "${OUTPUT_DIR}/${kernel}.${arch}.o" 2>/dev/null
    lld \
      -flavor gnu \
      -m elf64_amdgpu \
      --build-id=none \
      --no-undefined \
      -shared \
      -plugin-opt=mcpu=${arch} \
      -plugin-opt=O3 \
      --lto-CGO3 \
      --no-whole-archive \
      --gc-sections \
      --print-gc-sections \
      --strip-debug \
      --discard-all \
      --discard-locals \
      -o "${OUTPUT_DIR}/${kernel}.${arch}.so" \
      "${OUTPUT_DIR}/${kernel}.${arch}.o"
    rm "${OUTPUT_DIR}/${kernel}.${arch}.o"
  done
}

# Compile for Vulkan (SPIR-V)
compile_spirv() {
  local kernel=$1
  echo "  Compiling ${kernel} for Vulkan SPIR-V..."

  # Try to compile to SPIR-V
  # Note: This requires clang with SPIR-V support
  clang --target=spirv64-unknown-vulkan1.3 \
    -O3 \
    -c "${KERNEL_DIR}/${kernel}.c" \
    -o "${OUTPUT_DIR}/${kernel}.spv" 2>/dev/null || {
      echo "    Warning: SPIR-V compilation not available, creating placeholder"
      # Create a placeholder file
      echo "SPIR-V compilation requires clang with SPIR-V backend" > "${OUTPUT_DIR}/${kernel}.spv.txt"
    }
}

# Compile for CPU (native shared library using IREE executable library interface)
compile_cpu() {
  local kernel=$1
  echo "  Compiling ${kernel} for CPU (IREE executable library)..."

  # Check if CPU-specific version exists
  if [ -f "${KERNEL_DIR}/${kernel}_cpu.c" ]; then
    # Compile the CPU-specific version as a shared library
    clang -O0 -march=native \
      -fPIC -shared \
      -g \
      -I./runtime/src \
      "${KERNEL_DIR}/${kernel}_cpu.c" \
      -o "${OUTPUT_DIR}/${kernel}.cpu.so"
  else
    # Fallback: compile generic version as object file
    echo "    Warning: No CPU-specific version found, using generic fallback"
    clang -O3 -march=native \
      -fPIC -shared \
      -I./runtime/src \
      -c "${KERNEL_DIR}/${kernel}.c" \
      -o "${OUTPUT_DIR}/${kernel}.cpu.so"
  fi
}

# Main compilation loop
for kernel in "${KERNELS[@]}"; do
  if [ ! -f "${KERNEL_DIR}/${kernel}.c" ]; then
    echo "Warning: ${kernel}.c not found, skipping"
    continue
  fi

  echo "Compiling ${kernel}..."
  compile_amdgpu "$kernel"
  compile_spirv "$kernel"
  compile_cpu "$kernel"
done

echo "Kernel compilation complete. Output in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}/"
