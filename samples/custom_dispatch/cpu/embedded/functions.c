// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: it is not safe to use the standard library functions in here.
// Attempting to allocate memory (outside of small stack allocations), make
// syscalls, use thread-local state, or pull in externally defined standard
// library functions will result in a bad time.
//
// It is safe to include definitions/macros/etc but be veeery careful!
// Embedded ELF shared libraries are intended to be portable across operating
// systems and environments (including to bare-metal systems) and though the
// IREE compiler can ensure it does not pull in things that may run afoul of
// that it's the user's responsibility when injecting code like this.
#include <stddef.h>
#include <stdint.h>

// NOTE: kernels must be exported with C naming (no C++ mangling) in order to
// match the names used in the IR declarations.

// NOTE: IREE ensures all bindings don't alias their active subranges and
// it is safe to mark them as restrict. This is critical as the C compiler can't
// analyze the codegen when compiling and has to play it safe by assuming any
// write to any binding could be visible through other bindings.

// NOTE: memref lowering in MLIR -> LLVM currently expands to two pointers and
// three ints - do not rely on this behavior and only use the first pointer.
// At some point someone will fix upstream to allow for passing raw base
// pointers and the function signatures here will become much less verbose.

// NOTE: MLIR's index type will map to either i32 or i64 based on the target
// pointer width. size_t (or ssize_t) can be used in source to match that type.

// `ret = lhs * rhs`
//
// Conforms to ABI:
// #hal.pipeline.layout<push_constants = 1, sets = [
//   <0, bindings = [
//       <0, storage_buffer, ReadOnly>,
//       <1, storage_buffer, ReadOnly>,
//       <2, storage_buffer>
//   ]>
// ]>
// With a workgroup size of 64x1x1.
void simple_mul_workgroup(
    // vvvv simplification pending (buffer + offset)
    const float* restrict binding0, const float* restrict binding0_aligned,
    size_t binding0_offset, size_t binding0_size, size_t binding0_stride,
    const float* restrict binding1, const float* restrict binding1_aligned,
    size_t binding1_offset, size_t binding1_size, size_t binding1_stride,
    float* restrict binding2, float* restrict binding2_aligned,
    size_t binding2_offset, size_t binding2_size, size_t binding2_stride,
    // ^^^^ simplification pending (buffer + offset)
    size_t dim, size_t tid) {
  size_t end = tid + 64;
  if (end > dim) end = dim;
  for (size_t i = tid; i < end; ++i) {
    binding2[i] = binding0[i] * binding1[i];
  }
}

// `rhs *= lhs`
//
// Conforms to ABI:
// #hal.pipeline.layout<push_constants = 1, sets = [
//   <0, bindings = [
//       <0, storage_buffer, ReadOnly>,
//       <1, storage_buffer>
//   ]>
// ]>
// With a workgroup size of 64x1x1.
void simple_mul_inplace_workgroup(
    // vvvv simplification pending (buffer + offset)
    const float* restrict binding0, const float* restrict binding0_aligned,
    size_t binding0_offset, size_t binding0_size, size_t binding0_stride,
    float* restrict binding1, float* restrict binding1_aligned,
    size_t binding1_offset, size_t binding1_size, size_t binding1_stride,
    // ^^^^ simplification pending (buffer + offset)
    size_t dim, size_t tid) {
  size_t end = tid + 64;
  if (end > dim) end = dim;
  for (size_t i = tid; i < end; ++i) {
    binding1[i] *= binding0[i];
  }
}
