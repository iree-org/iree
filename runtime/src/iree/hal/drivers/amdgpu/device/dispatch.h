// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_

#include "iree/hal/drivers/amdgpu/abi/kernel_args.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/device/support/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Dispatch Kernarg Layout
//===----------------------------------------------------------------------===//

// Device-visible kernarg byte layout for one dispatch.
//
// This is intentionally a prevalidated data contract instead of a status-
// producing API: once device-side replay is emitting packets there is no sane
// way to report malformed ABI metadata or recover from partially-written
// packet/kernarg storage. Host recording/submission code must validate kernel
// metadata and user-provided arguments before passing a layout here.
typedef struct iree_hal_amdgpu_device_dispatch_kernarg_layout_t {
  // Size in bytes of the explicitly provided dispatch arguments.
  size_t explicit_kernarg_size;
  // Offset in bytes of the implicit HIP/OpenCL suffix, if present.
  size_t implicit_args_offset;
  // Total kernarg reservation size in bytes required for this dispatch.
  size_t total_kernarg_size;
  // True if a HIP/OpenCL implicit args suffix is appended at
  // |implicit_args_offset| and must be populated during emplace.
  bool has_implicit_args;
} iree_hal_amdgpu_device_dispatch_kernarg_layout_t;

// Returns the HAL ABI kernarg layout for |kernel_args|.
//
// Explicit args are laid out as:
//   uint64_t bindings[kernel_args->binding_count]
//   uint32_t constants[kernel_args->constant_count]
//   zero padding to 8-byte alignment
//
// If kernel metadata declares more bytes than those explicit args, a
// HIP/OpenCL implicit-args suffix is appended at the aligned explicit size and
// the reservation is extended to cover at least
// IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE bytes of suffix storage.
//
// Caller must have validated that kernel_args->kernarg_size is not smaller than
// the explicit HAL ABI size if that is considered malformed for the current
// executable.
static inline iree_hal_amdgpu_device_dispatch_kernarg_layout_t
iree_hal_amdgpu_device_dispatch_make_hal_kernarg_layout(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args) {
  const size_t binding_bytes =
      (size_t)kernel_args->binding_count * sizeof(uint64_t);
  const size_t constant_bytes = iree_amdgpu_align(
      (size_t)kernel_args->constant_count * sizeof(uint32_t), 8);
  const size_t explicit_kernarg_size = binding_bytes + constant_bytes;
  const bool has_implicit_args =
      (size_t)kernel_args->kernarg_size > explicit_kernarg_size;
  const size_t total_kernarg_size =
      has_implicit_args
          ? IREE_AMDGPU_MAX(
                (size_t)kernel_args->kernarg_size,
                explicit_kernarg_size + IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE)
          : explicit_kernarg_size;
  return (iree_hal_amdgpu_device_dispatch_kernarg_layout_t){
      .explicit_kernarg_size = explicit_kernarg_size,
      .implicit_args_offset = explicit_kernarg_size,
      .total_kernarg_size = total_kernarg_size,
      .has_implicit_args = has_implicit_args,
  };
}

// Returns a custom-direct-argument layout for a raw kernarg blob of
// |kernarg_size| bytes.
//
// The caller owns all packing and padding in the raw argument blob. No implicit
// suffix is synthesized in this mode.
static inline iree_hal_amdgpu_device_dispatch_kernarg_layout_t
iree_hal_amdgpu_device_dispatch_make_custom_kernarg_layout(
    size_t kernarg_size) {
  return (iree_hal_amdgpu_device_dispatch_kernarg_layout_t){
      .explicit_kernarg_size = kernarg_size,
      .implicit_args_offset = kernarg_size,
      .total_kernarg_size = kernarg_size,
      .has_implicit_args = false,
  };
}

//===----------------------------------------------------------------------===//
// Dispatch Packet/Kernarg Emission
//===----------------------------------------------------------------------===//

// Populates a kernel dispatch packet body in already-reserved storage.
//
// The caller owns packet header commit, completion-signal assignment, and
// doorbell writes. Zero workgroup counts are preserved verbatim and produce a
// valid zero-grid dispatch packet.
//
// Preconditions:
//   - |kernel_args|, |workgroup_count|, |dispatch_packet|, and |kernarg_ptr|
//     are non-NULL.
//   - |kernel_args->workgroup_size| and
//     |kernel_args->group_segment_size + dynamic_workgroup_local_memory| are
//     valid for the target kernel.
//   - Each grid dimension product
//     |workgroup_count[i] * kernel_args->workgroup_size[i]| fits in uint32_t.
//   - |kernarg_ptr| satisfies |kernel_args->kernarg_alignment|.
void iree_hal_amdgpu_device_dispatch_emplace_packet(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates HAL ABI kernargs in already-reserved storage.
//
// |binding_ptrs| must provide |kernel_args->binding_count| device pointers as
// raw 64-bit values. |constants| must provide
// |kernel_args->constant_count * sizeof(uint32_t)| bytes. Either pointer may be
// NULL when its corresponding count is zero.
//
// Preconditions:
//   - |kernel_args|, |workgroup_count|, |layout|, and |kernarg_ptr| are
//     non-NULL.
//   - |layout| was derived from |kernel_args| using
//     iree_hal_amdgpu_device_dispatch_make_hal_kernarg_layout.
//   - |kernarg_ptr| points to at least |layout->total_kernarg_size| bytes of
//     writable storage.
void iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    const uint32_t workgroup_count[3], uint32_t dynamic_workgroup_local_memory,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    const uint64_t* IREE_AMDGPU_RESTRICT binding_ptrs,
    const uint32_t* IREE_AMDGPU_RESTRICT constants,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

// Populates custom direct kernargs in already-reserved storage.
//
// |custom_kernarg_ptr| must provide |layout->total_kernarg_size| bytes in the
// final kernel ABI shape expected by the target kernel.
//
// Preconditions:
//   - |layout| and |kernarg_ptr| are non-NULL.
//   - |layout| was derived with
//     iree_hal_amdgpu_device_dispatch_make_custom_kernarg_layout.
//   - |custom_kernarg_ptr| is non-NULL when |layout->total_kernarg_size| > 0.
void iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* IREE_AMDGPU_RESTRICT
        layout,
    const void* IREE_AMDGPU_RESTRICT custom_kernarg_ptr,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_DISPATCH_H_
