// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AMDGPU_METADATA_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AMDGPU_METADATA_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Code Object Metadata
//===----------------------------------------------------------------------===//

// Kernel argument ABI classification from AMDGPU metadata `.value_kind`.
typedef enum iree_hal_amdgpu_metadata_arg_kind_e {
  // Unknown or unsupported value kind.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_UNKNOWN = 0,
  // Argument bytes are copied directly into the kernarg segment.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_BY_VALUE,
  // Argument is a pointer to global memory.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_GLOBAL_BUFFER,
  // Argument is a pointer to dynamically allocated LDS.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_DYNAMIC_SHARED_POINTER,
  // Argument is an image descriptor pointer.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_IMAGE,
  // Argument is a sampler descriptor pointer.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_SAMPLER,
  // Argument is an OpenCL pipe pointer.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_PIPE,
  // Argument is an OpenCL device enqueue queue pointer.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_QUEUE,
  // Argument is a hidden ABI/runtime value.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_HIDDEN,
  // Argument reserves hidden ABI space but does not need a value.
  IREE_HAL_AMDGPU_METADATA_ARG_KIND_HIDDEN_NONE,
} iree_hal_amdgpu_metadata_arg_kind_t;

// Decoded kernel argument metadata.
typedef struct iree_hal_amdgpu_metadata_arg_t {
  // Byte offset in the kernel's kernarg segment.
  uint32_t offset;
  // Byte length of the kernarg storage for this argument.
  uint32_t size;
  // Storage alignment in bytes when explicitly available; otherwise 0.
  uint32_t alignment;
  // Parsed classification of |value_kind|.
  iree_hal_amdgpu_metadata_arg_kind_t kind;
  // Raw `.value_kind` string view borrowed from the metadata blob.
  iree_string_view_t value_kind;
  // Raw `.address_space` string view borrowed from the metadata blob, if any.
  iree_string_view_t address_space;
  // Effective access string view borrowed from the metadata blob, if any.
  // `.actual_access` is preferred over `.access` when both are present.
  iree_string_view_t access;
} iree_hal_amdgpu_metadata_arg_t;

// Decoded per-kernel metadata.
typedef struct iree_hal_amdgpu_metadata_kernel_t {
  // Source-level kernel name from `.name`, if present.
  iree_string_view_t name;
  // Kernel descriptor symbol name from `.symbol`, usually `foo.kd`.
  iree_string_view_t symbol_name;
  // Kernel kernarg segment size from `.kernarg_segment_size`.
  uint32_t kernarg_segment_size;
  // Kernel kernarg segment alignment from `.kernarg_segment_align`.
  uint32_t kernarg_segment_alignment;
  // Fixed group segment size from `.group_segment_fixed_size`.
  uint32_t group_segment_fixed_size;
  // Fixed private segment size from `.private_segment_fixed_size`.
  uint32_t private_segment_fixed_size;
  // Required workgroup size from `.reqd_workgroup_size`, if present.
  uint32_t required_workgroup_size[3];
  // True when |required_workgroup_size| was present.
  bool has_required_workgroup_size;
  // Number of argument records in |args|.
  iree_host_size_t arg_count;
  // Argument records borrowed from the owning metadata object.
  const iree_hal_amdgpu_metadata_arg_t* args;
} iree_hal_amdgpu_metadata_kernel_t;

// Decoded AMDGPU code object metadata.
//
// All string views and |message_pack_data| borrow from |elf_data|. Callers must
// keep the ELF bytes alive for as long as this metadata object is in use.
typedef struct iree_hal_amdgpu_metadata_t {
  // Allocator used for kernel and argument arrays.
  iree_allocator_t host_allocator;
  // Borrowed ELF bytes used as the source of all string views.
  iree_const_byte_span_t elf_data;
  // Borrowed AMDGPU MessagePack note descriptor payload.
  iree_const_byte_span_t message_pack_data;
  // Number of decoded kernels.
  iree_host_size_t kernel_count;
  // Decoded kernel records.
  iree_hal_amdgpu_metadata_kernel_t* kernels;
  // Total number of decoded argument records.
  iree_host_size_t arg_count;
  // Contiguous argument storage referenced by |kernels|.
  iree_hal_amdgpu_metadata_arg_t* args;
} iree_hal_amdgpu_metadata_t;

// Initializes |out_metadata| from a raw AMDGPU ELF code object.
//
// This locates the `AMDGPU`/`NT_AMDGPU_METADATA` note and decodes only the
// fields needed for kernel argument reflection. The parser accepts a normal
// LLVM-produced 64-bit little-endian AMDGPU ELF. It intentionally does not
// implement HIP fat binary, clang offload bundle, or compressed code object
// handling.
iree_status_t iree_hal_amdgpu_metadata_initialize_from_elf(
    iree_const_byte_span_t elf_data, iree_allocator_t host_allocator,
    iree_hal_amdgpu_metadata_t* out_metadata);

// Releases storage owned by |metadata|.
void iree_hal_amdgpu_metadata_deinitialize(
    iree_hal_amdgpu_metadata_t* metadata);

// Finds a decoded kernel by its descriptor symbol name.
iree_status_t iree_hal_amdgpu_metadata_find_kernel_by_symbol(
    const iree_hal_amdgpu_metadata_t* metadata, iree_string_view_t symbol_name,
    const iree_hal_amdgpu_metadata_kernel_t** out_kernel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AMDGPU_METADATA_H_
