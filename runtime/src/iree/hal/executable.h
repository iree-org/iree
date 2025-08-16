// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_EXECUTABLE_H_
#define IREE_HAL_EXECUTABLE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// An ordinal of an exported function from an executable.
// Ordinals begin at 0 and go up to the total export count.
typedef uint32_t iree_hal_executable_export_ordinal_t;

typedef struct iree_hal_occupancy_info_t {
  int reserved;
} iree_hal_occupancy_info_t;

// Flags defining executable export behavior.
enum iree_hal_executable_export_flag_bits_e {
  IREE_HAL_EXECUTABLE_EXPORT_FLAG_NONE = 0ull,
  // Contiguous workgroups in workgroup space process data sequentially.
  // Dispatch performance can benefit from scheduling multiple contiguous
  // workgroups on execution units that share caches.
  IREE_HAL_EXECUTABLE_EXPORT_FLAG_SEQUENTIAL = 1ull << 0,
  // Workgroup size is dynamic at dispatch time.
  // The workgroup size specified on the export info is the minimum size and
  // granularity and any dynamic workgroup size chosen must be a multiple.
  IREE_HAL_EXECUTABLE_EXPORT_FLAG_WORKGROUP_SIZE_DYNAMIC = 1ull << 1,
};
typedef uint64_t iree_hal_executable_export_flags_t;

// Reflected information about an executable export.
typedef struct iree_hal_executable_export_info_t {
  // Optional C-style name, if the export had one.
  // May not be present if reflection information was stripped.
  iree_string_view_t name;
  // Flags defining the export behavior.
  iree_hal_executable_export_flags_t flags;
  // Total number of 32-bit constants expected.
  uint16_t constant_count;
  // Total number of bindings expected.
  uint16_t binding_count;
  // Total number of logical parameters.
  uint16_t parameter_count;
  // Static or minimum workgroup size of the export.
  // If IREE_HAL_EXECUTABLE_EXPORT_FLAG_WORKGROUP_SIZE_DYNAMIC is set then
  // any dynamic workgroup size must be a multiple of this value.
  uint32_t workgroup_size[3];
  // Occupancy information hinting at how this export should be scheduled.
  iree_hal_occupancy_info_t occupancy_info;
} iree_hal_executable_export_info_t;

// Specifies the type of a parameter.
enum iree_hal_executable_export_parameter_type_e {
  // Parameter is a constant uniform value.
  // Passed to the dispatch in the constants table. The offset indicates the
  // byte offset from the start of the constants table. The size is the total
  // bytes the constant occupies in the constant table without padding.
  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_CONSTANT = 0,
  // Parameter is a buffer binding.
  // Passed to the dispatch in the binding_ptrs table and the length is
  // available. The offset indicates which binding in the table this parameter
  // maps to. The parameter size is ignored.
  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BINDING = 1,
  // Parameter is a raw buffer pointer.
  // Passed to the dispatch in the constants table and the length is
  // unavailable. The offset indicates the byte offset from the start of the
  // constants table. The size is the width in bytes of the pointer (always the
  // machine pointer width).
  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_TYPE_BUFFER_PTR = 2,
};
typedef uint8_t iree_hal_executable_export_parameter_type_t;

// Defines parameter handling behavior.
enum iree_hal_executable_export_parameter_flag_bits_e {
  IREE_HAL_EXECUTABLE_EXPORT_PARAMETER_FLAG_NONE = 0,
};
typedef uint16_t iree_hal_executable_export_parameter_flags_t;

// Declares properties of a parameter to an export function.
typedef struct iree_hal_executable_export_parameter_t {
  // Type of the parameter.
  iree_hal_executable_export_parameter_type_t type;
  // Size of the parameter in bytes. Does not contain padding.
  uint8_t size;
  // Flags indicating parameter behavior.
  iree_hal_executable_export_parameter_flags_t flags;
  // Parameter name if available, otherwise empty.
  iree_string_view_t name;
  // Offset of the parameter in bytes or binding ordinal, depending on type.
  uint16_t offset;
} iree_hal_executable_export_parameter_t;

// Handle to a loaded executable.
// Loading of executables routes through an executable cache, allowing for
// context-aware scoped caches. HAL implementations can use this to preserve
// JIT'ed executables across processes or reuse executables across device
// instances.
//
// Executables provide one or more entry points that can be dispatched via
// iree_hal_command_buffer_dispatch. Some entry points may represent the same
// computation but specialized in different ways such that the runtime can
// switch strategies and choose between them per-dispatch.
//
//
// Maps (roughly) to vkShaderModule + VkPipeline[].
typedef struct iree_hal_executable_t iree_hal_executable_t;

// Retains the given |executable| for the caller.
IREE_API_EXPORT void iree_hal_executable_retain(
    iree_hal_executable_t* executable);

// Releases the given |executable| from the caller.
IREE_API_EXPORT void iree_hal_executable_release(
    iree_hal_executable_t* executable);

// Returns the total number of exported dispatch functions in the |executable|.
IREE_API_EXPORT iree_host_size_t
iree_hal_executable_export_count(iree_hal_executable_t* executable);

// Returns information about the export with the given |export_ordinal|.
IREE_API_EXPORT iree_status_t iree_hal_executable_export_info(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info);

// Populates the |out_parameters| array with up to |capacity| parameters.
// Returns the total number populated up to |capacity|. Callers should allocate
// the parameter array based on the parameter_count in the export info.
IREE_API_EXPORT iree_status_t iree_hal_executable_export_parameters(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters);

// Finds the export with the given |name| and returns its ordinal if found.
IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_export_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal);

//===----------------------------------------------------------------------===//
// iree_hal_executable_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_t* executable);

  iree_host_size_t(IREE_API_PTR* export_count)(
      iree_hal_executable_t* executable);

  iree_status_t(IREE_API_PTR* export_info)(
      iree_hal_executable_t* executable,
      iree_hal_executable_export_ordinal_t export_ordinal,
      iree_hal_executable_export_info_t* out_info);

  iree_status_t(IREE_API_PTR* export_parameters)(
      iree_hal_executable_t* executable,
      iree_hal_executable_export_ordinal_t export_ordinal,
      iree_host_size_t capacity,
      iree_hal_executable_export_parameter_t* out_parameters);

  iree_status_t(IREE_API_PTR* lookup_export_by_name)(
      iree_hal_executable_t* executable, iree_string_view_t name,
      iree_hal_executable_export_ordinal_t* out_export_ordinal);
} iree_hal_executable_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_executable_vtable_t);

IREE_API_EXPORT void iree_hal_executable_destroy(
    iree_hal_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_H_
