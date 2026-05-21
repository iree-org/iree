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
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;
typedef struct iree_hal_buffer_t iree_hal_buffer_t;

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// Dispatchable function resolved within an executable.
//
// Function values are only meaningful with the executable that returned them.
// They are not stable across executable loads, drivers, process runs, or
// replacement artifacts. Stable cross-artifact identity is the function name.
typedef struct iree_hal_executable_function_t {
  // Executable-local function token.
  uint64_t value;
} iree_hal_executable_function_t;

#define IREE_HAL_EXECUTABLE_FUNCTION_INVALID_VALUE UINT64_MAX

// Returns an invalid executable function value.
static inline iree_hal_executable_function_t
iree_hal_executable_function_invalid(void) {
  return (iree_hal_executable_function_t){
      IREE_HAL_EXECUTABLE_FUNCTION_INVALID_VALUE};
}

// Returns a function value for the given executable-local table index.
static inline iree_hal_executable_function_t
iree_hal_executable_function_from_index(uint32_t index) {
  return (iree_hal_executable_function_t){index};
}

// Returns a function value for a raw executable-local token.
static inline iree_hal_executable_function_t
iree_hal_executable_function_from_value(uint64_t value) {
  return (iree_hal_executable_function_t){value};
}

// Returns true if |function| contains a valid executable-local token.
static inline bool iree_hal_executable_function_is_valid(
    iree_hal_executable_function_t function) {
  return function.value != IREE_HAL_EXECUTABLE_FUNCTION_INVALID_VALUE;
}

// Checks whether |function| is a dense table index into |function_count|.
// This is intended for current executable implementations that use the function
// value as an array index. Implementations with richer token encodings should
// decode the token with their own executable-local rules instead.
static inline bool iree_hal_executable_function_is_index_in_range(
    iree_hal_executable_function_t function, iree_host_size_t function_count) {
  return function.value <= UINT32_MAX && function.value < function_count;
}

// Returns |function| as a dense table index.
// The caller must have checked iree_hal_executable_function_is_index_in_range.
static inline uint32_t iree_hal_executable_function_index(
    iree_hal_executable_function_t function) {
  IREE_ASSERT(function.value <= UINT32_MAX);
  return (uint32_t)function.value;
}

typedef struct iree_hal_occupancy_info_t {
  int reserved;
} iree_hal_occupancy_info_t;

// Flags defining executable function behavior.
enum iree_hal_executable_function_flag_bits_e {
  IREE_HAL_EXECUTABLE_FUNCTION_FLAG_NONE = 0ull,
  // Contiguous workgroups in workgroup space process data sequentially.
  // Dispatch performance can benefit from scheduling multiple contiguous
  // workgroups on execution units that share caches.
  IREE_HAL_EXECUTABLE_FUNCTION_FLAG_SEQUENTIAL = 1ull << 0,
  // Workgroup size is dynamic at dispatch time.
  // The workgroup size specified on the function info is the minimum size and
  // granularity and any dynamic workgroup size chosen must be a multiple.
  IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC = 1ull << 1,
};
typedef uint64_t iree_hal_executable_function_flags_t;

// Reflected information about an executable function.
typedef struct iree_hal_executable_function_info_t {
  // Optional C-style name, if the function had one.
  // May not be present if reflection information was stripped.
  iree_string_view_t name;
  // Flags defining the function behavior.
  iree_hal_executable_function_flags_t flags;
  // Total number of 32-bit constants expected.
  uint16_t constant_count;
  // Total number of bindings expected.
  uint16_t binding_count;
  // Total number of logical parameters.
  uint16_t parameter_count;
  // Static or minimum workgroup size of the function.
  // If IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC is set then
  // any dynamic workgroup size must be a multiple of this value.
  uint32_t workgroup_size[3];
  // Occupancy information hinting at how this function should be scheduled.
  iree_hal_occupancy_info_t occupancy_info;
} iree_hal_executable_function_info_t;

// Specifies the type of a parameter.
enum iree_hal_executable_function_parameter_type_e {
  // Parameter is a constant uniform value.
  // Passed to the dispatch in the constants table. The offset indicates the
  // byte offset from the start of the constants table. The size is the total
  // bytes the constant occupies in the constant table without padding.
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT = 0,
  // Parameter is a buffer binding.
  // Passed to the dispatch in the binding_ptrs table and the length is
  // available. The offset indicates which binding in the table this parameter
  // maps to. The parameter size is ignored.
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING = 1,
  // Parameter is a raw buffer pointer.
  // Passed to the dispatch in the constants table and the length is
  // unavailable. The offset indicates the byte offset from the start of the
  // constants table. The size is the width in bytes of the pointer (always the
  // machine pointer width).
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR = 2,
};
typedef uint8_t iree_hal_executable_function_parameter_type_t;

// Defines parameter handling behavior.
enum iree_hal_executable_function_parameter_flag_bits_e {
  IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE = 0,
};
typedef uint16_t iree_hal_executable_function_parameter_flags_t;

// Declares properties of a parameter to a function.
typedef struct iree_hal_executable_function_parameter_t {
  // Type of the parameter.
  iree_hal_executable_function_parameter_type_t type;
  // Size of the parameter in bytes. Does not contain padding.
  uint8_t size;
  // Flags indicating parameter behavior.
  iree_hal_executable_function_parameter_flags_t flags;
  // Parameter name if available, otherwise empty.
  iree_string_view_t name;
  // Offset of the parameter in bytes or binding ordinal, depending on type.
  uint16_t offset;
} iree_hal_executable_function_parameter_t;

// Handle to a loaded executable.
// Loading of executables routes through an executable cache, allowing for
// context-aware scoped caches. HAL implementations can use this to preserve
// JIT'ed executables across processes or reuse executables across device
// instances.
//
// Executables provide one or more functions that can be dispatched via
// iree_hal_command_buffer_dispatch. Some functions may represent the same
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

// Returns the total number of dispatchable functions in the |executable|.
IREE_API_EXPORT iree_host_size_t
iree_hal_executable_function_count(iree_hal_executable_t* executable);

// Returns information about the given |function|.
IREE_API_EXPORT iree_status_t iree_hal_executable_function_info(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info);

// Populates the |out_parameters| array with up to |capacity| parameters.
// Returns the total number populated up to |capacity|. Callers should allocate
// the parameter array based on the parameter_count in the function info.
IREE_API_EXPORT iree_status_t iree_hal_executable_function_parameters(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters);

// Finds the function with the given |name|.
IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_function_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function);

// Finds the executable global variable with the given |name| and returns a
// device buffer aliasing its storage.
//
// |queue_affinity| selects the device instance for per-device globals. Empty or
// any affinities select an implementation-defined valid device instance. The
// returned buffer retains the executable storage it aliases and must be
// released by the caller.
//
// Returns IREE_STATUS_NOT_FOUND when no such global variable exists and
// IREE_STATUS_UNIMPLEMENTED when the executable format or backend cannot expose
// globals.
IREE_API_EXPORT iree_status_t iree_hal_executable_lookup_global_by_name(
    iree_hal_executable_t* executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_executable_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_t* executable);

  iree_host_size_t(IREE_API_PTR* function_count)(
      iree_hal_executable_t* executable);

  iree_status_t(IREE_API_PTR* function_info)(
      iree_hal_executable_t* executable,
      iree_hal_executable_function_t function,
      iree_hal_executable_function_info_t* out_info);

  iree_status_t(IREE_API_PTR* function_parameters)(
      iree_hal_executable_t* executable,
      iree_hal_executable_function_t function, iree_host_size_t capacity,
      iree_hal_executable_function_parameter_t* out_parameters);

  iree_status_t(IREE_API_PTR* lookup_function_by_name)(
      iree_hal_executable_t* executable, iree_string_view_t name,
      iree_hal_executable_function_t* out_function);

  iree_status_t(IREE_API_PTR* lookup_global_by_name)(
      iree_hal_executable_t* executable, iree_string_view_t name,
      iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer);
} iree_hal_executable_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_executable_vtable_t);

IREE_API_EXPORT void iree_hal_executable_destroy(
    iree_hal_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_H_
