// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_

// NOTE: this file is designed to be a standalone header: it is embedded in the
// compiler and must not take any dependences on the runtime HAL code.
// Changes here will require changes to the compiler and must be versioned as if
// this was a schema: backwards-incompatible changes require version bumps or
// the ability to feature-detect at runtime.

#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Versioning and interface querying
//===----------------------------------------------------------------------===//

// Known valid version values.
enum iree_hal_executable_library_version_e {
  // iree_hal_executable_library_v0_t is used as the API communication
  // structure.
  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0 = 0u,
};
typedef uint32_t iree_hal_executable_library_version_t;

// The latest version of the library API; can be used to populate the
// iree_hal_executable_library_header_t::version when building libraries.
#define IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION \
  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0

// A header present at the top of all versions of the library API used by the
// runtime to ensure version compatibility.
typedef struct {
  // Version of the API this library was built with, which was likely the value
  // of IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION.
  iree_hal_executable_library_version_t version;

  // Name used for logging/diagnostics.
  const char* name;
} iree_hal_executable_library_header_t;

// Exported function from dynamic libraries for querying library information.
// The provided |max_version| is the maximum version the caller supports;
// callees must return NULL if their lowest available version is greater
// than the max version supported by the caller.
typedef const iree_hal_executable_library_header_t* (
    *iree_hal_executable_library_query_fn_t)(
    iree_hal_executable_library_version_t max_version);

// Function name exported from dynamic libraries (pass to dlsym).
#define IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME \
  "iree_hal_executable_library_query"

//===----------------------------------------------------------------------===//
// IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0
//===----------------------------------------------------------------------===//

// Read-only per-dispatch state passed to each tile in a dispatch.
typedef struct {
  uint32_t reserved;
} iree_hal_executable_dispatch_state_v0_t;

typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t value[3];
} iree_hal_vec3_t;

#if defined(_MSC_VER)
typedef __declspec(
    align(16)) const uint32_t* iree_hal_executable_push_constants_ptr_t;
#else
typedef const uint32_t* iree_hal_executable_push_constants_ptr_t
    __attribute__((align_value(16)));
#endif  // MSVC

typedef void* iree_hal_executable_binding_ptr_t;

// Function signature of exported executable entry points.
// The same |state| is passed to all tiles in a dispatch, with other arguments
// such as |workgroup_id| varying per-tile (counting to the |workgroup_count|).
// Each tile represents |workgroup_size| local invocations in the global
// |workgroup_count| grid.
//
// 0 or more push constants are available at |push_constants| with the count
// being determined by the sidechannel information provided by the compiler.
//
// The |bindings| list is a dense set of pointers to I/O data with the count and
// ordering determined by the compiler.
typedef void (*iree_hal_executable_dispatch_v0_t)(
    const iree_hal_executable_dispatch_state_v0_t* state,
    const iree_hal_vec3_t* workgroup_id, const iree_hal_vec3_t* workgroup_size,
    const iree_hal_vec3_t* workgroup_count,
    const iree_hal_executable_push_constants_ptr_t push_constants,
    const iree_hal_executable_binding_ptr_t* bindings);

// Structure used for v0 library interfaces.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
//
// Implementations may still choose to heap allocate this structure and modify
// at runtime so long as they observe the thread-safety guarantees. For example,
// a JIT may default all entry_points to JIT thunk functions and then swap them
// out for the translated function pointers.
typedef struct {
  // Version/metadata header. Will have a version of
  // IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0.
  const iree_hal_executable_library_header_t* header;

  // The total number of entry points available in the library. Bounds all of
  // the tables below.
  uint32_t entry_point_count;

  // Table of export function entry points matching the ordinals defined during
  // library generation. The runtime will use this table to map the ordinals to
  // function pointers for execution.
  const iree_hal_executable_dispatch_v0_t* entry_points;

  // Optional table of export function entry point names 1:1 with entry_points.
  // These names are only used for tracing/debugging and can be omitted to save
  // binary size.
  const char** entry_point_names;

  // Optional table of entry point tags that describe the entry point in a
  // human-readable format useful for verbose logging. The string values, when
  // present, may be attached to tracing/debugging events related to the entry
  // point.
  const char** entry_point_tags;
} iree_hal_executable_library_v0_t;

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
