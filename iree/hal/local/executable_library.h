// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_

// NOTE: this file is designed to be a standalone header: it is embedded in the
// compiler and must not take any dependences on the runtime HAL code.
// Changes here will require changes to the compiler and must be versioned as if
// this was a schema: backwards-incompatible changes require version bumps or
// the ability to feature-detect at runtime.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Common utilities included to reduce dependencies
//===----------------------------------------------------------------------===//

// `restrict` keyword, not supported by some older compilers.
// We define our own macro in case dependencies use `restrict` differently.
#if defined(_MSC_VER) && _MSC_VER >= 1900
#define IREE_RESTRICT __restrict
#elif defined(_MSC_VER)
#define IREE_RESTRICT
#elif defined(__cplusplus)
#define IREE_RESTRICT __restrict__
#else
#define IREE_RESTRICT restrict
#endif  // _MSC_VER

//===----------------------------------------------------------------------===//
// Runtime feature support metadata
//===----------------------------------------------------------------------===//

// Defines a bitfield of features that the library requires or supports.
enum iree_hal_executable_library_feature_bits_t {
  IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE = 0u,
  // TODO(benvanik): declare features for debugging/coverage/printf/etc.
  // These will control which symbols are injected into the library at runtime.
};
typedef uint32_t iree_hal_executable_library_features_t;

// Defines a set of supported sanitizers that libraries may be compiled with.
// Loaders can use this declaration to check as to whether the library is
// compatible with the hosting environment for cases where the sanitizer
// requires host support.
typedef enum iree_hal_executable_library_sanitizer_kind_e {
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE = 0,
  // Indicates the library is compiled to use AddressSanitizer:
  // https://clang.llvm.org/docs/AddressSanitizer.html
  // Equivalent compiler flag: -fsanitize=address
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS = 1,
  // Indicates the library is compiled to use MemorySanitizer:
  // https://clang.llvm.org/docs/MemorySanitizer.html
  // Equivalent compiler flag: -fsanitize=memory
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MEMORY = 2,
  // Indicates the library is compiled to use ThreadSanitizer:
  // https://clang.llvm.org/docs/ThreadSanitizer.html
  // Equivalent compiler flag: -fsanitize=thread
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD = 3,
  // Indicates the library is compiled to use UndefinedBehaviorSanitizer:
  // https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
  // Equivalent compiler flag: -fsanitize=undefined
  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_UNDEFINED = 4,

  IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MAX_ENUM = INT32_MAX,
} iree_hal_executable_library_sanitizer_kind_t;

//===----------------------------------------------------------------------===//
// Versioning and interface querying
//===----------------------------------------------------------------------===//

// Known valid version values.
typedef enum iree_hal_executable_library_version_e {
  // iree_hal_executable_library_v0_t is used as the API communication
  // structure.
  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0 = 0,

  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_MAX_ENUM = INT32_MAX,
} iree_hal_executable_library_version_t;
static_assert(sizeof(iree_hal_executable_library_version_t) == 4, "uint32_t");

// The latest version of the library API; can be used to populate the
// iree_hal_executable_library_header_t::version when building libraries.
#define IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION \
  IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0

// A header present at the top of all versions of the library API used by the
// runtime to ensure version compatibility.
typedef struct iree_hal_executable_library_header_t {
  // Version of the API this library was built with, which was likely the value
  // of IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION.
  iree_hal_executable_library_version_t version;
  // Name used for logging/diagnostics.
  const char* name;
  // Bitfield of features required/supported by this executable.
  iree_hal_executable_library_features_t features;
  // Which sanitizer the library is compiled to use, if any.
  // Libraries meant for use with a particular sanitizer will are only usable
  // with hosting code that is using the same sanitizer.
  iree_hal_executable_library_sanitizer_kind_t sanitizer;
} iree_hal_executable_library_header_t;

// Exported function from dynamic libraries for querying library information.
// The provided |max_version| is the maximum version the caller supports;
// callees must return NULL if their lowest available version is greater
// than the max version supported by the caller.
typedef const iree_hal_executable_library_header_t** (
    *iree_hal_executable_library_query_fn_t)(
    iree_hal_executable_library_version_t max_version, void* reserved);

// Function name exported from dynamic libraries (pass to dlsym).
#define IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME \
  "iree_hal_executable_library_query"

//===----------------------------------------------------------------------===//
// IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0
//===----------------------------------------------------------------------===//

// TBD: do not use this yet.
typedef struct iree_hal_executable_import_table_v0_t {
  size_t import_count;
  void* import_fns;
} iree_hal_executable_import_table_v0_t;

typedef union iree_hal_vec3_t {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t value[3];
} iree_hal_vec3_t;

// Read-only per-dispatch state passed to each workgroup in a dispatch.
typedef struct iree_hal_executable_dispatch_state_v0_t {
  // Total workgroup count for the dispatch. This is sourced from either the
  // original dispatch call (for iree_hal_command_buffer_dispatch) or the
  // indirection buffer (for iree_hal_command_buffer_dispatch_indirect).
  iree_hal_vec3_t workgroup_count;
  // Workgroup size chosen for the dispatch. For compilation modes where the
  // workgroup size is constant this may be ignored.
  iree_hal_vec3_t workgroup_size;

  // Total number of available 4 byte push constant values in |push_constants|.
  size_t push_constant_count;
  // |push_constant_count| values.
  const uint32_t* push_constants;

  // Total number of binding base pointers in |binding_ptrs| and
  // |binding_lengths|. The set is packed densely based on which bindings are
  // used (known at compile-time).
  size_t binding_count;
  // Base pointers to each binding buffer.
  void* const* binding_ptrs;
  // The length of each binding in bytes, 1:1 with |binding_ptrs|.
  const size_t* binding_lengths;

  // Optional imported functions available for use within the executable.
  const iree_hal_executable_import_table_v0_t* imports;
} iree_hal_executable_dispatch_state_v0_t;

// Function signature of exported executable entry points.
// The same |dispatch_state| is passed to all workgroups in a dispatch while
// |workgroup_id| will vary for each workgroup.
//
// Returns 0 on success and non-zero on failure. Failures will cause device loss
// and should only be used to communicate serious issues that should abort all
// execution within the current device. Buffer overflows are a good example of
// a useful failure though the HAL does not mandate that all overflows are
// caught and only that they are not harmful - clamping byte ranges and never
// returning a failure is sufficient.
typedef int (*iree_hal_executable_dispatch_v0_t)(
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id);

// Structure used for v0 library interfaces.
// The entire structure is designed to be read-only and able to live embedded in
// the binary .rdata section.
//
// Implementations may still choose to heap allocate this structure and modify
// at runtime so long as they observe the thread-safety guarantees. For example,
// a JIT may default all entry_points to JIT thunk functions and then swap them
// out for the translated function pointers.
typedef struct iree_hal_executable_library_v0_t {
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
  const char* const* entry_point_names;
  // Optional table of entry point tags that describe the entry point in a
  // human-readable format useful for verbose logging. The string values, when
  // present, may be attached to tracing/debugging events related to the entry
  // point.
  const char* const* entry_point_tags;

  // TODO(benvanik): optional import declarations.
} iree_hal_executable_library_v0_t;

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_H_
