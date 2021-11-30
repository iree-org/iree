// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// API Versioning
// -----------------------------------------------------------------------------
//
// The C API is designed to be versioned such that breaking changes either in
// ABI (data types, struct sizes, etc) or signatures (function arguments change)
// will result in a bump of the IREE_API_VERSION_LATEST value.
//
// When linked in statically the runtime should never have a version conflict,
// however dynamic linking where the runtime is a shared object loaded at
// runtime (via dlopen/etc) must always verify the version is as expected.
//
// In the current experimental state of the runtime the API may break frequently
// and the version is pinned at 0.
//
// Example:
//   void* library = dlopen("iree_rt.so", RTLD_LAZY | RTLD_LOCAL);
//   iree_api_version_t actual_version;
//   iree_status_t status = \
//       ((PFN_iree_api_version_check)dlsym(library, "iree_api_version_check"))(
//       IREE_API_VERSION_LATEST, &actual_version);
//   IREE_CHECK_OK(status);
//   dlclose(library);
//
// Object Ownership and Lifetime
// -----------------------------------------------------------------------------
//
// The API follows the CoreFoundation ownership policies:
// https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFMemoryMgmt/Concepts/Ownership.html
//
// These boil down to:
// * Objects returned from *_create or *_copy functions are owned by the caller
//   and must be released when the caller no longer needs them.
// * Objects returned from accessors are not owned by the caller and must be
//   retained by the caller if the object lifetime needs to be extended.
// * Objects passed to functions by argument may be retained by the callee if
//   required.
//
// Example:
//   iree_file_mapping_t* file_mapping;
//   s = iree_file_mapping_open_read(..., &file_mapping);
//   // file_mapping is now owned by this function.
//   s = iree_file_mapping_some_call(file_mapping, ...);
//   // Must release ownership when no longer required.
//   s = iree_file_mapping_release(file_mapping);
//
// String Formatting
// -----------------------------------------------------------------------------
//
// Functions that produce variable-length strings follow a standard usage
// pattern with the arguments:
//   `iree_host_size_t buffer_capacity`: total bytes including \0 available.
//   `char* buffer`: optional buffer to write into.
//   `iree_host_size_t* out_buffer_length`: required/actual length excluding \0.
//
// To query the size required for the output and allocate storage:
//   iree_host_size_t required_length = 0;
//   iree_format_xyz(/*buffer_capacity=*/0, /*buffer=*/NULL, &required_length);
//   iree_host_size_t buffer_capacity = required_length + 1;
//   char* buffer = iree_allocator_malloc(buffer_capacity);
//   iree_host_size_t actual_length = 0;
//   iree_format_xyz(buffer_capacity, buffer, &actual_length);
//   ASSERT(required_length == actual_length);
//
// To handle fixed-length maximum strings (common):
//   // Fails if the string is longer than 127 characters (127 + \0 >= 128).
//   char buffer[128];
//   IREE_RETURN_IF_ERROR(iree_format_xyz(sizeof(buffer), buffer, NULL));
//
// Try fixed-length and fallback to a dynamic allocation:
//   char inline_buffer[128];
//   iree_host_size_t required_length = 0;
//   iree_status_t inline_status = iree_format_xyz(sizeof(inline_buffer),
//                                                 inline_buffer,
//                                                 &required_length);
//   if (iree_status_is_out_of_range(inline_status)) {
//     // Spilled inline_buffer, need to allocate required_length bytes and
//     // try again.
//     // ... see above for example ...
//   } else if (iree_status_is_ok(inline_status)) {
//     // Fit inside inline_buffer, required_length contains actual length.
//   } else {
//     return inline_status;
//   }

#ifndef IREE_BASE_API_H_
#define IREE_BASE_API_H_

#include "iree/base/alignment.h"       // IWYU pragma: export
#include "iree/base/allocator.h"       // IWYU pragma: export
#include "iree/base/assert.h"          // IWYU pragma: export
#include "iree/base/attributes.h"      // IWYU pragma: export
#include "iree/base/bitfield.h"        // IWYU pragma: export
#include "iree/base/config.h"          // IWYU pragma: export
#include "iree/base/status.h"          // IWYU pragma: export
#include "iree/base/string_builder.h"  // IWYU pragma: export
#include "iree/base/string_view.h"     // IWYU pragma: export
#include "iree/base/time.h"            // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

// Sprinkle this wherever to make it easier to find structs/functions that are
// not yet stable.
#define IREE_API_UNSTABLE

// Known versions of the API that can be referenced in code.
// Out-of-bounds values are possible in forward-versioned changes.
typedef enum iree_api_version_e {
  IREE_API_VERSION_0 = 0,
  // Always set to the latest version of the library from source.
  IREE_API_VERSION_LATEST = IREE_API_VERSION_0,
} iree_api_version_t;

// Checks whether the |expected_version| of the caller matches the implemented
// version of |out_actual_version|. Forward compatibility of the API is
// supported but backward compatibility is not: newer binaries using older
// shared libraries of the runtime will fail.
//
// Returns IREE_STATUS_OUT_OF_RANGE if the actual version is not compatible with
// the expected version.
IREE_API_EXPORT iree_status_t
iree_api_version_check(iree_api_version_t expected_version,
                       iree_api_version_t* out_actual_version);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_API_H_
