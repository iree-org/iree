// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// API Versioning
// -----------------------------------------------------------------------------
//
// The core IREE runtime is designed to be statically linked into either hosting
// applications or binding layers (python, rust, etc). It is not designed to be
// stable across shared library versions and no guarantees are made about
// exported function signatures or structures. If a user does package the
// runtime in a shared library and exports the symbols they will need to handle
// versioning themselves if attempting to perform version shifting.
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
#include "iree/base/loop.h"            // IWYU pragma: export
#include "iree/base/loop_inline.h"     // IWYU pragma: export
#include "iree/base/status.h"          // IWYU pragma: export
#include "iree/base/string_builder.h"  // IWYU pragma: export
#include "iree/base/string_view.h"     // IWYU pragma: export
#include "iree/base/time.h"            // IWYU pragma: export
#include "iree/base/tracing.h"         // IWYU pragma: export
#include "iree/base/wait_source.h"     // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// IREE Core API
//===----------------------------------------------------------------------===//

// Sprinkle this wherever to make it easier to find structs/functions that are
// not yet stable.
#define IREE_API_UNSTABLE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_API_H_
