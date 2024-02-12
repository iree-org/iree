// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_STRING_BUILDER_H_
#define IREE_BASE_STRING_BUILDER_H_

#include <stdbool.h>
#include <string.h>

#include "iree/base/allocator.h"
#include "iree/base/attributes.h"
#include "iree/base/status.h"
#include "iree/base/string_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Lightweight string builder.
// Used to dynamically produce strings in a growable buffer.
//
// Usage:
//  iree_string_builder_t builder;
//  iree_string_builder_initialize(iree_allocator_system(), &builder);
//  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(&builder, "hel"));
//  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(&builder, "lo"));
//  fprintf(stream, "%.*s", (int)iree_string_builder_size(&builder),
//                          iree_string_builder_buffer(&builder));
//  iree_string_builder_deinitialize(&builder);
//
// Usage for preallocation:
//  iree_string_builder_t builder;
//  iree_string_builder_initialize(iree_allocator_null(), &builder);
//  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(&builder, "123"));
//  // str_length is total number of characters (excluding NUL).
//  iree_host_size_t str_length = iree_string_builder_size(builder);
//  iree_string_builder_deinitialize(&builder);
typedef struct iree_string_builder_t {
  // Allocator used for buffer storage.
  // May be iree_allocator_null() to have the builder total up the required
  // size.
  iree_allocator_t allocator;
  // Allocated storage buffer, if any.
  char* buffer;
  // Total length of the string in the buffer in characters (excluding NUL).
  iree_host_size_t size;
  // Total allocated buffer capacity in bytes.
  iree_host_size_t capacity;
} iree_string_builder_t;

// Initializes a string builder in |out_builder| with the given |allocator|.
IREE_API_EXPORT void iree_string_builder_initialize(
    iree_allocator_t allocator, iree_string_builder_t* out_builder);

// Initializes a string builder in |out_builder| using the given storage.
// Once the capacity is reached further appending will fail.
IREE_API_EXPORT void iree_string_builder_initialize_with_storage(
    char* buffer, iree_host_size_t buffer_capacity,
    iree_string_builder_t* out_builder);

// Deinitializes |builder| and releases allocated storage.
IREE_API_EXPORT void iree_string_builder_deinitialize(
    iree_string_builder_t* builder);

// Returns a pointer into the builder storage.
// The pointer is only valid so long as the string builder is initialized and
// unmodified.
IREE_API_EXPORT const char* iree_string_builder_buffer(
    const iree_string_builder_t* builder);

// Returns the total length of the string in the buffer in characters (excluding
// NUL).
IREE_API_EXPORT iree_host_size_t
iree_string_builder_size(const iree_string_builder_t* builder);

// Returns the total allocated buffer capacity in bytes.
IREE_API_EXPORT iree_host_size_t
iree_string_builder_capacity(const iree_string_builder_t* builder);

// Returns a string view into the builder storage.
// The pointer is only valid so long as the string builder is initialized and
// unmodified.
IREE_API_EXPORT iree_string_view_t
iree_string_builder_view(const iree_string_builder_t* builder);

// Releases the storage from the builder and returns ownership to the caller.
// The caller must free the string using the same allocator used by the builder.
// Returns NULL if the string builder is empty.
//
// Usage:
//  iree_string_builder_t builder;
//  iree_string_builder_initialize(iree_allocator_system(), &builder);
//  ...
//  char* buffer = iree_string_builder_take_storage(&builder);
//  iree_host_size_t buffer_size = iree_string_builder_size(&builder);
//  iree_string_builder_deinitialize(&builder);
//  ...
//  iree_allocator_free(iree_allocator_system(), buffer);
IREE_API_EXPORT IREE_MUST_USE_RESULT char* iree_string_builder_take_storage(
    iree_string_builder_t* builder);

// Reserves storage for at least |minimum_capacity|.
IREE_API_EXPORT iree_status_t iree_string_builder_reserve(
    iree_string_builder_t* builder, iree_host_size_t minimum_capacity);

// Resets the string builder length to 0 without releasing storage.
IREE_API_EXPORT void iree_string_builder_reset(iree_string_builder_t* builder);

// Reserves storage for |count| characters (including NUL) and returns a mutable
// pointer in |out_head| for the caller to write the characters.
// The pointer is only valid so long as the string builder is initialized and
// unmodified. No NUL terminator is added by this call.
// |out_head| will be NULL if the string builder is operating in size
// calculation mode.
IREE_API_EXPORT iree_status_t iree_string_builder_append_inline(
    iree_string_builder_t* builder, iree_host_size_t count, char** out_head);

// Appends a string to the builder.
IREE_API_EXPORT iree_status_t iree_string_builder_append_string(
    iree_string_builder_t* builder, iree_string_view_t value);

// Appends a NUL-terminated C string to the builder.
IREE_API_EXPORT iree_status_t iree_string_builder_append_cstring(
    iree_string_builder_t* builder, const char* value);

// Appends a printf-style formatted string to the builder.
IREE_API_EXPORT IREE_PRINTF_ATTRIBUTE(2, 3) iree_status_t
    iree_string_builder_append_format(iree_string_builder_t* builder,
                                      const char* format, ...);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_STRING_BUILDER_H_
