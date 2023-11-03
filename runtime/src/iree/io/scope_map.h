// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_SCOPE_MAP_H_
#define IREE_IO_SCOPE_MAP_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_io_scope_map_entry_t {
  iree_string_view_t scope;
  iree_io_parameter_index_t* index;
} iree_io_scope_map_entry_t;

typedef struct iree_io_scope_map_t {
  iree_allocator_t host_allocator;
  iree_host_size_t count;
  iree_host_size_t capacity;
  iree_io_scope_map_entry_t** entries;
} iree_io_scope_map_t;

IREE_API_EXPORT void iree_io_scope_map_initialize(
    iree_allocator_t host_allocator, iree_io_scope_map_t* out_scope_map);

IREE_API_EXPORT void iree_io_scope_map_deinitialize(
    iree_io_scope_map_t* scope_map);

IREE_API_EXPORT iree_status_t iree_io_scope_map_lookup(
    iree_io_scope_map_t* scope_map, iree_string_view_t scope,
    iree_io_parameter_index_t** out_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_SCOPE_MAP_H_
