// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/scope_map.h"

IREE_API_EXPORT void iree_io_scope_map_initialize(
    iree_allocator_t host_allocator, iree_io_scope_map_t* out_scope_map) {
  IREE_TRACE_ZONE_BEGIN(z0);
  out_scope_map->host_allocator = host_allocator;
  out_scope_map->count = 0;
  out_scope_map->capacity = 0;
  out_scope_map->entries = NULL;
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_io_scope_map_deinitialize(
    iree_io_scope_map_t* scope_map) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = scope_map->host_allocator;
  for (iree_host_size_t i = 0; i < scope_map->count; ++i) {
    iree_io_scope_map_entry_t* entry = scope_map->entries[i];
    iree_io_parameter_index_release(entry->index);
    iree_allocator_free(host_allocator, entry);
  }
  iree_allocator_free(host_allocator, scope_map->entries);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_io_scope_map_lookup(
    iree_io_scope_map_t* scope_map, iree_string_view_t scope,
    iree_io_parameter_index_t** out_index) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, scope.data, scope.size);

  for (iree_host_size_t i = 0; i < scope_map->count; ++i) {
    iree_io_scope_map_entry_t* entry = scope_map->entries[i];
    if (iree_string_view_equal(scope, entry->scope)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hit");
      *out_index = entry->index;
      return iree_ok_status();
    }
  }
  IREE_TRACE_ZONE_APPEND_TEXT(z0, "miss");

  if (scope_map->count == scope_map->capacity) {
    iree_host_size_t new_capacity = iree_max(8, scope_map->capacity * 2);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(
                scope_map->host_allocator,
                new_capacity * sizeof(iree_io_scope_map_entry_t*),
                (void**)&scope_map->entries));
    scope_map->capacity = new_capacity;
  }

  iree_io_scope_map_entry_t* entry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(scope_map->host_allocator,
                                sizeof(*entry) + scope.size, (void**)&entry));
  entry->scope =
      iree_make_string_view((const char*)entry + sizeof(*entry), scope.size);
  memcpy((char*)entry->scope.data, scope.data, scope.size);

  iree_status_t status =
      iree_io_parameter_index_create(scope_map->host_allocator, &entry->index);

  if (iree_status_is_ok(status)) {
    scope_map->entries[scope_map->count++] = entry;
    *out_index = entry->index;
  } else {
    iree_allocator_free(scope_map->host_allocator, entry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
