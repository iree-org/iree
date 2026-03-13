// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/resource_table.h"

#include "iree/hal/api.h"

iree_status_t iree_hal_remote_resource_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_hal_remote_resource_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));

  iree_status_t status = iree_allocator_malloc(
      host_allocator, capacity * sizeof(void*), (void**)&out_table->entries);
  if (iree_status_is_ok(status)) {
    memset(out_table->entries, 0, capacity * sizeof(void*));
    status = iree_allocator_malloc(host_allocator, capacity * sizeof(uint16_t),
                                   (void**)&out_table->generations);
  }
  if (iree_status_is_ok(status)) {
    memset(out_table->generations, 0, capacity * sizeof(uint16_t));
    out_table->capacity = capacity;
    out_table->next_slot = 0;
  } else {
    // Partial init cleanup.
    iree_allocator_free(host_allocator, out_table->entries);
    memset(out_table, 0, sizeof(*out_table));
  }
  return status;
}

void iree_hal_remote_resource_table_deinitialize(
    iree_hal_remote_resource_table_t* table, iree_allocator_t host_allocator) {
  if (!table->entries) return;
  for (uint32_t i = 0; i < table->capacity; ++i) {
    if (table->entries[i]) {
      iree_hal_resource_release(table->entries[i]);
    }
  }
  iree_allocator_free(host_allocator, table->entries);
  iree_allocator_free(host_allocator, table->generations);
  memset(table, 0, sizeof(*table));
}

iree_status_t iree_hal_remote_resource_table_assign(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t resource_type, void* resource,
    iree_hal_remote_resource_id_t* out_resource_id) {
  // Linear scan from next_slot for a free entry.
  uint32_t capacity = table->capacity;
  uint32_t start = table->next_slot;
  for (uint32_t i = 0; i < capacity; ++i) {
    uint32_t slot = (start + i) % capacity;
    if (!table->entries[slot]) {
      iree_hal_resource_retain(resource);
      table->entries[slot] = resource;
      uint16_t generation = ++table->generations[slot];
      table->next_slot = (slot + 1) % capacity;

      *out_resource_id = ((uint64_t)resource_type << 56) |
                         ((uint64_t)generation << 32) |
                         ((uint64_t)slot & 0xFFFFFF);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "resource table full (capacity=%u)", capacity);
}

void* iree_hal_remote_resource_table_lookup(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_type_t expected_type,
    iree_hal_remote_resource_id_t resource_id) {
  if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(resource_id) != expected_type) {
    return NULL;
  }
  uint32_t slot = IREE_HAL_REMOTE_RESOURCE_ID_SLOT(resource_id);
  if (slot >= table->capacity) return NULL;

  uint16_t expected_generation =
      IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(resource_id);
  if (table->generations[slot] != expected_generation) return NULL;

  return table->entries[slot];
}

void iree_hal_remote_resource_table_release(
    iree_hal_remote_resource_table_t* table,
    iree_hal_remote_resource_id_t resource_id) {
  uint32_t slot = IREE_HAL_REMOTE_RESOURCE_ID_SLOT(resource_id);
  if (slot >= table->capacity) return;

  uint16_t expected_generation =
      IREE_HAL_REMOTE_RESOURCE_ID_GENERATION(resource_id);
  if (table->generations[slot] != expected_generation) return;

  if (table->entries[slot]) {
    iree_hal_resource_release(table->entries[slot]);
    table->entries[slot] = NULL;
  }
}
