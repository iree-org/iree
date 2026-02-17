// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/fd_map.h"

#include "iree/async/operation.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Hash table helpers
//===----------------------------------------------------------------------===//

// Default initial capacity (power of 2).
#define IREE_ASYNC_POSIX_FD_MAP_DEFAULT_CAPACITY 16

// Load factor threshold (70%). Resize when entry_count exceeds this.
#define IREE_ASYNC_POSIX_FD_MAP_LOAD_THRESHOLD_NUM 7
#define IREE_ASYNC_POSIX_FD_MAP_LOAD_THRESHOLD_DEN 10

// Hash function for integer fds.
// Uses multiplicative hashing with a prime multiplier.
static inline iree_host_size_t iree_async_posix_fd_map_hash(
    int fd, iree_host_size_t bucket_count) {
  // Knuth multiplicative hash.
  uint32_t hash = (uint32_t)fd * 2654435761u;
  return hash & (bucket_count - 1);  // bucket_count is power of 2.
}

// Returns true if the slot is available for insertion (empty or tombstone).
static inline bool iree_async_posix_fd_map_slot_available(int slot_fd) {
  return slot_fd == IREE_ASYNC_POSIX_FD_MAP_EMPTY ||
         slot_fd == IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE;
}

// Returns true if the slot is empty (never used or cleared).
static inline bool iree_async_posix_fd_map_slot_empty(int slot_fd) {
  return slot_fd == IREE_ASYNC_POSIX_FD_MAP_EMPTY;
}

// Allocates and initializes bucket storage.
static iree_status_t iree_async_posix_fd_map_allocate_buckets(
    iree_host_size_t bucket_count, iree_allocator_t allocator,
    iree_async_posix_fd_map_entry_t** out_buckets) {
  iree_async_posix_fd_map_entry_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      allocator, bucket_count, sizeof(iree_async_posix_fd_map_entry_t),
      (void**)&buckets));

  // Initialize all slots to empty.
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    buckets[i].fd = IREE_ASYNC_POSIX_FD_MAP_EMPTY;
    buckets[i].handler_type = IREE_ASYNC_POSIX_FD_HANDLER_OPERATION;
    buckets[i].handler = NULL;
  }

  *out_buckets = buckets;
  return iree_ok_status();
}

// Resizes the hash table to the new capacity.
// Hash maps cannot use simple realloc/grow because all entries must be
// rehashed to their new positions in the larger table.
static iree_status_t iree_async_posix_fd_map_resize(
    iree_async_posix_fd_map_t* map, iree_host_size_t new_bucket_count) {
  iree_async_posix_fd_map_entry_t* old_buckets = map->buckets;
  iree_host_size_t old_bucket_count = map->bucket_count;

  // Allocate new bucket array.
  iree_async_posix_fd_map_entry_t* new_buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_async_posix_fd_map_allocate_buckets(
      new_bucket_count, map->allocator, &new_buckets));

  // Rehash all live entries into the new array.
  for (iree_host_size_t i = 0; i < old_bucket_count; ++i) {
    int fd = old_buckets[i].fd;
    if (!iree_async_posix_fd_map_slot_available(fd)) {
      // Live entry - insert into new buckets.
      iree_host_size_t index =
          iree_async_posix_fd_map_hash(fd, new_bucket_count);
      while (!iree_async_posix_fd_map_slot_empty(new_buckets[index].fd)) {
        index = (index + 1) & (new_bucket_count - 1);
      }
      new_buckets[index].fd = fd;
      new_buckets[index].handler_type = old_buckets[i].handler_type;
      new_buckets[index].handler = old_buckets[i].handler;
    }
  }

  // Free old buckets and install new.
  iree_allocator_free(map->allocator, old_buckets);
  map->buckets = new_buckets;
  map->bucket_count = new_bucket_count;
  map->tombstone_count = 0;  // Tombstones are discarded during resize.

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Resize and growth
//===----------------------------------------------------------------------===//

// Checks if resize is needed and performs it.
static iree_status_t iree_async_posix_fd_map_maybe_resize(
    iree_async_posix_fd_map_t* map) {
  // Resize when (entry_count + tombstone_count) exceeds 70% of capacity.
  // This ensures we don't degrade due to tombstone accumulation.
  iree_host_size_t used = map->entry_count + map->tombstone_count;
  iree_host_size_t threshold =
      (map->bucket_count * IREE_ASYNC_POSIX_FD_MAP_LOAD_THRESHOLD_NUM) /
      IREE_ASYNC_POSIX_FD_MAP_LOAD_THRESHOLD_DEN;
  if (used < threshold) {
    return iree_ok_status();
  }

  // Double the capacity. Resize performs a full rehash.
  iree_host_size_t new_bucket_count = map->bucket_count * 2;
  return iree_async_posix_fd_map_resize(map, new_bucket_count);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_fd_map_initialize(
    iree_host_size_t initial_capacity, iree_allocator_t allocator,
    iree_async_posix_fd_map_t* out_map) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_map, 0, sizeof(*out_map));
  out_map->allocator = allocator;

  // Round up to power of 2, minimum default capacity.
  if (initial_capacity < IREE_ASYNC_POSIX_FD_MAP_DEFAULT_CAPACITY) {
    initial_capacity = IREE_ASYNC_POSIX_FD_MAP_DEFAULT_CAPACITY;
  }
  iree_host_size_t bucket_count =
      iree_math_round_up_to_pow2_u64(initial_capacity);

  iree_status_t status = iree_async_posix_fd_map_allocate_buckets(
      bucket_count, allocator, &out_map->buckets);
  if (iree_status_is_ok(status)) {
    out_map->bucket_count = bucket_count;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_posix_fd_map_deinitialize(iree_async_posix_fd_map_t* map) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(map->allocator, map->buckets);
  memset(map, 0, sizeof(*map));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Lookup, insert, and remove
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_fd_map_insert(
    iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t handler_type, void* handler) {
  // Check for resize before insertion.
  IREE_RETURN_IF_ERROR(iree_async_posix_fd_map_maybe_resize(map));

  iree_host_size_t index = iree_async_posix_fd_map_hash(fd, map->bucket_count);
  iree_host_size_t tombstone_index = IREE_HOST_SIZE_MAX;

  // Linear probe until we find empty, tombstone, or matching fd.
  for (;;) {
    int slot_fd = map->buckets[index].fd;

    if (iree_async_posix_fd_map_slot_empty(slot_fd)) {
      // Empty slot - insert here (or at earlier tombstone).
      if (tombstone_index != IREE_HOST_SIZE_MAX) {
        index = tombstone_index;
        map->tombstone_count--;
      }
      map->buckets[index].fd = fd;
      map->buckets[index].handler_type = handler_type;
      map->buckets[index].handler = handler;
      map->entry_count++;
      return iree_ok_status();
    }

    if (slot_fd == IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE) {
      // Remember first tombstone for potential insertion.
      if (tombstone_index == IREE_HOST_SIZE_MAX) {
        tombstone_index = index;
      }
    } else if (slot_fd == fd) {
      // Duplicate key.
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "fd %d already in map", fd);
    }

    // Continue probing.
    index = (index + 1) & (map->bucket_count - 1);
  }
}

iree_status_t iree_async_posix_fd_map_insert_or_chain(
    iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t handler_type, void* handler,
    bool* out_was_chained) {
  *out_was_chained = false;

  // Check for resize before insertion.
  IREE_RETURN_IF_ERROR(iree_async_posix_fd_map_maybe_resize(map));

  iree_host_size_t index = iree_async_posix_fd_map_hash(fd, map->bucket_count);
  iree_host_size_t tombstone_index = IREE_HOST_SIZE_MAX;

  // Linear probe until we find empty, tombstone, or matching fd.
  for (;;) {
    int slot_fd = map->buckets[index].fd;

    if (iree_async_posix_fd_map_slot_empty(slot_fd)) {
      // Empty slot - insert here (or at earlier tombstone).
      if (tombstone_index != IREE_HOST_SIZE_MAX) {
        index = tombstone_index;
        map->tombstone_count--;
      }
      map->buckets[index].fd = fd;
      map->buckets[index].handler_type = handler_type;
      map->buckets[index].handler = handler;
      map->entry_count++;
      return iree_ok_status();
    }

    if (slot_fd == IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE) {
      // Remember first tombstone for potential insertion.
      if (tombstone_index == IREE_HOST_SIZE_MAX) {
        tombstone_index = index;
      }
    } else if (slot_fd == fd) {
      // fd already in map. For OPERATION handlers, chain the new operation.
      if (handler_type == IREE_ASYNC_POSIX_FD_HANDLER_OPERATION &&
          map->buckets[index].handler_type ==
              IREE_ASYNC_POSIX_FD_HANDLER_OPERATION) {
        // Chain at HEAD to get FIFO execution order.
        //
        // The pending_queue is a LIFO stack (atomic_slist_pop returns most
        // recently pushed). If op1 is submitted before op2, they're drained as
        // op2 then op1. By inserting at head, we get: [op2] then [op1 -> op2].
        // Executing from head gives op1 first, which is the correct FIFO order.
        iree_async_operation_t* new_operation =
            (iree_async_operation_t*)handler;
        iree_async_operation_t* existing =
            (iree_async_operation_t*)map->buckets[index].handler;
        new_operation->next = existing;
        map->buckets[index].handler = new_operation;
        *out_was_chained = true;
        return iree_ok_status();
      }
      // Non-OPERATION or mismatched types: duplicate key error.
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "fd %d already in map", fd);
    }

    // Continue probing.
    index = (index + 1) & (map->bucket_count - 1);
  }
}

bool iree_async_posix_fd_map_remove_operation(
    iree_async_posix_fd_map_t* map, int fd, iree_async_operation_t* operation) {
  iree_host_size_t index = iree_async_posix_fd_map_hash(fd, map->bucket_count);

  for (;;) {
    int slot_fd = map->buckets[index].fd;

    if (iree_async_posix_fd_map_slot_empty(slot_fd)) {
      // fd not in map - treat as already removed.
      return true;
    }

    if (slot_fd == fd) {
      // Found the fd entry. Unlink the operation from the chain.
      iree_async_operation_t* head =
          (iree_async_operation_t*)map->buckets[index].handler;

      if (head == operation) {
        // Removing the head of the chain.
        iree_async_operation_t* new_head = operation->next;
        operation->next = NULL;
        if (new_head == NULL) {
          // Chain is now empty - remove fd from map.
          map->buckets[index].fd = IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE;
          map->buckets[index].handler = NULL;
          map->entry_count--;
          map->tombstone_count++;
          return true;  // Chain empty.
        } else {
          // Update head.
          map->buckets[index].handler = new_head;
          return false;  // Chain still has operations.
        }
      } else {
        // Search for the operation in the chain.
        iree_async_operation_t* prev = head;
        while (prev->next != NULL && prev->next != operation) {
          prev = prev->next;
        }
        if (prev->next == operation) {
          // Found it - unlink.
          prev->next = operation->next;
          operation->next = NULL;
        }
        // Chain is non-empty (at least head remains).
        return false;
      }
    }

    // Continue probing.
    index = (index + 1) & (map->bucket_count - 1);
  }
}

bool iree_async_posix_fd_map_lookup(
    const iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t* out_handler_type, void** out_handler) {
  iree_host_size_t index = iree_async_posix_fd_map_hash(fd, map->bucket_count);

  for (;;) {
    int slot_fd = map->buckets[index].fd;

    if (iree_async_posix_fd_map_slot_empty(slot_fd)) {
      return false;
    }

    if (slot_fd == fd) {
      *out_handler_type = map->buckets[index].handler_type;
      *out_handler = map->buckets[index].handler;
      return true;
    }

    // Continue probing (skip tombstones).
    index = (index + 1) & (map->bucket_count - 1);
  }
}

void* iree_async_posix_fd_map_remove(iree_async_posix_fd_map_t* map, int fd) {
  iree_host_size_t index = iree_async_posix_fd_map_hash(fd, map->bucket_count);

  for (;;) {
    int slot_fd = map->buckets[index].fd;

    if (iree_async_posix_fd_map_slot_empty(slot_fd)) {
      return NULL;
    }

    if (slot_fd == fd) {
      void* handler = map->buckets[index].handler;
      map->buckets[index].fd = IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE;
      map->buckets[index].handler = NULL;
      map->entry_count--;
      map->tombstone_count++;
      return handler;
    }

    // Continue probing.
    index = (index + 1) & (map->bucket_count - 1);
  }
}

void iree_async_posix_fd_map_clear(iree_async_posix_fd_map_t* map) {
  for (iree_host_size_t i = 0; i < map->bucket_count; ++i) {
    map->buckets[i].fd = IREE_ASYNC_POSIX_FD_MAP_EMPTY;
    map->buckets[i].handler_type = IREE_ASYNC_POSIX_FD_HANDLER_OPERATION;
    map->buckets[i].handler = NULL;
  }
  map->entry_count = 0;
  map->tombstone_count = 0;
}
