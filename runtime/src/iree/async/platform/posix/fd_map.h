// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Hash table mapping file descriptors to typed handlers.
//
// Used by the POSIX proactor as the central dispatch table: every fd monitored
// by the event_set has a corresponding entry here with its handler type and
// pointer. On poll readiness the proactor does a single O(1) lookup to find
// and dispatch the correct handler.
//
// The hash table approach (vs dense array) handles arbitrary fd values without
// memory explosion for sparse or high-numbered fds.

#ifndef IREE_ASYNC_PLATFORM_POSIX_FD_MAP_H_
#define IREE_ASYNC_PLATFORM_POSIX_FD_MAP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_operation_t iree_async_operation_t;

//===----------------------------------------------------------------------===//
// Handler types
//===----------------------------------------------------------------------===//

// Identifies the type of handler stored in an fd_map entry.
// Used for dispatch after poll readiness: a single O(1) hash lookup returns
// both the handler pointer and its type, replacing the cascading linked-list
// searches that previously ran per ready fd.
typedef enum iree_async_posix_fd_handler_type_e {
  // One-shot async I/O operation (socket read/write/connect/accept).
  // Removed from the map after dispatch; re-inserted on EAGAIN retry.
  IREE_ASYNC_POSIX_FD_HANDLER_OPERATION = 0,
  // Persistent event source with user callback.
  // Stays in the map until explicitly unregistered.
  IREE_ASYNC_POSIX_FD_HANDLER_EVENT_SOURCE = 1,
  // Notification (may have pending async waits and/or relay subscribers).
  // Stays in the map while at least one consumer exists.
  IREE_ASYNC_POSIX_FD_HANDLER_NOTIFICATION = 2,
  // Primitive-source relay (owns its own fd).
  // Stays in the map until relay is unregistered or fires (one-shot).
  IREE_ASYNC_POSIX_FD_HANDLER_RELAY = 3,
  // One-shot fence import (external fd → semaphore signal).
  // Removed from the map after dispatch.
  IREE_ASYNC_POSIX_FD_HANDLER_FENCE_IMPORT = 4,
} iree_async_posix_fd_handler_type_t;

//===----------------------------------------------------------------------===//
// FD map
//===----------------------------------------------------------------------===//

// Sentinel values for empty and tombstone slots.
#define IREE_ASYNC_POSIX_FD_MAP_EMPTY (-1)
#define IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE (-2)

// Hash table entry. Stored inline in the buckets array.
// Layout: {int fd, uint8_t handler_type, void* handler} = 16 bytes on 64-bit
// (same size as the previous {int fd, operation*} entry due to alignment).
typedef struct iree_async_posix_fd_map_entry_t {
  int fd;  // Key (or EMPTY/TOMBSTONE sentinel).
  iree_async_posix_fd_handler_type_t handler_type;  // Type tag for dispatch.
  void* handler;  // Value (NULL if empty/tombstone).
} iree_async_posix_fd_map_entry_t;

// Hash table mapping fd (int) to operation pointer.
//
// Implementation: Open addressing with linear probing.
// Load factor threshold: 70% triggers resize.
// Tombstones: Used for deletion to maintain probe chains.
typedef struct iree_async_posix_fd_map_t {
  iree_async_posix_fd_map_entry_t* buckets;
  // Always a power of 2.
  iree_host_size_t bucket_count;
  iree_host_size_t entry_count;  // Excludes tombstones.
  iree_host_size_t tombstone_count;
  iree_allocator_t allocator;
} iree_async_posix_fd_map_t;

// Initializes an empty fd map with the given initial capacity.
// |initial_capacity| is rounded up to the next power of 2.
// Pass 0 for a reasonable default (16).
iree_status_t iree_async_posix_fd_map_initialize(
    iree_host_size_t initial_capacity, iree_allocator_t allocator,
    iree_async_posix_fd_map_t* out_map);

// Deinitializes the fd map and frees bucket storage.
void iree_async_posix_fd_map_deinitialize(iree_async_posix_fd_map_t* map);

// Returns true if the map is empty.
static inline bool iree_async_posix_fd_map_is_empty(
    const iree_async_posix_fd_map_t* map) {
  return map->entry_count == 0;
}

// Returns the number of entries in the map.
static inline iree_host_size_t iree_async_posix_fd_map_size(
    const iree_async_posix_fd_map_t* map) {
  return map->entry_count;
}

// Inserts an fd → handler mapping.
// Returns ALREADY_EXISTS if the fd is already in the map (except for OPERATION
// handlers, which support chaining - see insert_or_chain).
// May resize the table if load factor exceeds threshold.
iree_status_t iree_async_posix_fd_map_insert(
    iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t handler_type, void* handler);

// Inserts an OPERATION handler, chaining if the fd already has operations.
//
// For OPERATION handlers, multiple operations can be pending on the same fd
// (e.g., concurrent sends, or send + recv). When an fd already has an OPERATION
// entry, this function chains the new operation onto the existing chain using
// the operation's `next` pointer (FIFO order - new operations go to tail).
//
// For non-OPERATION handlers, this is equivalent to fd_map_insert (returns
// ALREADY_EXISTS if fd already present).
//
// |out_was_chained| is set to true if the operation was chained onto an
// existing entry. The caller uses this to determine whether to modify the
// event_set events mask (first operation: add, chained: modify if needed).
//
// May resize the table if load factor exceeds threshold.
iree_status_t iree_async_posix_fd_map_insert_or_chain(
    iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t handler_type, void* handler,
    bool* out_was_chained);

// Looks up a handler by fd.
// Returns true if found, writing the handler type and pointer to the output
// parameters. Returns false if the fd is not in the map.
bool iree_async_posix_fd_map_lookup(
    const iree_async_posix_fd_map_t* map, int fd,
    iree_async_posix_fd_handler_type_t* out_handler_type, void** out_handler);

// Removes an fd from the map and returns the associated handler.
// Returns NULL if the fd was not in the map. The caller is expected to already
// know the handler type from a prior lookup or from context.
//
// WARNING: For OPERATION handlers that may have chained operations, use
// iree_async_posix_fd_map_remove_operation instead to properly remove a
// specific operation from the chain.
void* iree_async_posix_fd_map_remove(iree_async_posix_fd_map_t* map, int fd);

// Removes a specific operation from an fd's operation chain.
//
// This handles the case where multiple operations are chained on the same fd.
// The operation is unlinked from the chain:
// - If it's the only operation, the fd is removed from the map entirely
// - If it's the head, the next operation becomes the new head
// - If it's in the middle or tail, it's simply unlinked
//
// Returns true if the chain is now empty (fd was removed from map).
// Returns false if other operations remain on this fd.
// Returns true with no-op if the fd was not in the map.
//
// |operation| must be in the chain for the given fd; behavior is undefined
// if the operation is not found in the chain.
bool iree_async_posix_fd_map_remove_operation(
    iree_async_posix_fd_map_t* map, int fd, iree_async_operation_t* operation);

// Clears all entries from the map without deallocating bucket storage.
void iree_async_posix_fd_map_clear(iree_async_posix_fd_map_t* map);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_FD_MAP_H_
