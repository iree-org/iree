// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse slot allocator for io_uring resource tables.
//
// io_uring supports sparse resource tables (kernel 5.19+) where an empty table
// of a fixed capacity is pre-registered, then individual slots are
// populated/cleared dynamically. This applies to both the fixed buffer table
// (IORING_REGISTER_BUFFERS2 + IORING_REGISTER_BUFFERS_UPDATE) and the fixed
// file table (IORING_REGISTER_FILES2 + IORING_REGISTER_FILES_UPDATE2).
//
// This type manages only the bitmap allocation of slots. Kernel syscalls are
// the caller's responsibility, since the register opcode and data format differ
// between buffers (iovec) and files (int fd). The intended usage pattern is:
//
//   iree_io_uring_sparse_table_lock(table);
//   int32_t slot = iree_io_uring_sparse_table_acquire(table, count);
//   if (slot < 0) { unlock; return RESOURCE_EXHAUSTED; }
//   long ret = syscall(IORING_REGISTER_*_UPDATE, ...);
//   if (ret < 0) { iree_io_uring_sparse_table_release(table, slot, count); }
//   iree_io_uring_sparse_table_unlock(table);
//
// The lock ensures the acquire-then-syscall sequence is atomic: no other
// thread can observe or reclaim the same slots between allocation and kernel
// registration.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_SPARSE_TABLE_H_
#define IREE_ASYNC_PLATFORM_IO_URING_SPARSE_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default sparse buffer table capacity (slot count). Bitmap cost is 1KB at
// 8192 slots. A 512-buffer slab consumes 512 contiguous slots, so this
// supports ~16 concurrent slab registrations plus individual dmabuf/file slots
// without fragmentation pressure.
#define IREE_IO_URING_SPARSE_TABLE_DEFAULT_CAPACITY 8192

// A sparse slot allocator for io_uring resource tables.
//
// Wraps an iree_bitmap_t with mutex protection and first-fit contiguous
// allocation semantics. Thread-safe: callers acquire the lock, perform
// allocation + kernel syscall atomically, then release the lock.
//
// Used for both IORING_REGISTER_BUFFERS2 (fixed buffer table) and
// IORING_REGISTER_FILES2 (fixed file table) sparse registrations.
typedef struct iree_io_uring_sparse_table_t {
  iree_slim_mutex_t mutex;
  // Bitmap of allocated slots. bit_count == table capacity.
  // The words array are allocated as trailing data after this struct.
  iree_bitmap_t bitmap;
} iree_io_uring_sparse_table_t;

// Allocates a sparse table with the given slot |capacity|.
// Single allocation: struct + trailing bitmap words. All slots start free.
iree_status_t iree_io_uring_sparse_table_allocate(
    uint16_t capacity, iree_allocator_t allocator,
    iree_io_uring_sparse_table_t** out_table);

// Frees the sparse table. No-op if |table| is NULL.
// The caller must ensure no kernel resource table references remain.
void iree_io_uring_sparse_table_free(iree_io_uring_sparse_table_t* table,
                                     iree_allocator_t allocator);

// Acquires the table lock. Callers must hold across acquire + syscall
// sequences to prevent concurrent slot assignment.
void iree_io_uring_sparse_table_lock(iree_io_uring_sparse_table_t* table);

// Releases the table lock.
void iree_io_uring_sparse_table_unlock(iree_io_uring_sparse_table_t* table);

// Acquires a contiguous range of |count| slots using first-fit strategy.
// Returns the starting index, or -1 if insufficient contiguous space.
// Caller MUST hold the lock.
int32_t iree_io_uring_sparse_table_acquire(iree_io_uring_sparse_table_t* table,
                                           uint16_t count);

// Releases a contiguous range of |count| slots starting at |start|.
// Caller MUST hold the lock.
void iree_io_uring_sparse_table_release(iree_io_uring_sparse_table_t* table,
                                        uint16_t start, uint16_t count);

// Returns the total slot capacity of the table.
static inline uint16_t iree_io_uring_sparse_table_capacity(
    const iree_io_uring_sparse_table_t* table) {
  return (uint16_t)table->bitmap.bit_count;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_SPARSE_TABLE_H_
