// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_LOOP_INLINE_H_
#define IREE_BASE_LOOP_INLINE_H_

#include <inttypes.h>

#include "iree/base/loop.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_loop_inline
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_loop_inline_ctl(void* self,
                                                   iree_loop_command_t command,
                                                   const void* params,
                                                   void** inout_ptr);
IREE_API_EXPORT iree_status_t
iree_loop_inline_using_storage_ctl(void* self, iree_loop_command_t command,
                                   const void* params, void** inout_ptr);

// Returns a loop that doesn't really loop.
// All operations are run as they are enqueued on the stack. This uses no
// additional memory and ensures that everything completes upon return to the
// user but does eliminate the ability for pipelining and overlapping work from
// multiple subprograms. This approach limits the amount of work that can be
// reentrantly scheduled and should only be used when in the tiniest of
// environments with programs tested to be compatible with it.
//
// Reentrant enqueuing is possible and can be used to create tail call chains
// (or recursion) that executes roughly in order.
//
// Caveats:
// - Reentrant enqueuing of operations is limited to some small number (~4).
// - Waits are performed as they are enqueued and the loop must be able to
//   make forward progress on each.
// - Execution deadlines are ignored in order to fully drain on each operation.
// - Errors propagate immediately to the top-level caller and abort all pending
//   operations.
//
// Thread-compatible: stateless and executes all work on the calling thread.
static inline iree_loop_t iree_loop_inline(iree_status_t* out_status) {
  iree_loop_t loop = {out_status, iree_loop_inline_ctl};
  return loop;
}

// Minimum size in bytes required for iree_loop_inline_storage_t.
// If we wanted to shrink this size to the absolute minimum we'd just expose the
// structures here; not the worst thing but messy (as this is a public API).
#define IREE_LOOP_INLINE_STORAGE_SIZE 512

// Storage for an inline loop.
// May be either allocated on the stack or on the heap and only needs to remain
// valid for the lifetime of the iree_loop_t referencing it.
typedef iree_alignas(iree_max_align_t) struct iree_loop_inline_storage_t {
  uint8_t opaque[IREE_LOOP_INLINE_STORAGE_SIZE];
  iree_status_t status;
} iree_loop_inline_storage_t;

// Returns an inline loop that uses an external |storage| instead of the stack.
// The storage will only be used while executing and can be reused if the caller
// knows it is safe (not reentrantly inside of a loop execution). Errors that
// arise will be set in the storage status field and must be checked (or
// ignored) by the caller to avoid leaks.
//
// See iree_loop_inline for details on the execution behavior.
static inline iree_loop_t iree_loop_inline_initialize(
    iree_loop_inline_storage_t* storage) {
  storage->status = iree_ok_status();
  iree_loop_t loop = {
      storage,
      iree_loop_inline_using_storage_ctl,
  };
  return loop;
}

static inline void iree_loop_inline_deinitialize(
    iree_loop_inline_storage_t* storage) {
  if (!storage) return;
  iree_status_ignore(storage->status);
  storage->status = iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_LOOP_INLINE_H_
