// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/loop_inline.h"

#include "iree/base/assert.h"
#include "iree/base/tracing.h"

static iree_status_t iree_loop_inline_reentrant_ctl(void* self,
                                                    iree_loop_command_t command,
                                                    const void* params,
                                                    void** inout_ptr);

static void iree_loop_inline_emit_error(iree_loop_t loop, iree_status_t status);

//===----------------------------------------------------------------------===//
// Inline execution of operations
//===----------------------------------------------------------------------===//

// IREE_LOOP_COMMAND_CALL
static void iree_loop_inline_run_call(iree_loop_t loop,
                                      iree_loop_call_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ideally a tail call (when not tracing).
  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, iree_ok_status());
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// IREE_LOOP_COMMAND_DISPATCH
static void iree_loop_inline_run_dispatch(iree_loop_t loop,
                                          iree_loop_dispatch_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  // We run all workgroups before issuing the completion callback.
  // If any workgroup fails we exit early and pass the failing status back to
  // the completion handler exactly once.
  uint32_t workgroup_count_x = params.workgroup_count_xyz[0];
  uint32_t workgroup_count_y = params.workgroup_count_xyz[1];
  uint32_t workgroup_count_z = params.workgroup_count_xyz[2];
  iree_status_t workgroup_status = iree_ok_status();
  for (uint32_t z = 0; z < workgroup_count_z; ++z) {
    for (uint32_t y = 0; y < workgroup_count_y; ++y) {
      for (uint32_t x = 0; x < workgroup_count_x; ++x) {
        workgroup_status =
            params.workgroup_fn(params.callback.user_data, loop, x, y, z);
        if (!iree_status_is_ok(workgroup_status)) goto workgroup_failed;
      }
    }
  }
workgroup_failed:

  // Fire the completion callback with either success or the first error hit by
  // a workgroup.
  // Ideally a tail call (when not tracing).
  status =
      params.callback.fn(params.callback.user_data, loop, workgroup_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// IREE_LOOP_COMMAND_WAIT_UNTIL
static void iree_loop_inline_run_wait_until(
    iree_loop_t loop, iree_loop_wait_until_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  bool did_wait = iree_wait_until(params.deadline_ns);

  iree_status_t status = params.callback.fn(
      params.callback.user_data, loop,
      did_wait ? iree_ok_status()
               : iree_make_status(IREE_STATUS_ABORTED,
                                  "sleep was aborted by a signal/alert"));
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// IREE_LOOP_COMMAND_WAIT_ONE
static void iree_loop_inline_run_wait_one(iree_loop_t loop,
                                          iree_loop_wait_one_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_timeout_t timeout = iree_make_deadline(params.deadline_ns);

  // Try waiting on the wait source directly; this is usually the most optimal
  // implementation when available and for others may drop down to a system
  // wait primitive.
  iree_status_t wait_status =
      iree_wait_source_wait_one(params.wait_source, timeout);

  // Callback after wait, whether it succeeded or failed.
  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, wait_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// IREE_LOOP_COMMAND_WAIT_ANY
static void iree_loop_inline_run_wait_any(
    iree_loop_t loop, iree_loop_wait_multi_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_timeout_t timeout = iree_make_deadline(params.deadline_ns);

  // Do a scan down the wait sources to see if any are already set - if so we
  // can bail early. Otherwise we need to wait on any one.
  // iree_wait_any is a much more efficient (and fair) way but this keeps the
  // code working on bare-metal.
  iree_status_t wait_status = iree_status_from_code(IREE_STATUS_DEFERRED);
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    iree_status_code_t wait_status_code = IREE_STATUS_OK;
    iree_status_t query_status =
        iree_wait_source_query(params.wait_sources[i], &wait_status_code);
    if (iree_status_is_ok(query_status)) {
      if (wait_status_code == IREE_STATUS_OK) {
        // Signaled - can bail early.
        break;
      } else if (wait_status_code == IREE_STATUS_DEFERRED) {
        // Not signaled yet - keep scanning.
        continue;
      } else {
        // Wait failed - can bail early.
        wait_status = iree_status_from_code(wait_status_code);
        break;
      }
    } else {
      // Failed to perform the query, which we treat the same as a wait error.
      wait_status = query_status;
      break;
    }
  }
  if (iree_status_is_deferred(wait_status)) {
    // No queries resolved/failed - commit any real wait.
    // We choose the first one to be (somewhat) deterministic but really it
    // should be randomized... or if the user cares they should use a real loop.
    wait_status = iree_wait_source_wait_one(params.wait_sources[0], timeout);
  }

  // Callback after wait, whether it succeeded or failed.
  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, wait_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

// IREE_LOOP_COMMAND_WAIT_ALL
static void iree_loop_inline_run_wait_all(
    iree_loop_t loop, iree_loop_wait_multi_params_t params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_timeout_t timeout = iree_make_deadline(params.deadline_ns);

  // Run down the list waiting on each source.
  // iree_wait_all is a much more efficient way but this keeps the code working
  // on bare-metal.
  iree_status_t wait_status = iree_ok_status();
  for (iree_host_size_t i = 0; i < params.count; ++i) {
    wait_status = iree_wait_source_wait_one(params.wait_sources[i], timeout);
    if (!iree_status_is_ok(wait_status)) break;
  }

  // Callback after wait, whether it succeeded or failed.
  iree_status_t status =
      params.callback.fn(params.callback.user_data, loop, wait_status);
  if (!iree_status_is_ok(status)) {
    iree_loop_inline_emit_error(loop, status);
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_loop_inline_ring_t
//===----------------------------------------------------------------------===//

// Total capacity of the ringbuffer in operations pending.
// The usable capacity is always 1 less than this as we mask it off,
// unfortunately wasting a slot but keeping this all stupid simple. If we wanted
// to drop another ~32B of stack space we could make this do the right thing.
#define IREE_LOOP_INLINE_RING_CAPACITY ((uint8_t)8)
static_assert((IREE_LOOP_INLINE_RING_CAPACITY &
               (IREE_LOOP_INLINE_RING_CAPACITY - 1)) == 0,
              "ringbuffer capacity must be a power of two");

// Bitmask used to perform a quick mod of the ringbuffer indices.
// This must always be ANDed with the indices before use:
//   uint8_t physical_idx = logical_idx % IREE_LOOP_INLINE_RING_CAPACITY;
// or this, way better (though the compiler can usually figure it out):
//   uint8_t physical_idx = logical_idx & IREE_LOOP_INLINE_RING_CAPACITY;
#define IREE_LOOP_INLINE_RING_MASK (IREE_LOOP_INLINE_RING_CAPACITY - 1)

// An operation in the inline loop ringbuffer containing all the information
// required to replay it at a future time. All pointers are unowned.
typedef struct iree_loop_inline_op_t {
  iree_loop_command_t command;
  union {
    iree_loop_callback_t callback;
    union {
      iree_loop_call_params_t call;
      iree_loop_dispatch_params_t dispatch;
      iree_loop_wait_until_params_t wait_until;
      iree_loop_wait_one_params_t wait_one;
      iree_loop_wait_multi_params_t wait_multi;
    } params;
  };
} iree_loop_inline_op_t;

// Returns the size of the parameters required by |command|.
static inline uint8_t iree_loop_params_size(iree_loop_command_t command) {
  // Keep this a tail call switch; compilers can work magic here.
  switch (command) {
    case IREE_LOOP_COMMAND_CALL:
      return sizeof(iree_loop_call_params_t);
    case IREE_LOOP_COMMAND_DISPATCH:
      return sizeof(iree_loop_dispatch_params_t);
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      return sizeof(iree_loop_wait_until_params_t);
    case IREE_LOOP_COMMAND_WAIT_ONE:
      return sizeof(iree_loop_wait_one_params_t);
    case IREE_LOOP_COMMAND_WAIT_ANY:
    case IREE_LOOP_COMMAND_WAIT_ALL:
      return sizeof(iree_loop_wait_multi_params_t);
    default:
      return 0;
  }
}

// Fixed-size ringbuffer of commands enqueued reentrantly.
// We ensure the size stays small so we don't blow the stack of tiny systems.
// The inline loop is explicitly not designed for multi-program cooperative
// scheduling and well-formed programs shouldn't hit the limit.
//
// NOTE: this structure must be in an initialized state if zeroed.
typedef struct iree_loop_inline_ring_t {
  iree_loop_inline_op_t ops[IREE_LOOP_INLINE_RING_CAPACITY];
  uint8_t read_head;
  uint8_t write_head;
  iree_status_t* status_ptr;
} iree_loop_inline_ring_t;
static_assert(
    sizeof(iree_loop_inline_ring_t) <= IREE_LOOP_INLINE_STORAGE_SIZE,
    "iree_loop_inline_ring_t needs to be tiny as it's allocated on the stack");

// Returns a loop that references the current ringbuffer for reentrant usage.
static inline iree_loop_t iree_loop_inline_reentrant(
    iree_loop_inline_ring_t* ring) {
  iree_loop_t loop = {
      .self = ring,
      .ctl = iree_loop_inline_reentrant_ctl,
  };
  return loop;
}

// Initializes |out_ring| for use.
// We don't clear the ops as we (hopefully) don't use them unless they are valid
// as defined by the ringbuffer parameters.
static inline void iree_loop_inline_ring_initialize(
    iree_status_t* status_ptr, iree_loop_inline_ring_t* out_ring) {
  out_ring->read_head = 0;
  out_ring->write_head = 0;
  out_ring->status_ptr = status_ptr;
}

// Returns true if the ringbuffer is empty (read has caught up to write).
static inline bool iree_loop_inline_ring_is_empty(
    const iree_loop_inline_ring_t* ring) {
  return ring->read_head == ring->write_head;
}

// Returns true if the ringbuffer is full (write has caught up to read).
static inline bool iree_loop_inline_ring_is_full(
    const iree_loop_inline_ring_t* ring) {
  return ((ring->write_head - ring->read_head) & IREE_LOOP_INLINE_RING_MASK) ==
         IREE_LOOP_INLINE_RING_MASK;
}

// Enqueues an operation into |ring|, capacity-permitting.
// |params| is copied into the ringbuffer and need not remain live upon return.
static iree_status_t iree_loop_inline_enqueue(iree_loop_inline_ring_t* ring,
                                              iree_loop_command_t command,
                                              const void* params) {
  // The only thing we need to do here is memcpy the params into our ring.
  // Since all the params differ in size we just effectively perform a lookup
  // and do the copy.
  uint8_t params_size = iree_loop_params_size(command);
  if (IREE_UNLIKELY(params_size) == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unimplemented loop command");
  }

  // Ensure there's space for the new operation.
  if (iree_loop_inline_ring_is_full(ring)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "inline ringbuffer capacity exceeded; reduce the amount of concurrent "
        "work or use a real loop implementation");
  }

  // Reserve a slot for the new operation.
  uint8_t slot = ring->write_head;
  ring->write_head = (ring->write_head + 1) & IREE_LOOP_INLINE_RING_MASK;

  // Copy the operation in; the params are on the stack and won't be valid after
  // the caller returns.
  ring->ops[slot].command = command;
  memcpy(&ring->ops[slot].params, params, params_size);
  return iree_ok_status();
}

// Dequeues the next operation in |ring| and executes it.
// The operation may reentrantly enqueue more operations.
static void iree_loop_inline_dequeue_and_run_next(
    iree_loop_inline_ring_t* ring) {
  IREE_ASSERT(!iree_loop_inline_ring_is_empty(ring));

  // Acquire the next operation.
  uint8_t slot = ring->read_head;
  ring->read_head = (ring->read_head + 1) & IREE_LOOP_INLINE_RING_MASK;

  // Copy out the parameters; the operation we execute may overwrite them by
  // enqueuing more work.
  iree_loop_inline_op_t op = ring->ops[slot];

  // We pass the callbacks a loop that has the reentrancy bit set.
  // This allows iree_loop_inline_ctl to determine whether it needs to alloc
  // more stack space.
  iree_loop_t loop = iree_loop_inline_reentrant(ring);

  // Tail call into the execution routine so we can hopefully tail call all the
  // way up the stack.
  // Ideally these are all tail calls.
  switch (op.command) {
    case IREE_LOOP_COMMAND_CALL:
      iree_loop_inline_run_call(loop, op.params.call);
      break;
    case IREE_LOOP_COMMAND_DISPATCH:
      iree_loop_inline_run_dispatch(loop, op.params.dispatch);
      break;
    case IREE_LOOP_COMMAND_WAIT_UNTIL:
      iree_loop_inline_run_wait_until(loop, op.params.wait_until);
      break;
    case IREE_LOOP_COMMAND_WAIT_ONE:
      iree_loop_inline_run_wait_one(loop, op.params.wait_one);
      break;
    case IREE_LOOP_COMMAND_WAIT_ANY:
      iree_loop_inline_run_wait_any(loop, op.params.wait_multi);
      break;
    case IREE_LOOP_COMMAND_WAIT_ALL:
      iree_loop_inline_run_wait_all(loop, op.params.wait_multi);
      break;
    default:
      break;
  }
}

// Aborts all operations in the ring and resets it to its initial state.
static void iree_loop_inline_abort_all(iree_loop_inline_ring_t* ring) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Issue the completion callback of each op to notify it of the abort.
  // To prevent enqueuing more work while aborting we pass in a NULL loop.
  // We can't do anything with the errors so we ignore them.
  while (!iree_loop_inline_ring_is_empty(ring)) {
    uint8_t slot = ring->read_head;
    ring->read_head = (ring->read_head + 1) & IREE_LOOP_INLINE_RING_MASK;
    iree_loop_callback_t callback = ring->ops[slot].callback;
    iree_status_ignore(callback.fn(callback.user_data, iree_loop_null(),
                                   iree_make_status(IREE_STATUS_ABORTED)));
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_loop_inline_emit_error(iree_loop_t loop,
                                        iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, iree_status_code_string(iree_status_code(status)));

  iree_loop_inline_ring_t* ring = (iree_loop_inline_ring_t*)loop.self;
  if (ring->status_ptr && iree_status_is_ok(*ring->status_ptr)) {
    *ring->status_ptr = status;
  } else {
    iree_status_ignore(status);
  }

  iree_loop_inline_abort_all(ring);

  IREE_TRACE_ZONE_END(z0);
}

// Runs the |ring| until it is empty or an operation fails.
static iree_status_t iree_loop_inline_run_all(iree_loop_inline_ring_t* ring) {
  IREE_TRACE_ZONE_BEGIN(z0);

  do {
    // Dequeue the next op and run it inline.
    iree_loop_inline_dequeue_and_run_next(ring);
  } while (!iree_loop_inline_ring_is_empty(ring));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_loop_inline_ctl functions
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_loop_inline_ctl(void* self,
                                                   iree_loop_command_t command,
                                                   const void* params,
                                                   void** inout_ptr) {
  IREE_ASSERT_ARGUMENT(self);

  if (command == IREE_LOOP_COMMAND_DRAIN) {
    // We don't really do anything with this; if called non-reentrantly then
    // there is no work to drain.
    return iree_ok_status();
  }

  iree_status_t* status_ptr = (iree_status_t*)self;

  // Initialize a new execution context on the stack.
  iree_loop_inline_ring_t stack_ring;
  iree_loop_inline_ring_initialize(status_ptr, &stack_ring);

  // Enqueue the initial command; we'll dequeue it right away but this keeps
  // the code size smaller.
  IREE_RETURN_IF_ERROR(iree_loop_inline_enqueue(&stack_ring, command, params));

  // If the status is not OK then we bail immediately; this allows for sticky
  // errors that mimic the abort behavior of an actual loop. Inline loops never
  // run work from multiple scopes as they don't persist beyond the loop
  // operation.
  if (iree_status_is_ok(*status_ptr)) {
    // Run until the ring is empty or we fail.
    return iree_loop_inline_run_all(&stack_ring);  // tail
  } else {
    // Abort all ops.
    iree_loop_inline_abort_all(&stack_ring);
    return iree_ok_status();
  }
}

IREE_API_EXPORT iree_status_t
iree_loop_inline_using_storage_ctl(void* self, iree_loop_command_t command,
                                   const void* params, void** inout_ptr) {
  if (command == IREE_LOOP_COMMAND_DRAIN) {
    // We don't really do anything with this; if called non-reentrantly then
    // there is no work to drain.
    return iree_ok_status();
  }

  iree_loop_inline_storage_t* storage = (iree_loop_inline_storage_t*)self;
  iree_loop_inline_ring_t* ring = (iree_loop_inline_ring_t*)storage->opaque;

  // Top-level call using external storage; run until the ring is empty or
  // we fail. Note that the storage contents are undefined and we have to
  // ensure the list is ready for use.
  iree_loop_inline_ring_initialize(&storage->status, ring);

  IREE_RETURN_IF_ERROR(iree_loop_inline_enqueue(ring, command, params));

  // If the status is not OK then we bail immediately; this allows for sticky
  // errors that mimic the abort behavior of an actual loop. Inline loops never
  // run work from multiple scopes as they don't persist beyond the loop
  // operation.
  if (iree_status_is_ok(storage->status)) {
    // Run until the ring is empty or we fail.
    return iree_loop_inline_run_all(ring);  // tail
  } else {
    // Abort all ops.
    iree_loop_inline_abort_all(ring);
    return iree_ok_status();
  }
}

static iree_status_t iree_loop_inline_reentrant_ctl(void* self,
                                                    iree_loop_command_t command,
                                                    const void* params,
                                                    void** inout_ptr) {
  if (command == IREE_LOOP_COMMAND_DRAIN) {
    // We don't really do anything with this; when called reentrantly we are
    // already draining as we drain on each top-level op.
    return iree_ok_status();
  }

  // Enqueue the new command and return to the caller - it'll be run by
  // the top-level control call.
  iree_loop_inline_ring_t* ring = (iree_loop_inline_ring_t*)self;
  return iree_loop_inline_enqueue(ring, command, params);  // tail
}
