// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for io_uring proactor implementation.
//
// This header exposes the proactor struct and internal helpers for use by
// io_uring-specific modules (socket.c, etc.). External users should include
// api.h instead.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_
#define IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_

#include "iree/async/platform/io_uring/api.h"
#include "iree/async/platform/io_uring/sparse_table.h"
#include "iree/async/platform/io_uring/uring.h"
#include "iree/async/platform/linux/signal.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/async/util/signal.h"
#include "iree/base/internal/atomic_slist.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_wait_operation_t
    iree_async_semaphore_wait_operation_t;

//===----------------------------------------------------------------------===//
// Event source tracking
//===----------------------------------------------------------------------===//

// Tracks a registered event source for persistent monitoring of an external fd.
// Uses multishot POLL_ADD to receive callbacks when the fd becomes readable.
// Doubly-linked list node; proactor owns the list.
struct iree_async_event_source_t {
  // Intrusive doubly-linked list for efficient removal.
  struct iree_async_event_source_t* next;
  struct iree_async_event_source_t* prev;

  // Owning proactor (for vtable access in callbacks).
  iree_async_proactor_t* proactor;

  // The monitored fd (not owned by the event source).
  int fd;

  // User callback invoked when the fd is readable.
  iree_async_event_source_callback_t callback;

  // Allocator used to allocate this struct (for deallocation).
  iree_allocator_t allocator;
};

//===----------------------------------------------------------------------===//
// Buffer pool tracking
//===----------------------------------------------------------------------===//

// Tracks a registered provided buffer ring (PBUF_RING) for multishot receives.
// Linked list node; proactor owns the list.
typedef struct iree_async_io_uring_buffer_pool_t {
  struct iree_async_io_uring_buffer_pool_t* next;

  // Buffer group ID assigned during registration.
  uint16_t group_id;

  // Number of buffer slots in the ring (power of 2).
  uint32_t ring_entries;

  // Size of each buffer in the pool.
  uint32_t buffer_size;

  // User-allocated ring memory. Layout:
  //   [0]: iree_io_uring_buf_ring_t header (with tail)
  //   [1..ring_entries]: iree_io_uring_buf_t entries
  void* ring_memory;

  // Size of ring_memory in bytes.
  size_t ring_memory_size;
} iree_async_io_uring_buffer_pool_t;

//===----------------------------------------------------------------------===//
// Proactor implementation struct
//===----------------------------------------------------------------------===//

// io_uring-specific proactor state.
typedef struct iree_async_proactor_io_uring_t {
  // Base proactor (must be first for safe casting).
  iree_async_proactor_t base;

  // io_uring ring buffer wrapper.
  iree_io_uring_ring_t ring;

  // Eventfd for cross-thread wake().
  int wake_eventfd;

  // Set when wake poll is armed to avoid duplicate submissions.
  bool wake_poll_armed;

  // When true, submit() fills SQEs in the ring but skips io_uring_enter.
  // Set during CQE processing to prevent synchronous CQE generation from
  // creating infinite re-submission loops (e.g., a multishot recv re-arm on
  // ENOBUFS immediately generates another ENOBUFS because the buffer ring is
  // still empty, which the CQE loop picks up and re-arms again, forever).
  // The poll function flushes deferred SQEs after the CQE loop completes.
  bool defer_submissions;

  // Cached capabilities from ring probing.
  iree_async_proactor_capabilities_t capabilities;

  // Registered buffer pools (PBUF_RING). Linked list, proactor-owned.
  iree_async_io_uring_buffer_pool_t* buffer_pools;

  // Next group ID to assign (monotonically increasing).
  uint16_t next_group_id;

  // Sparse buffer table for dynamic buffer registration (kernel 5.19+).
  // NULL on pre-5.19 kernels where the legacy singleton
  // IORING_REGISTER_BUFFERS path is used instead.
  iree_io_uring_sparse_table_t* buffer_table;

  // Legacy: number of buffers registered via IORING_REGISTER_BUFFERS on
  // pre-5.19 kernels (where buffer_table is NULL). Zero when buffer_table is
  // non-NULL (the sparse table tracks slot allocation instead).
  uint16_t legacy_registered_buffer_count;

  // Cross-proactor messaging state.
  // The pool is used by the fallback path (pre-5.18 kernels without MSG_RING).
  // MSG_RING delivers messages entirely through the kernel and does not use
  // the pool. The callback is invoked for both paths during poll().
  iree_async_message_pool_t message_pool;
  iree_async_proactor_message_callback_t message_callback;

  // Sequence emulator for IREE_ASYNC_OPERATION_TYPE_SEQUENCE operations.
  // Drives step-by-step execution when step_fn is set. When step_fn is NULL,
  // the LINK path is used instead (no emulator involvement).
  iree_async_sequence_emulator_t sequence_emulator;

  // MPSC queue of software operation completions (SEMAPHORE_SIGNAL, etc.)
  // ready for callback delivery. Submit threads push completed operations
  // here; poll() drains and fires callbacks. This allows software ops to
  // execute their side effects on the submit thread while deferring callback
  // delivery to the poll thread.
  //
  // Operations use base.next (offset 0) as the slist entry and base.linked_next
  // (repurposed after continuation chain is consumed) to carry the completion
  // status through the queue.
  iree_atomic_slist_t pending_software_completions;

  // MPSC queue of semaphore wait operations ready to complete.
  // Timepoint callbacks push trackers here, poll() drains and completes them.
  iree_atomic_slist_t pending_semaphore_waits;

  // Linked list of registered event sources. Proactor-owned.
  // Uses doubly-linked list for O(1) removal during unregister.
  iree_async_event_source_t* event_sources;

  // Linked list of registered relays. Proactor-owned.
  // Uses doubly-linked list for O(1) removal during unregister.
  struct iree_async_relay_t* relays;

  // Signal handling state (lazy-initialized on first subscribe).
  struct {
    // Whether signal handling has been initialized (signalfd poll registered).
    bool initialized;

    // Linux signalfd state (shared with potential future epoll backend).
    iree_async_linux_signal_state_t linux_state;

    // Dispatch state for deferred unsubscribe during callback iteration.
    iree_async_signal_dispatch_state_t dispatch_state;

    // Per-signal subscription lists. Each is a doubly-linked list.
    iree_async_signal_subscription_t* subscriptions[IREE_ASYNC_SIGNAL_COUNT];

    // Per-signal subscriber counts. Used to track when to add/remove signals
    // from the signalfd mask (only block signals that have subscribers).
    int subscriber_count[IREE_ASYNC_SIGNAL_COUNT];

    // Event source for monitoring the signalfd.
    iree_async_event_source_t* event_source;
  } signal;
} iree_async_proactor_io_uring_t;

// Upcast from base proactor to io_uring implementation.
static inline iree_async_proactor_io_uring_t* iree_async_proactor_io_uring_cast(
    iree_async_proactor_t* proactor) {
  return (iree_async_proactor_io_uring_t*)proactor;
}

//===----------------------------------------------------------------------===//
// Internal user data tagging
//===----------------------------------------------------------------------===//

// Internal operations use tagged user_data values to distinguish them from
// caller operations (which use the operation pointer directly).
//
// Encoding (LA57-safe, supports 5-level paging):
//   Bit 63:     INTERNAL_MARKER (1 = internal, 0 = user operation pointer)
//   Bits 56-62: Tag identifying the internal operation type (7 bits = 128 tags)
//   Bits 0-55:  Payload (56 bits - covers all LA57 userspace addresses)
//
// This encoding is safe for 5-level paging (LA57) because userspace addresses
// on x86-64 always have bit 56 = 0 (bit 56 distinguishes user/kernel space).
// The payload is stored unshifted, so pointers are preserved exactly.
//
// All internal tags are mutually exclusive and checked with equality.
#define IREE_IO_URING_INTERNAL_MARKER ((uint64_t)1 << 63)
#define IREE_IO_URING_INTERNAL_TAG_SHIFT 56
#define IREE_IO_URING_INTERNAL_TAG_MASK ((uint64_t)0x7F)
#define IREE_IO_URING_INTERNAL_PAYLOAD_MASK ((uint64_t)0x00FFFFFFFFFFFFFFULL)

// Internal operation tags (mutually exclusive, 7 bits = 0-127).
typedef enum iree_io_uring_internal_tag_e {
  IREE_IO_URING_TAG_WAKE = 1,             // Wake eventfd poll completion.
  IREE_IO_URING_TAG_CANCEL = 2,           // Async cancel completion.
  IREE_IO_URING_TAG_FENCE_IMPORT = 3,     // Fence import poll completion.
  IREE_IO_URING_TAG_MESSAGE_RECEIVE = 4,  // MSG_RING from another proactor.
  IREE_IO_URING_TAG_MESSAGE_SOURCE =
      5,  // MSG_RING source completion (no user callback).
  IREE_IO_URING_TAG_EVENT_SOURCE = 6,  // Event source multishot poll.
  IREE_IO_URING_TAG_RELAY = 7,         // Relay source completion.
  IREE_IO_URING_TAG_SIGNAL = 8,        // Signal fd multishot poll.
  // Linked POLL_ADD head for EVENT_WAIT and NOTIFICATION_WAIT (event mode).
  // The POLL_ADD CQE is always ignored; the linked READ CQE handles
  // resource release and user callback dispatch for both success and failure.
  IREE_IO_URING_TAG_LINKED_POLL = 9,
} iree_io_uring_internal_tag_t;

// Helpers for encoding/decoding internal user_data.
// Payload is stored in low 56 bits (unshifted), tag in bits 56-62.
#define iree_io_uring_internal_encode(tag, payload)        \
  (IREE_IO_URING_INTERNAL_MARKER |                         \
   ((uint64_t)(tag) << IREE_IO_URING_INTERNAL_TAG_SHIFT) | \
   ((uint64_t)(uintptr_t)(payload) & IREE_IO_URING_INTERNAL_PAYLOAD_MASK))
#define iree_io_uring_internal_tag(user_data)          \
  (((user_data) >> IREE_IO_URING_INTERNAL_TAG_SHIFT) & \
   IREE_IO_URING_INTERNAL_TAG_MASK)
#define iree_io_uring_internal_payload(user_data) \
  ((user_data) & IREE_IO_URING_INTERNAL_PAYLOAD_MASK)

#define iree_io_uring_is_internal_cqe(cqe) \
  iree_any_bit_set((cqe)->user_data, IREE_IO_URING_INTERNAL_MARKER)

//===----------------------------------------------------------------------===//
// Internal tracker types
//===----------------------------------------------------------------------===//

// Tracks a pending fence import (POLL_ADD on an external fd).
// Heap-allocated per import_fence call, freed when the CQE fires.
typedef struct iree_async_io_uring_fence_import_tracker_t {
  iree_async_semaphore_t* semaphore;  // Retained.
  // Value to signal semaphore to.
  uint64_t signal_value;
  // Owned fd; closed on completion.
  int fence_fd;
  iree_allocator_t allocator;  // For freeing this tracker.
} iree_async_io_uring_fence_import_tracker_t;

// Tracks a pending fence export (semaphore timepoint -> eventfd write).
// Heap-allocated per export_fence call, freed when the timepoint callback
// fires.
typedef struct iree_async_io_uring_fence_export_tracker_t {
  // Embedded; owned by semaphore until callback.
  iree_async_semaphore_timepoint_t timepoint;
  iree_async_semaphore_t* semaphore;  // Retained.
  // Written to on signal (caller-owned, not closed).
  int eventfd;
  iree_allocator_t allocator;  // For freeing this tracker.
} iree_async_io_uring_fence_export_tracker_t;

// Tracks a pending SEMAPHORE_WAIT operation.
// Heap-allocated per wait operation, freed when the operation completes.
// Contains embedded timepoints for each semaphore being waited on.
typedef struct iree_async_io_uring_semaphore_wait_tracker_t {
  // Intrusive MPSC list link for pending completion queue.
  iree_atomic_slist_entry_t slist_entry;

  // Back-pointer to the wait operation being tracked.
  iree_async_semaphore_wait_operation_t* operation;

  // Proactor to wake when a semaphore fires.
  iree_async_proactor_io_uring_t* proactor;

  // Allocator used for this tracker.
  iree_allocator_t allocator;

  // Number of semaphores being waited on.
  iree_host_size_t count;

  // For ALL mode: remaining semaphores to satisfy.
  // For ANY mode: first satisfied index (written by callback).
  iree_atomic_int32_t remaining_or_satisfied;

  // Completion status. Written by timepoint callback if failure occurs.
  iree_atomic_intptr_t completion_status;

  // Guard against double-enqueue: success callbacks (remaining_or_satisfied),
  // error callbacks, and cancel all independently decide to enqueue. Only the
  // first to CAS this from 0->1 actually pushes to the MPSC slist.
  iree_atomic_int32_t enqueued;

  // LINKED chain continuation head (for userspace chain emulation).
  // When the wait operation has LINKED flag, points to the first operation
  // in the continuation chain (linked via linked_next pointers).
  iree_async_operation_t* continuation_head;

  // Flexible array of timepoints (one per semaphore).
  iree_async_semaphore_timepoint_t timepoints[];
} iree_async_io_uring_semaphore_wait_tracker_t;

//===----------------------------------------------------------------------===//
// Internal APIs (shared across proactor implementation files)
//===----------------------------------------------------------------------===//

// Vtable for same-backend validation in submit.
extern const iree_async_proactor_vtable_t iree_async_proactor_io_uring_vtable;

// Capability probing (called from create).
// Probes the kernel for supported io_uring features and populates
// |out_capabilities| with the detected capabilities.
// Returns IREE_STATUS_UNAVAILABLE if the kernel is too old.
iree_status_t iree_async_proactor_io_uring_detect_capabilities(
    int ring_fd, uint32_t ring_features,
    iree_async_proactor_capabilities_t* out_capabilities);

// Buffer registration vtable implementations (in proactor_registration.c).
iree_status_t iree_async_proactor_io_uring_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry);
iree_status_t iree_async_proactor_io_uring_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry);
void iree_async_proactor_io_uring_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state);
iree_status_t iree_async_proactor_io_uring_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region);

// Continuation dispatch helpers (in proactor.c, used by proactor_submit.c).
void iree_async_proactor_io_uring_submit_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Pushes a completed software operation to the MPSC queue for callback
// delivery on the poll thread (in proactor.c, used by proactor_submit.c).
void iree_async_proactor_io_uring_push_software_completion(
    iree_async_proactor_io_uring_t* proactor, iree_async_operation_t* operation,
    iree_status_t status);

// Iteratively dispatches a continuation chain containing software operations.
// Software ops execute inline (side effects only) with completions pushed to
// MPSC for poll-thread callback delivery. Kernel ops are submitted to the ring
// for CQE-driven completion. (In proactor_submit.c, used by proactor.c.)
void iree_async_proactor_io_uring_dispatch_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Cancels a continuation chain by pushing CANCELLED completions to the MPSC
// queue. Resources are retained before pushing so the drain's release is
// balanced. (In proactor_submit.c, used by proactor.c.)
void iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head);

// Submit vtable implementation (in proactor_submit.c).
iree_status_t iree_async_proactor_io_uring_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_PROACTOR_H_
