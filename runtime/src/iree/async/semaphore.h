// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-layer synchronization primitive with timeline semantics.
//
// An async semaphore tracks a monotonically increasing uint64 timeline value.
// Operations wait for the semaphore to reach a target value, and signal it
// forward when work completes.
//
// ## Layer integration
//
// HAL drivers implement the vtable to bridge GPU timelines to the async world.
// The net layer waits on and signals these semaphores without knowing the
// underlying GPU primitive. The async layer provides a software-only
// implementation for testing and for semaphores with no native backing.
// All semaphores share the same concrete state (timeline, frontier,
// timepoints, failure tracking); hardware-backed semaphores add device-specific
// primitives as additive acceleration.
//
// HAL semaphores EMBED async semaphores as their first member, enabling
// toll-free downcast from iree_hal_semaphore_t* to iree_async_semaphore_t*.
// This allows the net layer to work with GPU semaphores without HAL knowledge.
//
// ## Causal tracking
//
// Each semaphore accumulates a frontier from signals. The frontier is the
// merge of all frontiers passed to signal() calls. This enables:
//   - Transitive causality: if A signals S1, and B waits on S1 then signals S2,
//     S2's frontier includes A's contribution.
//   - Remote ordering: frontiers propagate across machines via the net layer.
//   - Wait elision: if frontier dominance proves causality, waits can be
//   skipped.
//
// ## Tainting
//
// Values may be "tainted" — signaled from external sources (imports, IPC)
// where the HAL/async layer didn't witness the underlying work. Tainted values
// are tracked via a watermark: last_untainted_value. Values above the watermark
// should not be trusted for aggressive optimizations (e.g., buffer reuse).
//
// ## Failure semantics
//
// Failure is sticky: once a semaphore is failed, all current and future waiters
// receive the failure status. This propagates errors (e.g. GPU fault, network
// disconnect) to all dependents without polling.
//
// ## Timepoint callbacks
//
// Timepoint callbacks fire WITHOUT the semaphore's internal lock held.
// Callbacks may safely signal or fail any semaphore, enabling zero-hop
// semaphore chaining (e.g., "when A reaches X, signal B to Y"). This
// includes the triggering semaphore itself, though note that signaling
// the triggering semaphore from its own callback creates recursion on the
// call stack (signal → dispatch → callback → signal → ...), so chains
// must be finite.
//
// cancel_timepoint returns true if the timepoint was removed from the
// pending list before dispatch, guaranteeing the callback will not fire.
// If it returns false, the callback has been or is being invoked. Callers
// with stack-allocated state must wait for callback completion using their
// own synchronization before destroying the state (see multi_wait for an
// example using a per-timepoint completion flag).
//
// Callbacks SHOULD be fast to keep the signal path responsive. Long-running
// work should be deferred to a worker thread (via the task executor) or
// the proactor.
//
// Callbacks MUST NOT:
//   - Perform blocking I/O or blocking waits on semaphores.
//   - Acquire locks that might be held by threads signaling this semaphore.
//
// Callbacks MAY:
//   - Signal or fail any semaphore (the internal lock is not held).
//   - Perform atomic operations (fetch_add, CAS, store).
//   - Post notifications (iree_notification_post).
//   - Push to MPSC lock-free lists.
//   - Write to eventfd/pipe (non-blocking kernel calls).
//   - Submit tasks to executors (iree_task_executor_submit/flush).

#ifndef IREE_ASYNC_SEMAPHORE_H_
#define IREE_ASYNC_SEMAPHORE_H_

#include "iree/async/frontier.h"
#include "iree/async/operation.h"
#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_semaphore_vtable_t iree_async_semaphore_vtable_t;
typedef struct iree_async_semaphore_timepoint_t
    iree_async_semaphore_timepoint_t;

typedef struct iree_async_semaphore_t iree_async_semaphore_t;

//===----------------------------------------------------------------------===//
// Timepoint
//===----------------------------------------------------------------------===//

// Callback invoked when a timepoint is triggered.
// Fires when the semaphore reaches the target value, is failed, or the
// timepoint is cancelled.
//
// |status| is OK if the value was reached, or the failure/cancellation status.
//   Ownership of the status is transferred to the callback.
// |timepoint| is the triggered timepoint. The callback owns the storage after
//   this call and may reuse or release it.
//
// This callback is invoked WITHOUT the semaphore's internal lock held.
// See the file header for constraints on what callbacks may do.
typedef void (*iree_async_semaphore_timepoint_fn_t)(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

// A pending wait on a semaphore value. Caller-owned storage (intrusive — no
// allocation by the semaphore). The semaphore links active timepoints into an
// internal list; the caller must not move or free the timepoint while active.
//
// Lifecycle:
//   1. Caller allocates timepoint storage (stack, heap, or embedded in struct)
//   2. Caller sets callback and user_data
//   3. Caller calls acquire_timepoint() — semaphore takes ownership of storage
//   4. Callback fires (semaphore releases ownership) OR caller calls
//      cancel_timepoint() (semaphore releases ownership)
//   5. Caller may reuse or free the storage
//
// After the callback fires (or after cancel_timepoint returns true,
// guaranteeing the callback will not fire), the caller owns the storage.
typedef struct iree_async_semaphore_timepoint_t {
  // Intrusive doubly-linked list (semaphore-internal, for O(1) removal).
  struct iree_async_semaphore_timepoint_t* next;
  struct iree_async_semaphore_timepoint_t* prev;

  // The semaphore this timepoint is registered with.
  // The timepoint does NOT hold a reference; the caller must ensure the
  // semaphore outlives active timepoints (or cancel before releasing).
  iree_async_semaphore_t* semaphore;

  // The timeline value that triggers this timepoint.
  uint64_t minimum_value;

  // Callback and context.
  iree_async_semaphore_timepoint_fn_t callback;
  void* user_data;
} iree_async_semaphore_timepoint_t;

//===----------------------------------------------------------------------===//
// Link (zero-allocation semaphore relay)
//===----------------------------------------------------------------------===//

// A relay from one semaphore to another. When the source semaphore reaches
// |source_value|, the target semaphore is signaled to |target_value|.
//
// This is the common case for heterogeneous device relays (GPU completion →
// CPU task start, DMA chain stages) and pipeline plumbing where no user code
// needs to run. Uses the timepoint mechanism internally — zero allocation,
// processed during the normal dispatch path.
//
// On source failure, the failure status is propagated to the target
// automatically. On source cancellation (deinitialize), the CANCELLED status
// is propagated as a failure to the target.
//
// Caller-owned storage with the same lifecycle as timepoints:
//   1. Caller allocates link storage (stack, heap, or embedded in struct)
//   2. Caller calls iree_async_semaphore_link() — source takes ownership
//   3. Link fires (ownership returns to caller) OR caller calls
//      iree_async_semaphore_unlink() (ownership returns to caller)
//   4. Caller may reuse or free the storage
//
// The caller must ensure both source and target semaphores outlive the link.
// The link does NOT hold references to either semaphore.
typedef struct iree_async_semaphore_link_t {
  // Embedded timepoint (at offset 0 for direct cast from timepoint callback).
  iree_async_semaphore_timepoint_t timepoint;

  // Target semaphore to signal when the source reaches its value.
  iree_async_semaphore_t* target;

  // Value to signal on the target semaphore.
  uint64_t signal_value;
} iree_async_semaphore_link_t;

//===----------------------------------------------------------------------===//
// Semaphore
//===----------------------------------------------------------------------===//

// Vtable for semaphore implementations.
// HAL backends provide these to bridge GPU timeline semaphores. A software-only
// implementation is available for testing and for pure-CPU coordination.
//
// HAL backends typically:
//   1. Embed iree_async_semaphore_t as the first member of their struct
//   2. Store the native GPU primitive (CUevent, hsa_signal_t, VkSemaphore)
//   3. Implement vtable methods by calling iree_async_semaphore_* helpers
//      for common operations, plus native GPU code for hardware signaling
typedef struct iree_async_semaphore_vtable_t {
  // Destroy. At vtable offset 0 for toll-free bridging.
  void (*destroy)(iree_async_semaphore_t* semaphore);

  // Query current timeline value (non-blocking, acquire semantics).
  // Returns the latest signaled value. Thread-safe.
  uint64_t (*query)(iree_async_semaphore_t* semaphore);

  // Signal the semaphore to |value| with causal context |frontier|.
  // Thread-safe.
  //
  // |value| must be strictly greater than the current timeline value.
  // Signaling a value <= current returns IREE_STATUS_INVALID_ARGUMENT.
  //
  // |frontier| is the causal context of this signal: it encodes what has
  // happened before this point. The remoting layer uses this to propagate
  // ordering guarantees across machines — remote receivers know they can
  // trust that everything in the frontier has completed when they observe
  // this signal. The semaphore merges this frontier with its accumulated
  // frontier (component-wise max).
  //
  // May be NULL for local-only signals where causal tracking is not needed
  // (testing, pure-CPU coordination). Implementations that don't participate
  // in remoting may ignore the frontier parameter.
  //
  // Returns IREE_STATUS_RESOURCE_EXHAUSTED if the frontier merge would exceed
  // the semaphore's internal frontier capacity.
  iree_status_t (*signal)(iree_async_semaphore_t* semaphore, uint64_t value,
                          const iree_async_frontier_t* frontier);

  // Read the semaphore's current accumulated frontier into |out_frontier|.
  // The frontier is the merge of all frontiers passed to prior signal() calls.
  // Copies up to |capacity| entries into the caller-provided storage.
  // Returns the actual entry count in the internal frontier (may exceed
  // |capacity|, indicating truncation — caller can retry with larger storage).
  //
  // Thread-safe. Uses acquire semantics. For GPU-backed semaphores, this reads
  // software-side state only (not the hardware timeline value).
  //
  // Returns 0 if no frontier has been attached (all prior signals passed NULL).
  uint8_t (*query_frontier)(iree_async_semaphore_t* semaphore,
                            iree_async_frontier_t* out_frontier,
                            uint8_t capacity);

  // Permanently fail the semaphore.
  // All current timepoints fire with |status|. All future acquire_timepoint
  // calls return the failure immediately. Takes ownership of |status|.
  // Thread-safe. First failure wins (subsequent fails are ignored).
  void (*fail)(iree_async_semaphore_t* semaphore, iree_status_t status);

  // Export the semaphore as a pollable platform primitive.
  // The primitive will become signaled when |minimum_value| is reached.
  // Not all backends support this; returns IREE_STATUS_UNAVAILABLE if not.
  // The caller is responsible for closing the returned primitive.
  iree_status_t (*export_primitive)(iree_async_semaphore_t* semaphore,
                                    uint64_t minimum_value,
                                    iree_async_primitive_t* out_primitive);
} iree_async_semaphore_vtable_t;

//===----------------------------------------------------------------------===//
// Semaphore type
//===----------------------------------------------------------------------===//

// Default inline frontier capacity for semaphores.
// 16 entries covers typical multi-GPU workloads (4-8 devices x 2-4 queues).
// Exceeding this capacity during signal() returns RESOURCE_EXHAUSTED.
#define IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY 16

// Timeline semaphore with atomic state, frontier accumulation, and timepoint
// dispatch. HAL backends embed this as their first member for toll-free
// bridging (iree_hal_semaphore_t* -> iree_async_semaphore_t*).
//
// Provides:
//   - Atomic timeline value (monotonically increasing uint64)
//   - Frontier accumulation (causal vector clock, merge on signal)
//   - Timepoint dispatch (callbacks when values are reached)
//   - Failure tracking (sticky first-failure-wins)
//   - Taint watermark (external source tracking for buffer reuse safety)
//
// Hardware backends (AMDGPU, CUDA, Vulkan) embed this at offset 0 and add
// device-specific state (HSA signals, CUevents, VkSemaphores) after it.
// The hardware primitives are additive acceleration.
//
// ## Embedding
//
// Embedders allocate a single block containing their struct (with the
// semaphore at offset 0) followed by trailing frontier storage:
//
//   [embedder_struct_t]                       offset 0
//     [iree_async_semaphore_t]                offset 0 (toll-free bridge)
//     [device-specific fields]
//   [iree_async_frontier_t + entries[N]]      at frontier_offset
//
// Use iree_async_semaphore_layout() to compute the total allocation
// size and frontier offset. Pass the offset to _initialize().
typedef struct iree_async_semaphore_t {
  iree_atomic_ref_count_t ref_count;
  const iree_async_semaphore_vtable_t* vtable;

  // Allocator used for standalone semaphores (via iree_async_semaphore_create).
  // Embedders leave this zeroed — they manage their own allocation.
  iree_allocator_t allocator;

  // Proactor for async I/O operations (import_fence, export_fence).
  // HAL semaphores receive this from their device, which selects a proactor
  // from the pool based on NUMA affinity. Not retained — the proactor's
  // lifetime is managed by the proactor pool retained by the device.
  iree_async_proactor_t* proactor;

  // Current timeline value. Monotonically increasing.
  // Uses int64_t for safe CAS operations (negative values not used).
  iree_atomic_int64_t timeline_value;

  // Untainted watermark. Values <= this are known to have been signaled by
  // witnessed work. Values > this may be tainted (from imports/IPC).
  iree_atomic_int64_t last_untainted_value;

  // Protects timepoints_head and frontier during mutation.
  iree_slim_mutex_t mutex;

  // Sticky failure status (atomic for lock-free read paths).
  // Non-zero indicates failure. Once set (via CAS), all waiters receive this
  // status. The pointer is published with release semantics; readers use
  // acquire semantics, which guarantees visibility of the status payload.
  // Owned by the semaphore (freed on deinitialize).
  iree_atomic_intptr_t failure_status;

  // Doubly-linked list of pending timepoints.
  iree_async_semaphore_timepoint_t* timepoints_head;

  // Pointer to the frontier in trailing storage. Set during initialization
  // from the frontier_offset parameter. The frontier header and entries[]
  // live in the same allocation block, after the embedder's struct.
  iree_async_frontier_t* frontier;

  // Maximum number of frontier entries the trailing storage can hold.
  uint8_t frontier_capacity;
} iree_async_semaphore_t;

// Computes the allocation size and frontier offset for a semaphore
// (or any struct that embeds one) with trailing frontier storage.
//
// |base_size| is sizeof(embedder_struct_t) — or sizeof(iree_async_semaphore_t)
// for standalone allocation. The frontier is placed after the struct with
// proper alignment. |out_frontier_offset| receives the byte offset from the
// start of the allocation to the frontier header.
//
// Usage:
//   iree_host_size_t frontier_offset = 0, total_size = 0;
//   IREE_RETURN_IF_ERROR(iree_async_semaphore_layout(
//       sizeof(my_embedder_t), capacity, &frontier_offset, &total_size));
//   my_embedder_t* sem = NULL;
//   IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc, total_size, &sem));
//   memset(sem, 0, total_size);
//   iree_async_semaphore_initialize(
//       &my_vtable, proactor, initial_value, frontier_offset, capacity,
//       &sem->async);
static inline iree_status_t iree_async_semaphore_layout(
    iree_host_size_t base_size, uint8_t frontier_capacity,
    iree_host_size_t* out_frontier_offset, iree_host_size_t* out_total_size) {
  return IREE_STRUCT_LAYOUT(
      base_size, out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                iree_alignof(iree_async_frontier_t),
                                out_frontier_offset),
      IREE_STRUCT_FIELD_FAM(frontier_capacity, iree_async_frontier_entry_t));
}

// Initializes a semaphore in-place within caller-provided memory.
// The caller must have allocated at least the size returned by
// iree_async_semaphore_layout() and zeroed the memory.
//
// |vtable| is the embedder's vtable (which may override signal/query/etc.).
// |proactor| is the proactor for async I/O (import_fence, export_fence).
// |frontier_offset| is the byte offset from |out_semaphore| to the frontier
// storage (as returned by iree_async_semaphore_layout).
//
// Does NOT set the allocator field — that's only used by
// iree_async_semaphore_create. Embedders manage their own allocation lifecycle.
IREE_API_EXPORT void iree_async_semaphore_initialize(
    const iree_async_semaphore_vtable_t* vtable,
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_host_size_t frontier_offset, uint8_t frontier_capacity,
    iree_async_semaphore_t* out_semaphore);

// Increments the reference count.
IREE_API_EXPORT void iree_async_semaphore_retain(
    iree_async_semaphore_t* semaphore);

// Decrements the reference count and destroys if it reaches zero.
IREE_API_EXPORT void iree_async_semaphore_release(
    iree_async_semaphore_t* semaphore);

// Deinitializes a semaphore, releasing internal resources.
// Dispatches all pending timepoints with CANCELLED status.
// Frees failure_status if set. Deinitializes the mutex.
// Does NOT free the semaphore memory — the embedder manages that.
IREE_API_EXPORT void iree_async_semaphore_deinitialize(
    iree_async_semaphore_t* semaphore);

// Creates a standalone semaphore (no embedding).
// Allocates the semaphore + frontier in a single block.
// |proactor| is the proactor for async I/O (import_fence, export_fence).
// |frontier_capacity| is the maximum number of frontier entries the semaphore
// can track. Use IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY for typical
// workloads.
IREE_API_EXPORT iree_status_t iree_async_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    uint8_t frontier_capacity, iree_allocator_t allocator,
    iree_async_semaphore_t** out_semaphore);

//===----------------------------------------------------------------------===//
// Inline vtable dispatch (hot path)
//===----------------------------------------------------------------------===//

// Returns the current timeline value of the semaphore. Thread-safe with
// acquire semantics. The returned value reflects the latest signal that has
// been observed; a concurrent signal may have advanced it further.
static inline uint64_t iree_async_semaphore_query(
    iree_async_semaphore_t* semaphore) {
  return semaphore->vtable->query(semaphore);
}

// Signals the semaphore to |value| with causal context |frontier|. The value
// must be strictly greater than the current timeline value; signaling a value
// at or below the current value returns IREE_STATUS_INVALID_ARGUMENT.
//
// The |frontier| encodes the causal context of this signal for cross-machine
// ordering propagation. The semaphore merges it with its accumulated frontier
// (component-wise max). Pass NULL for local-only signals where causal tracking
// is not needed.
//
// Thread-safe. Returns IREE_STATUS_RESOURCE_EXHAUSTED if the frontier merge
// would exceed the semaphore's internal frontier capacity.
static inline iree_status_t iree_async_semaphore_signal(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier) {
  return semaphore->vtable->signal(semaphore, value, frontier);
}

// Reads the semaphore's accumulated frontier into |out_frontier|, copying up
// to |capacity| entries. Returns the actual entry count in the internal
// frontier, which may exceed |capacity| (indicating truncation -- caller can
// retry with larger storage). Returns 0 if no frontier has been attached.
// Thread-safe with acquire semantics.
static inline uint8_t iree_async_semaphore_query_frontier(
    iree_async_semaphore_t* semaphore, iree_async_frontier_t* out_frontier,
    uint8_t capacity) {
  return semaphore->vtable->query_frontier(semaphore, out_frontier, capacity);
}

// Permanently fails the semaphore. All current timepoints fire with |status|
// and all future acquire_timepoint calls return the failure immediately. Takes
// ownership of |status|. Thread-safe; the first failure wins and subsequent
// fails are ignored.
static inline void iree_async_semaphore_fail(iree_async_semaphore_t* semaphore,
                                             iree_status_t status) {
  semaphore->vtable->fail(semaphore, status);
}

// Registers a timepoint that fires when the semaphore reaches |minimum_value|.
// The |timepoint| is caller-owned storage that must remain valid and at a
// stable address until the callback fires or cancel_timepoint completes.
// The caller must set timepoint->callback and timepoint->user_data before
// calling. The callback may fire before this function returns if the value is
// already reached or the semaphore is already failed.
//
// Checks for immediate satisfaction (value already reached) or immediate
// failure (semaphore already failed) and fires the callback synchronously in
// those cases. Otherwise inserts the timepoint into the pending list.
IREE_API_EXPORT iree_status_t iree_async_semaphore_acquire_timepoint(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_semaphore_timepoint_t* timepoint);

// Cancels a pending timepoint. Returns true if the timepoint was found in the
// pending list and removed — the callback will not fire. Returns false if the
// timepoint was not found, meaning the callback has already been dispatched
// (it may still be executing or may have completed). Callers that need to wait
// for an in-flight callback to finish must use their own synchronization.
IREE_API_EXPORT bool iree_async_semaphore_cancel_timepoint(
    iree_async_semaphore_t* semaphore,
    iree_async_semaphore_timepoint_t* timepoint);

// Exports the semaphore as a pollable platform primitive that becomes signaled
// when |minimum_value| is reached. The caller is responsible for closing the
// returned primitive. Returns IREE_STATUS_UNAVAILABLE if the backend does not
// support primitive export.
static inline iree_status_t iree_async_semaphore_export_primitive(
    iree_async_semaphore_t* semaphore, uint64_t minimum_value,
    iree_async_primitive_t* out_primitive) {
  return semaphore->vtable->export_primitive(semaphore, minimum_value,
                                             out_primitive);
}

//===----------------------------------------------------------------------===//
// Semaphore linking (zero-allocation relay)
//===----------------------------------------------------------------------===//

// Links two semaphores: when |source| reaches |source_value|, |target| is
// signaled to |target_value|. Uses the timepoint mechanism internally — zero
// allocation, processed during the normal dispatch path.
//
// |out_link| is caller-owned storage that must remain valid and at a stable
// address until the link fires or is cancelled via iree_async_semaphore_unlink.
// The caller must ensure both |source| and |target| outlive the link.
//
// On source failure, the failure is propagated to |target|. On source
// cancellation (deinitialize), CANCELLED is propagated as a failure.
//
// The link may fire synchronously if |source| has already reached
// |source_value| or is already failed. Returns OK in all cases (the link
// itself never fails — signal errors on the target are ignored because they
// indicate the target is already at or past |target_value|).
//
// Links compose naturally: A→B→C chains work via recursive dispatch (signal A
// → dispatch A's link → signal B → dispatch B's link → signal C). Each link
// fires without holding the source semaphore's lock, so there is no deadlock.
// Chains must be finite (no cycles).
IREE_API_EXPORT iree_status_t
iree_async_semaphore_link(iree_async_semaphore_t* source, uint64_t source_value,
                          iree_async_semaphore_t* target, uint64_t target_value,
                          iree_async_semaphore_link_t* out_link);

// Cancels a pending link. Returns true if the link was removed before it
// fired — the target will NOT be signaled. Returns false if the link has
// already fired (or is currently firing). Same semantics as cancel_timepoint.
IREE_API_EXPORT bool iree_async_semaphore_unlink(
    iree_async_semaphore_link_t* link);

//===----------------------------------------------------------------------===//
// Tainting (external source tracking)
//===----------------------------------------------------------------------===//

// Returns true if |value| may have come from an external source (import, IPC).
// Tainted values should not be trusted for aggressive optimizations like buffer
// reuse. Returns false for values <= the last untainted signal value.
//
// Implementation note: This queries the semaphore's internal watermark. For
// HAL semaphores, the watermark is advanced when HAL-internal signals occur
// and held back when external imports are signaled.
IREE_API_EXPORT bool iree_async_semaphore_is_value_tainted(
    iree_async_semaphore_t* semaphore, uint64_t value);

// Marks values above |threshold| as tainted.
// Called by HAL import operations when signaling values from external sources.
// Thread-safe. The watermark only decreases (marking more values as tainted);
// if the watermark is already below |threshold|, this is a no-op.
IREE_API_EXPORT void iree_async_semaphore_mark_tainted_above(
    iree_async_semaphore_t* semaphore, uint64_t threshold);

// Signals the semaphore and advances the untainted watermark.
// Equivalent to signal() followed by marking the value as untainted.
// Called by HAL signal implementations for internally-witnessed GPU work.
// This is the "normal" signal path for GPU work completing.
IREE_API_EXPORT iree_status_t iree_async_semaphore_signal_untainted(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

// Returns the current untainted watermark value.
// Values <= this are guaranteed to have been signaled by witnessed work.
// Values > this may be tainted.
IREE_API_EXPORT uint64_t
iree_async_semaphore_query_untainted_value(iree_async_semaphore_t* semaphore);

//===----------------------------------------------------------------------===//
// HAL composition helpers
//===----------------------------------------------------------------------===//

// The following functions provide common semaphore operations that HAL backends
// can use when implementing their vtables. This avoids duplicating timeline
// tracking, frontier merging, and timepoint management across backends.
//
// Typical HAL signal implementation:
//
//   static iree_status_t my_semaphore_signal(
//       iree_async_semaphore_t* base, uint64_t value,
//       const iree_async_frontier_t* frontier) {
//     my_semaphore_t* sem = (my_semaphore_t*)base;
//
//     // Advance timeline value and merge frontier.
//     IREE_RETURN_IF_ERROR(
//         iree_async_semaphore_advance_timeline(base, value, frontier));
//
//     // Signal native GPU primitive.
//     cudaEventRecord(sem->event, sem->stream);
//
//     // Dispatch satisfied timepoints.
//     iree_async_semaphore_dispatch_timepoints(base, value);
//
//     return iree_ok_status();
//   }

// Legacy aliases. Callers in HAL backends that call insert/remove directly
// should migrate to iree_async_semaphore_acquire_timepoint/cancel_timepoint.
#define iree_async_semaphore_insert_timepoint \
  iree_async_semaphore_acquire_timepoint
#define iree_async_semaphore_remove_timepoint \
  iree_async_semaphore_cancel_timepoint

// Advances the timeline value (CAS) and merges the frontier.
// Does NOT dispatch timepoints (caller does that after native signaling).
// Returns INVALID_ARGUMENT if the timeline is already at or past |value| —
// each timeline value must be signaled exactly once and concurrent races are
// not valid (they indicate duplicate signals or non-monotonic scheduling).
// Returns frontier merge errors if a frontier is provided and the merge fails.
IREE_API_EXPORT iree_status_t iree_async_semaphore_advance_timeline(
    iree_async_semaphore_t* semaphore, uint64_t value,
    const iree_async_frontier_t* frontier);

// Dispatches all timepoints with minimum_value <= |value|.
// Collects satisfied timepoints under the internal lock, then fires their
// callbacks after releasing it. Callbacks receive OK status.
IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints(
    iree_async_semaphore_t* semaphore, uint64_t value);

// Dispatches all timepoints with the given failure status.
// Takes ownership of |status| (clones for each timepoint, frees original).
// Called by fail() implementations.
IREE_API_EXPORT void iree_async_semaphore_dispatch_timepoints_failed(
    iree_async_semaphore_t* semaphore, iree_status_t status);

//===----------------------------------------------------------------------===//
// Multi-wait
//===----------------------------------------------------------------------===//

// Blocks the calling thread until the wait condition is satisfied or the
// timeout expires.
//
// For ALL mode: returns OK when every semaphore reaches its minimum_value.
// For ANY mode: returns OK when at least one semaphore reaches its
// minimum_value.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the timeout expires before the
// condition is met.
// Returns IREE_STATUS_ABORTED if any semaphore fails (regardless of mode).
//
// Handles all timepoint management internally: no event pools, wait sets, or
// per-semaphore ceremony needed. For small counts (<=8 semaphores), uses
// stack allocation for timepoint storage.
//
// Thread-safe. May be called from any thread with any mix of semaphore types
// (software, HAL-backed, GPU-backed). HAL semaphores are compatible via
// toll-free bridging (cast iree_hal_semaphore_t* to iree_async_semaphore_t*).
IREE_API_EXPORT iree_status_t iree_async_semaphore_multi_wait(
    iree_async_wait_mode_t wait_mode, iree_async_semaphore_t** semaphores,
    const uint64_t* minimum_values, iree_host_size_t count,
    iree_timeout_t timeout, iree_allocator_t allocator);

//===----------------------------------------------------------------------===//
// Device fence bridging
//===----------------------------------------------------------------------===//

// Imports an external device fence and bridges it to a proactor-managed
// semaphore. When the fence signals (i.e., the GPU or device reaches the
// associated execution point), the semaphore is advanced to |signal_value|.
//
// This enables GPU->proactor synchronization: a GPU command buffer exports a
// sync_file when done, the proactor imports it, and downstream operations
// (sends, writes) can wait on the semaphore to fire automatically.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   poll    | poll     | yes  | poll
//
// Platform-specific handle types:
//   - Linux: FD (sync_file fd from GPU driver, e.g., amdgpu, i915)
//   - macOS: MACH_PORT or FD depending on GPU API (Metal, MoltenVK)
//   - Windows: WIN32_HANDLE (D3D12 fence HANDLE)
//
// Handle ownership:
//   The proactor takes ownership of the handle in |fence| and will close it
//   after the fence signals or on cancellation. The caller must not close the
//   handle after a successful import.
//
// Semaphore lifetime:
//   The semaphore must outlive the imported fence. If the semaphore is
//   destroyed while waiting on a fence, behavior is undefined.
//
// Implementation:
//   io_uring: IORING_OP_POLL_ADD on the sync_file fd.
//   kqueue: EVFILT_READ on the fd.
//   IOCP: Thread pool wait on the HANDLE (native, no polling).
//   generic: select/poll on the fd.
//
// Threading:
//   Thread-safe. May be called from any thread, including GPU driver
//   completion callbacks and worker threads. The proactor defers internal
//   registration to its poll thread; the caller does not need to synchronize
//   with poll().
//
// Returns:
//   IREE_STATUS_OK: Fence imported; semaphore will signal when ready.
//   IREE_STATUS_UNAVAILABLE: Platform doesn't support fence import.
//   IREE_STATUS_INVALID_ARGUMENT: Invalid fence handle.
IREE_API_EXPORT iree_status_t iree_async_semaphore_import_fence(
    iree_async_proactor_t* proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value);

// Exports a semaphore wait point as a device fence.
//
// The returned handle signals when the semaphore reaches |wait_value|. This
// enables proactor->GPU synchronization: a network receive completes and
// signals a semaphore, the exported fence signals, and the GPU can begin
// processing the received data without host-side round-trips.
//
// Availability:
//   generic | io_uring | IOCP | kqueue
//   poll    | poll     | yes  | poll
//
// Platform-specific handle types:
//   - Linux: FD (eventfd that can be imported as sync_file by GPU drivers)
//   - macOS: MACH_PORT or FD
//   - Windows: WIN32_HANDLE (event HANDLE for D3D12 fence wait)
//
// Handle ownership:
//   The caller owns the returned handle and must close it when done. The handle
//   remains valid even after the semaphore reaches the target value - it simply
//   becomes signaled and stays signaled.
//
// Implementation:
//   The proactor creates an eventfd (or platform equivalent), watches the
//   semaphore internally, and signals the eventfd when the value is reached.
//   On IOCP, uses native Windows event objects that D3D12 can wait on directly.
//
// Threading:
//   Thread-safe. May be called from any thread. The implementation only
//   touches the semaphore's internal lock and creates new file descriptors;
//   no proactor poll-thread state is accessed.
//
// Returns:
//   IREE_STATUS_OK: Fence exported successfully.
//   IREE_STATUS_UNAVAILABLE: Platform doesn't support fence export.
//   IREE_STATUS_RESOURCE_EXHAUSTED: Too many exported fences.
IREE_API_EXPORT iree_status_t iree_async_semaphore_export_fence(
    iree_async_proactor_t* proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_SEMAPHORE_H_
