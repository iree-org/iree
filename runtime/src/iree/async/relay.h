// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Relay primitive for declarative source-to-sink event dataflow.
//
// A relay connects an event source (fd becoming ready, notification signaled)
// to an event sink (signal another fd, signal a notification) with optional
// kernel-optimized paths on io_uring using LINK chains.
//
// Relays are **event-based**: "when X happens, trigger Y". This is distinct
// from **timeline-based** semaphore operations (import_fence/export_fence)
// which deal with monotonic values.
//
// Use cases:
//   - Bridge external device fds to notifications for thread wakeup
//   - Fan-out: multiple relays from one source to different sinks
//   - Device-to-device signaling without userspace round-trips (io_uring LINK)
//
// Ownership model:
//   - The relay does NOT own the source/sink resources by default
//   - Use IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE to transfer fd ownership
//   - Notifications are retained during relay lifetime
//
// Thread safety:
//   - Registration/unregistration must be serialized with poll()
//   - Sink firing happens from within poll() on the polling thread

#ifndef IREE_ASYNC_RELAY_H_
#define IREE_ASYNC_RELAY_H_

#include "iree/async/notification.h"
#include "iree/async/primitive.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_relay_t iree_async_relay_t;

//===----------------------------------------------------------------------===//
// Relay source types
//===----------------------------------------------------------------------===//

// Event sources that can trigger a relay.
typedef enum iree_async_relay_source_type_e {
  // File descriptor or HANDLE becomes ready (poll indicates readability).
  // Uses POLL_ADD on the primitive's fd.
  IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE = 0,

  // Notification is signaled (epoch advances).
  // Uses FUTEX_WAIT (6.7+) or eventfd poll depending on notification mode.
  IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION = 1,
} iree_async_relay_source_type_t;

// Source specification for a relay.
typedef struct iree_async_relay_source_t {
  iree_async_relay_source_type_t type;
  union {
    // For PRIMITIVE type: the fd/HANDLE to monitor.
    iree_async_primitive_t primitive;

    // For NOTIFICATION type: the notification to wait on.
    // Retained during relay lifetime.
    iree_async_notification_t* notification;
  };
} iree_async_relay_source_t;

// Creates a relay source from a primitive (fd/HANDLE).
static inline iree_async_relay_source_t iree_async_relay_source_from_primitive(
    iree_async_primitive_t primitive) {
  iree_async_relay_source_t source;
  source.type = IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE;
  source.primitive = primitive;
  return source;
}

// Creates a relay source from a notification.
static inline iree_async_relay_source_t
iree_async_relay_source_from_notification(
    iree_async_notification_t* notification) {
  iree_async_relay_source_t source;
  source.type = IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION;
  source.notification = notification;
  return source;
}

//===----------------------------------------------------------------------===//
// Relay sink types
//===----------------------------------------------------------------------===//

// Event sinks that a relay can trigger.
typedef enum iree_async_relay_sink_type_e {
  // Write a value to an eventfd/event handle to signal it.
  // This makes the primitive readable for waiters.
  IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE = 0,

  // Signal a notification, waking up to wake_count waiters.
  // Uses FUTEX_WAKE (6.7+) or eventfd write depending on notification mode.
  IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION = 1,
} iree_async_relay_sink_type_t;

// Sink specification for a relay.
typedef struct iree_async_relay_sink_t {
  iree_async_relay_sink_type_t type;
  union {
    // For SIGNAL_PRIMITIVE type.
    struct {
      // The primitive (eventfd/event) to signal.
      iree_async_primitive_t primitive;
      // Value to write (typically 1). For eventfd, this increments the counter.
      uint64_t value;
    } signal_primitive;

    // For SIGNAL_NOTIFICATION type.
    struct {
      // The notification to signal. Retained during relay lifetime.
      iree_async_notification_t* notification;
      // Number of waiters to wake. Use INT32_MAX to wake all waiters.
      int32_t wake_count;
    } signal_notification;
  };
} iree_async_relay_sink_t;

// Creates a relay sink that signals a primitive (eventfd/event).
static inline iree_async_relay_sink_t iree_async_relay_sink_signal_primitive(
    iree_async_primitive_t primitive, uint64_t value) {
  iree_async_relay_sink_t sink;
  sink.type = IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE;
  sink.signal_primitive.primitive = primitive;
  sink.signal_primitive.value = value;
  return sink;
}

// Creates a relay sink that signals a notification.
static inline iree_async_relay_sink_t iree_async_relay_sink_signal_notification(
    iree_async_notification_t* notification, int32_t wake_count) {
  iree_async_relay_sink_t sink;
  sink.type = IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION;
  sink.signal_notification.notification = notification;
  sink.signal_notification.wake_count = wake_count;
  return sink;
}

//===----------------------------------------------------------------------===//
// Relay flags
//===----------------------------------------------------------------------===//

// Flags controlling relay behavior.
enum iree_async_relay_flag_bits_e {
  IREE_ASYNC_RELAY_FLAG_NONE = 0,

  // Re-arm the source after each transfer (persistent monitoring).
  // When set: relay stays active until explicitly unregistered.
  // When not set: relay fires once and auto-cleans up.
  IREE_ASYNC_RELAY_FLAG_PERSISTENT = 1u << 0,

  // Take ownership of the source primitive (fd will be closed on cleanup).
  // Only valid for PRIMITIVE source type. The proactor closes the fd when
  // the relay is unregistered or when a one-shot relay fires.
  IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE = 1u << 1,

  // Check source for error status before firing sink.
  // When set: POLLERR/POLLHUP on source suppresses sink firing.
  // When not set: sink fires on any source activity (including errors).
  IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE = 1u << 2,
};
typedef uint32_t iree_async_relay_flags_t;

//===----------------------------------------------------------------------===//
// Relay error callback
//===----------------------------------------------------------------------===//

// Callback invoked when a persistent relay fails to re-arm its source.
//
// This can happen if the underlying syscall fails (ENOMEM, EBADF on source fd,
// etc.). After this callback fires, the relay transitions to a faulted state
// and will no longer monitor the source. The relay should be unregistered.
//
// The callback is invoked from within poll() on the proactor's thread.
// The callback takes ownership of |status| and must consume it (e.g., log it,
// convert it, or call iree_status_ignore()).
typedef void(IREE_API_PTR* iree_async_relay_error_fn_t)(
    void* user_data, iree_async_relay_t* relay, iree_status_t status);

// Error callback with user data. Set fn to NULL if no error callback is needed.
typedef struct iree_async_relay_error_callback_t {
  iree_async_relay_error_fn_t fn;
  void* user_data;
} iree_async_relay_error_callback_t;

// Returns an empty error callback (no-op on error).
static inline iree_async_relay_error_callback_t
iree_async_relay_error_callback_none(void) {
  iree_async_relay_error_callback_t callback = {NULL, NULL};
  return callback;
}

//===----------------------------------------------------------------------===//
// Relay struct
//===----------------------------------------------------------------------===//

// Relay instance connecting a source event to a sink action.
//
// Doubly-linked into the proactor's relay list for O(1) removal.
// Platform-specific fields are in the |platform| union.
struct iree_async_relay_t {
  // Intrusive doubly-linked list links.
  struct iree_async_relay_t* next;
  struct iree_async_relay_t* prev;

  // Owning proactor.
  iree_async_proactor_t* proactor;

  // Source and sink specifications (copied from registration).
  iree_async_relay_source_t source;
  iree_async_relay_sink_t sink;

  // Flags from registration.
  iree_async_relay_flags_t flags;

  // Error callback for re-arm failures (may be NULL).
  iree_async_relay_error_callback_t error_callback;

  // For notification source: the epoch we're waiting on.
  // Updated on registration and after each fire.
  uint32_t wait_epoch;

  // Allocator used for this struct (for deallocation).
  iree_allocator_t allocator;

  union {
    struct {
      // Lifecycle state (iree_async_io_uring_relay_state_t values).
      // io_uring needs ACTIVE/ZOMBIE/PENDING_REARM/FAULTED because
      // POLL_REMOVE and ASYNC_CANCEL are asynchronous.
      uint32_t state;
      // Stable buffer for async eventfd WRITE SQE.
      // Must remain valid while the SQE is in flight.
      uint64_t write_buffer;
    } io_uring;
    struct {
      // Per-notification relay chain linkage (poll thread only).
      // Non-NULL when this relay has a NOTIFICATION source and is linked
      // into the source notification's relay_list.
      struct iree_async_relay_t* notification_relay_next;
    } posix;
    struct {
      // Per-notification relay chain linkage (poll thread only).
      // Same pattern as POSIX — singly-linked through the source
      // notification's relay_list.
      struct iree_async_relay_t* notification_relay_next;
    } iocp;
  } platform;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_RELAY_H_
