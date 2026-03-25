// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/relay.h"

#include <errno.h>
#include <poll.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/platform/io_uring/uring.h"

//===----------------------------------------------------------------------===//
// Relay CQE encoding
//===----------------------------------------------------------------------===//

// Encodes a relay pointer into a user_data value for SQEs. The relay pointer
// is stored as the payload with TAG_RELAY, using the shared internal encoding
// scheme from proactor.h.
#define iree_io_uring_relay_encode(relay) \
  iree_io_uring_internal_encode(IREE_IO_URING_TAG_RELAY, (relay))

//===----------------------------------------------------------------------===//
// Source and sink operations
//===----------------------------------------------------------------------===//

// Executes the sink action synchronously.
// For SIGNAL_PRIMITIVE: writes to eventfd.
// For SIGNAL_NOTIFICATION: signals the notification directly.
// Returns true on success, false on failure with errno set.
static bool iree_async_io_uring_relay_fire_sink(iree_async_relay_t* relay) {
  switch (relay->sink.type) {
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE: {
      // Write the value to the eventfd/event handle.
      relay->platform.io_uring.write_buffer =
          relay->sink.signal_primitive.value;
      ssize_t written = write(relay->sink.signal_primitive.primitive.value.fd,
                              &relay->platform.io_uring.write_buffer,
                              sizeof(relay->platform.io_uring.write_buffer));
      if (written != sizeof(relay->platform.io_uring.write_buffer)) {
        // Write failed. errno is set by write().
        return false;
      }
      break;
    }
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION: {
      // Signal the notification directly. This uses an atomic increment and
      // either a futex wake or eventfd write internally. The epoch update is
      // always successful; the wake/write is best-effort (waiters will see
      // the epoch change regardless).
      iree_async_notification_signal(
          relay->sink.signal_notification.notification,
          relay->sink.signal_notification.wake_count);
      break;
    }
  }
  return true;
}

// Drains a level-triggered source fd to prevent busy-loops with multishot POLL.
//
// eventfd and similar level-triggered fds remain readable until drained. With
// multishot POLL_ADD, the kernel continuously delivers CQEs while the fd is
// readable. Draining resets the fd to non-readable state so the poll only
// fires again when new data arrives.
//
// This is only needed for persistent poll-based sources. One-shot relays clean
// up after first fire.
//
// Returns true on success (drained or nothing to drain). Returns false on hard
// error (errno set) — caller should fault the relay.
static bool iree_async_io_uring_relay_drain_source(iree_async_relay_t* relay) {
  uint64_t drain_buffer;
  switch (relay->source.type) {
    case IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE: {
      // Drain the source fd. For eventfd this reads and resets the counter.
      // EAGAIN/EWOULDBLOCK means already drained (non-blocking fd). Any other
      // error indicates a broken fd (EBADF, EIO) that would busy-loop the
      // multishot poll.
      ssize_t result = read(relay->source.primitive.value.fd, &drain_buffer,
                            sizeof(drain_buffer));
      if (result < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        return false;
      }
      break;
    }
    case IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION: {
      // Drain the notification's eventfd.
      ssize_t result =
          read(relay->source.notification->platform.io_uring.primitive.value.fd,
               &drain_buffer, sizeof(drain_buffer));
      if (result < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        return false;
      }
      break;
    }
  }
  return true;
}

// Returns the fd to poll for the relay's source.
// For PRIMITIVE: the primitive's fd.
// For NOTIFICATION: the notification's eventfd.
static int iree_async_io_uring_relay_source_fd(iree_async_relay_t* relay) {
  switch (relay->source.type) {
    case IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE:
      return relay->source.primitive.value.fd;
    case IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION:
      return relay->source.notification->platform.io_uring.primitive.value.fd;
  }
  return -1;
}

// Fills an SQE to monitor the relay's source. Caller must provide a valid SQE.
static void iree_async_io_uring_relay_fill_source_sqe(
    iree_async_relay_t* relay, iree_io_uring_sqe_t* sqe) {
  memset(sqe, 0, sizeof(*sqe));

  switch (relay->source.type) {
    case IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE: {
      // POLL_ADD on the primitive fd.
      sqe->opcode = IREE_IORING_OP_POLL_ADD;
      sqe->fd = relay->source.primitive.value.fd;
      sqe->poll32_events = POLLIN;
      // Use multishot for persistent relays.
      if (iree_any_bit_set(relay->flags, IREE_ASYNC_RELAY_FLAG_PERSISTENT)) {
        sqe->len = IREE_IORING_POLL_ADD_MULTI;
      }
      break;
    }
    case IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION: {
      // POLL_ADD on the notification's eventfd.
      iree_async_notification_t* notification = relay->source.notification;
      sqe->opcode = IREE_IORING_OP_POLL_ADD;
      sqe->fd = notification->platform.io_uring.primitive.value.fd;
      sqe->poll32_events = POLLIN;
      if (iree_any_bit_set(relay->flags, IREE_ASYNC_RELAY_FLAG_PERSISTENT)) {
        sqe->len = IREE_IORING_POLL_ADD_MULTI;
      }
      break;
    }
  }

  sqe->user_data = iree_io_uring_relay_encode(relay);
}

// Acquires an SQE and fills it to monitor the relay's source.
// Returns a status on failure - use only for initial registration where
// failure is a real error. For re-arming, use get_sqe + fill_source_sqe
// directly to avoid status allocation on expected backpressure.
static iree_status_t iree_async_io_uring_relay_submit_source(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay) {
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit relay source monitoring");
  }
  iree_async_io_uring_relay_fill_source_sqe(relay, sqe);
  iree_io_uring_ring_sq_unlock(&proactor->ring);
  return iree_ok_status();
}

// Transitions a relay to FAULTED state and invokes the error callback.
// Takes ownership of |status|. If no callback is registered, the status is
// ignored (freed).
static void iree_async_io_uring_relay_fault(iree_async_relay_t* relay,
                                            iree_status_t status) {
  relay->platform.io_uring.state = IREE_ASYNC_IO_URING_RELAY_STATE_FAULTED;
  if (relay->error_callback.fn) {
    // Transfer ownership to callback.
    relay->error_callback.fn(relay->error_callback.user_data, relay, status);
  } else {
    // No callback - must still consume the status.
    iree_status_ignore(status);
  }
}

// Performs final cleanup of a relay: unlinks from the proactor's relay list,
// closes owned source fd, releases retained notifications, and frees the
// struct. The relay must have no in-flight kernel operation (PENDING_REARM,
// FAULTED, or received final CQE). Caller must not access the relay after this
// call.
static void iree_async_io_uring_relay_cleanup(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay) {
  // Unlink from proactor's relay list.
  if (relay->prev) {
    relay->prev->next = relay->next;
  } else {
    proactor->relays = relay->next;
  }
  if (relay->next) {
    relay->next->prev = relay->prev;
  }

  // Close source fd if we own it.
  if (iree_any_bit_set(relay->flags,
                       IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE) &&
      relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
    close(relay->source.primitive.value.fd);
  }

  // Release retained notifications.
  if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_notification_release(relay->source.notification);
  }
  if (relay->sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_release(
        relay->sink.signal_notification.notification);
  }

  // Free the relay struct.
  iree_allocator_free(relay->allocator, relay);
}

//===----------------------------------------------------------------------===//
// Register relay
//===----------------------------------------------------------------------===//

iree_status_t iree_async_io_uring_register_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_relay);
  *out_relay = NULL;

  // Validate source.
  switch (source.type) {
    case IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE:
      if (source.primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          source.primitive.value.fd < 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "relay source primitive must be a valid fd");
      }
      break;
    case IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION:
      if (!source.notification) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "relay source notification must not be NULL");
      }
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown relay source type %d", (int)source.type);
  }

  // Validate sink.
  switch (sink.type) {
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE:
      if (sink.signal_primitive.primitive.type !=
              IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          sink.signal_primitive.primitive.value.fd < 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "relay sink signal_primitive must be a valid fd");
      }
      break;
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION:
      if (!sink.signal_notification.notification) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "relay sink signal_notification must not be NULL");
      }
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown relay sink type %d", (int)sink.type);
  }

  // Allocate relay struct.
  iree_async_relay_t* relay = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->base.allocator, sizeof(*relay),
                                (void**)&relay));

  // Initialize relay.
  relay->next = NULL;
  relay->prev = NULL;
  relay->proactor = &proactor->base;
  relay->source = source;
  relay->sink = sink;
  relay->flags = flags;
  relay->error_callback = error_callback;
  relay->platform.io_uring.state = IREE_ASYNC_IO_URING_RELAY_STATE_ACTIVE;
  relay->wait_epoch = 0;
  relay->platform.io_uring.write_buffer = 0;
  relay->allocator = proactor->base.allocator;

  // Retain notifications used in source/sink.
  if (source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_notification_retain(source.notification);
  }
  if (sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_retain(sink.signal_notification.notification);
  }

  // Submit source monitoring.
  iree_status_t status =
      iree_async_io_uring_relay_submit_source(proactor, relay);
  if (!iree_status_is_ok(status)) {
    // Rollback: release notifications and free relay.
    if (source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
      iree_async_notification_release(source.notification);
    }
    if (sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
      iree_async_notification_release(sink.signal_notification.notification);
    }
    iree_allocator_free(proactor->base.allocator, relay);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Flush the SQE to the kernel so monitoring begins and the SQ slot is
  // reclaimed. ring_submit errors are ignored because the SQE is already
  // committed via *sq_tail (see register_event_source for full rationale).
  iree_status_ignore(iree_io_uring_ring_submit(&proactor->ring,
                                               /*min_complete=*/0,
                                               /*flags=*/0));

  // Link into proactor's relay list.
  relay->next = proactor->relays;
  if (proactor->relays) {
    proactor->relays->prev = relay;
  }
  proactor->relays = relay;

  *out_relay = relay;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Unregister relay
//===----------------------------------------------------------------------===//

void iree_async_io_uring_unregister_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay) {
  if (!relay) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // PENDING_REARM and FAULTED relays have no in-flight kernel operation.
  // There's nothing to cancel, so clean up immediately.
  if (relay->platform.io_uring.state ==
          IREE_ASYNC_IO_URING_RELAY_STATE_PENDING_REARM ||
      relay->platform.io_uring.state ==
          IREE_ASYNC_IO_URING_RELAY_STATE_FAULTED) {
    iree_async_io_uring_relay_cleanup(proactor, relay);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Mark as zombie so CQE handler knows not to fire sink.
  relay->platform.io_uring.state = IREE_ASYNC_IO_URING_RELAY_STATE_ZOMBIE;

  // Submit POLL_REMOVE or ASYNC_CANCEL to stop source monitoring.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (sqe) {
    memset(sqe, 0, sizeof(*sqe));

    sqe->opcode = IREE_IORING_OP_POLL_REMOVE;
    sqe->fd = -1;
    sqe->addr = iree_io_uring_relay_encode(relay);
    sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL, 0);
    iree_io_uring_ring_sq_unlock(&proactor->ring);

    iree_status_ignore(iree_io_uring_ring_submit(&proactor->ring,
                                                 /*min_complete=*/0,
                                                 /*flags=*/0));
  } else {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    // SQ full — the relay stays zombie. For multishot poll-based relays, the
    // CQE handler will retry POLL_REMOVE on each subsequent CQE delivery.
    // For one-shot sources, the final CQE will trigger cleanup directly.
  }

  // The relay stays in the list as a zombie. The CQE handler will unlink and
  // free it when the final CQE arrives (no CQE_F_MORE or cancel completion).
  // If the proactor is destroyed first, destroy() cleans up zombies.
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// CQE handling
//===----------------------------------------------------------------------===//

void iree_async_io_uring_handle_relay_cqe(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay,
    int32_t result, uint32_t cqe_flags) {
  if (!relay) return;

  bool is_zombie = (relay->platform.io_uring.state ==
                    IREE_ASYNC_IO_URING_RELAY_STATE_ZOMBIE);
  bool has_more = (cqe_flags & IREE_IORING_CQE_F_MORE) != 0;
  bool is_persistent =
      iree_any_bit_set(relay->flags, IREE_ASYNC_RELAY_FLAG_PERSISTENT);

  // Fire sink if not zombie and no error (or not error-sensitive).
  if (!is_zombie) {
    bool should_fire = true;

    // Check for source errors if ERROR_SENSITIVE flag is set.
    if (iree_any_bit_set(relay->flags, IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE)) {
      if (result < 0) {
        // Kernel error (e.g., ECANCELED, EBADF).
        should_fire = false;
      } else if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE ||
                 relay->source.type ==
                     IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
        // For poll-based sources, result contains poll events.
        // Check for error conditions without POLLIN.
        uint32_t poll_events = (uint32_t)result;
        if ((poll_events & (POLLERR | POLLHUP)) && !(poll_events & POLLIN)) {
          should_fire = false;
        }
      }
    }

    if (should_fire) {
      if (!iree_async_io_uring_relay_fire_sink(relay)) {
        // Sink write failed. Capture errno before any other calls.
        int saved_errno = errno;
        iree_async_io_uring_relay_fault(
            relay, iree_make_status(iree_status_code_from_errno(saved_errno),
                                    "relay sink write failed"));
        // Faulted relays are cleaned up below via is_final.
      } else {
        // For persistent poll-based sources, drain the source fd to prevent
        // busy-loops. Level-triggered fds (like eventfd) remain readable until
        // drained; with multishot POLL_ADD, the kernel would otherwise deliver
        // CQEs continuously.
        if (is_persistent && has_more) {
          if (!iree_async_io_uring_relay_drain_source(relay)) {
            int saved_errno = errno;
            iree_async_io_uring_relay_fault(
                relay,
                iree_make_status(iree_status_code_from_errno(saved_errno),
                                 "relay source drain failed"));
          }
        }
      }
    }
  }

  // For zombie relays with multishot still active, attempt to stop the
  // kernel-side poll so the relay can be cleaned up. This handles the case
  // where unregister_relay couldn't submit POLL_REMOVE due to SQ pressure.
  // The multishot keeps delivering CQEs, giving us repeated opportunities
  // to retry until the SQ has space.
  if (is_zombie && has_more) {
    iree_io_uring_ring_sq_lock(&proactor->ring);
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
    if (sqe) {
      memset(sqe, 0, sizeof(*sqe));
      sqe->opcode = IREE_IORING_OP_POLL_REMOVE;
      sqe->fd = -1;
      sqe->addr = iree_io_uring_relay_encode(relay);
      sqe->user_data =
          iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL, 0);
    }
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    // SQE is submitted with the next ring_submit in poll().
    // If SQ was full, the multishot keeps delivering CQEs so we retry.
  }

  // Determine if this is the final CQE for this relay.
  // Faulted relays always need cleanup regardless of has_more.
  bool is_final = !has_more || relay->platform.io_uring.state ==
                                   IREE_ASYNC_IO_URING_RELAY_STATE_FAULTED;

  // Clean up if this is the final CQE. One-shot relays clean up after first
  // completion; persistent poll-based relays clean up when multishot ends.
  if (is_final) {
    iree_async_io_uring_relay_cleanup(proactor, relay);
  }
}

//===----------------------------------------------------------------------===//
// Retry pending re-arms
//===----------------------------------------------------------------------===//

void iree_async_io_uring_retry_pending_relays(
    iree_async_proactor_io_uring_t* proactor) {
  iree_io_uring_ring_sq_lock(&proactor->ring);
  for (iree_async_relay_t* relay = proactor->relays; relay;
       relay = relay->next) {
    if (relay->platform.io_uring.state !=
        IREE_ASYNC_IO_URING_RELAY_STATE_PENDING_REARM) {
      continue;
    }

    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
    if (!sqe) {
      // Still no SQ space. Remaining relays stay pending until next poll.
      break;
    }

    iree_async_io_uring_relay_fill_source_sqe(relay, sqe);
    relay->platform.io_uring.state = IREE_ASYNC_IO_URING_RELAY_STATE_ACTIVE;
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);
  // SQEs are submitted by the caller's next ring_submit in poll().
}
