// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/relay.h"

#include <errno.h>
#include <poll.h>
#include <unistd.h>

#include "iree/async/notification.h"
#include "iree/async/platform/posix/event_set.h"
#include "iree/async/platform/posix/fd_map.h"
#include "iree/async/platform/posix/proactor.h"

//===----------------------------------------------------------------------===//
// Source and sink operations
//===----------------------------------------------------------------------===//

// Executes the sink action synchronously.
// Returns true on success, false on failure with errno set.
static bool iree_async_posix_relay_fire_sink(iree_async_relay_t* relay) {
  switch (relay->sink.type) {
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE: {
      uint64_t value = relay->sink.signal_primitive.value;
      ssize_t written = write(relay->sink.signal_primitive.primitive.value.fd,
                              &value, sizeof(value));
      if (written != sizeof(value)) {
        return false;
      }
      break;
    }
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION: {
      iree_async_notification_signal(
          relay->sink.signal_notification.notification,
          relay->sink.signal_notification.wake_count);
      break;
    }
  }
  return true;
}

// Drains a level-triggered source fd to prevent busy-loops.
//
// eventfd and pipes remain readable until drained. With poll/epoll
// level-triggered monitoring, the fd will be reported as ready on every poll
// cycle until drained. Draining resets the fd to non-readable state so poll
// only fires again when new data arrives.
//
// Handles EINTR (common on macOS with signal-heavy environments) by retrying.
// Stops on EAGAIN/EWOULDBLOCK (fully drained) or read returning 0.
static void iree_async_posix_relay_drain_source(iree_async_relay_t* relay) {
  if (relay->source.type != IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) return;
  uint64_t drain_buffer;
  for (;;) {
    ssize_t bytes_read = read(relay->source.primitive.value.fd, &drain_buffer,
                              sizeof(drain_buffer));
    if (bytes_read > 0)
      break;  // Drained one value (eventfd atomically resets).
    if (bytes_read == 0) break;    // EOF.
    if (errno == EINTR) continue;  // Retry on signal interruption.
    break;  // EAGAIN or other error — already drained or unusable.
  }
}

// Invokes the error callback (if registered) and cleans up the relay.
// Takes ownership of |status|.
static void iree_async_posix_relay_fault(iree_async_proactor_posix_t* proactor,
                                         iree_async_relay_t* relay,
                                         iree_status_t status) {
  if (relay->error_callback.fn) {
    relay->error_callback.fn(relay->error_callback.user_data, relay, status);
  } else {
    iree_status_ignore(status);
  }
  // Clean up the relay immediately (POSIX lifecycle is synchronous).
  iree_async_proactor_posix_unregister_relay(proactor, relay);
}

// Unlinks a relay from the proactor's doubly-linked list.
static void iree_async_posix_relay_unlink(iree_async_proactor_posix_t* proactor,
                                          iree_async_relay_t* relay) {
  if (relay->prev) {
    relay->prev->next = relay->next;
  } else {
    proactor->relays = relay->next;
  }
  if (relay->next) {
    relay->next->prev = relay->prev;
  }
  relay->next = NULL;
  relay->prev = NULL;
}

// Releases retained notifications and frees the relay struct.
static void iree_async_posix_relay_release_resources(
    iree_async_relay_t* relay) {
  if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_notification_release(relay->source.notification);
  }
  if (relay->sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_release(
        relay->sink.signal_notification.notification);
  }
  iree_allocator_free(relay->allocator, relay);
}

// Removes a relay from its source notification's relay_list.
// Returns true if the relay was found and removed.
static bool iree_async_posix_relay_remove_from_notification_list(
    iree_async_relay_t* relay) {
  iree_async_notification_t* notification = relay->source.notification;
  iree_async_relay_t** previous = &notification->platform.posix.relay_list;
  iree_async_relay_t* current = notification->platform.posix.relay_list;
  while (current) {
    if (current == relay) {
      *previous = current->platform.posix.notification_relay_next;
      current->platform.posix.notification_relay_next = NULL;
      return true;
    }
    previous = &current->platform.posix.notification_relay_next;
    current = current->platform.posix.notification_relay_next;
  }
  return false;
}

// Returns true if the notification has any consumers (pending async waits or
// relay subscribers) that require its fd to be monitored.
static bool iree_async_posix_notification_has_consumers(
    iree_async_notification_t* notification) {
  return notification->platform.posix.pending_waits != NULL ||
         notification->platform.posix.relay_list != NULL;
}

// Activates a notification's fd in the event_set and fd_map if not already
// active (i.e., if this is the first consumer). Called when a new async wait
// or relay is registered.
static iree_status_t iree_async_posix_notification_activate(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification) {
  int fd = notification->platform.posix.primitive.value.fd;
  iree_status_t status =
      iree_async_posix_event_set_add(proactor->event_set, fd, POLLIN);
  if (!iree_status_is_ok(status)) return status;
  status = iree_async_posix_fd_map_insert(
      &proactor->fd_map, fd, IREE_ASYNC_POSIX_FD_HANDLER_NOTIFICATION,
      notification);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(
        iree_async_posix_event_set_remove(proactor->event_set, fd));
    return status;
  }
  return iree_ok_status();
}

// Deactivates a notification's fd from the event_set and fd_map when the last
// consumer is removed.
static void iree_async_posix_notification_deactivate(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification) {
  int fd = notification->platform.posix.primitive.value.fd;
  iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
  iree_status_ignore(
      iree_async_posix_event_set_remove(proactor->event_set, fd));
}

//===----------------------------------------------------------------------===//
// Register relay
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_posix_register_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_source_t source,
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
  relay->wait_epoch = 0;
  relay->allocator = proactor->base.allocator;
  memset(&relay->platform, 0, sizeof(relay->platform));

  // Retain notifications used in source/sink.
  if (source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_notification_retain(source.notification);
  }
  if (sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_retain(sink.signal_notification.notification);
  }

  // Source-specific registration.
  if (source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
    // Primitive source: register fd in event_set + fd_map.
    int fd = source.primitive.value.fd;
    iree_status_t status =
        iree_async_posix_event_set_add(proactor->event_set, fd, POLLIN);
    if (iree_status_is_ok(status)) {
      status = iree_async_posix_fd_map_insert(
          &proactor->fd_map, fd, IREE_ASYNC_POSIX_FD_HANDLER_RELAY, relay);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(
            iree_async_posix_event_set_remove(proactor->event_set, fd));
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_async_posix_relay_release_resources(relay);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  } else {
    // Notification source: link into the notification's per-notification
    // relay_list and activate the notification fd if this is the first
    // consumer.
    iree_async_notification_t* notification = source.notification;
    bool was_active = iree_async_posix_notification_has_consumers(notification);

    // Head insert into notification's relay list.
    relay->platform.posix.notification_relay_next =
        notification->platform.posix.relay_list;
    notification->platform.posix.relay_list = relay;

    // Capture the current epoch for change detection.
    relay->wait_epoch = (uint32_t)iree_atomic_load(&notification->epoch,
                                                   iree_memory_order_acquire);

    // Activate the notification's fd if this is the first consumer.
    if (!was_active) {
      iree_status_t status =
          iree_async_posix_notification_activate(proactor, notification);
      if (!iree_status_is_ok(status)) {
        // Rollback: remove from notification relay list.
        notification->platform.posix.relay_list =
            relay->platform.posix.notification_relay_next;
        relay->platform.posix.notification_relay_next = NULL;
        iree_async_posix_relay_release_resources(relay);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
  }

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

void iree_async_proactor_posix_unregister_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_t* relay) {
  if (!relay) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
    // Remove from fd_map and event_set BEFORE closing the fd.
    // kqueue returns EBADF if the fd is already closed, and epoll has
    // fd-reuse races if close happens first.
    int fd = relay->source.primitive.value.fd;
    iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
    iree_status_ignore(
        iree_async_posix_event_set_remove(proactor->event_set, fd));
  } else if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    // Remove from the source notification's relay list.
    iree_async_posix_relay_remove_from_notification_list(relay);
    // Deactivate the notification fd if this was the last consumer.
    iree_async_notification_t* notification = relay->source.notification;
    if (!iree_async_posix_notification_has_consumers(notification)) {
      iree_async_posix_notification_deactivate(proactor, notification);
    }
  }

  // Close source fd if we own it.
  if ((relay->flags & IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE) &&
      relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
    close(relay->source.primitive.value.fd);
  }

  // Unlink from relay list.
  iree_async_posix_relay_unlink(proactor, relay);

  // Release retained notifications and free.
  iree_async_posix_relay_release_resources(relay);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Poll loop dispatch helpers
//===----------------------------------------------------------------------===//

void iree_async_proactor_posix_dispatch_relay(
    iree_async_proactor_posix_t* proactor, iree_async_relay_t* relay,
    short revents) {
  bool should_fire = true;

  // Check ERROR_SENSITIVE flag: suppress sink on error/hangup conditions.
  // On Linux epoll, pipe close delivers POLLERR|POLLHUP without POLLIN, but
  // on macOS kqueue, EVFILT_READ+EV_EOF translates to POLLIN|POLLHUP (the fd
  // is technically readable — returning EOF). We suppress on any POLLERR or
  // POLLHUP regardless of POLLIN because the connection is dead either way.
  if (relay->flags & IREE_ASYNC_RELAY_FLAG_ERROR_SENSITIVE) {
    if (revents & (POLLERR | POLLHUP)) {
      should_fire = false;
    }
  }

  if (should_fire) {
    if (!iree_async_posix_relay_fire_sink(relay)) {
      int saved_errno = errno;
      iree_async_posix_relay_fault(
          proactor, relay,
          iree_make_status(iree_status_code_from_errno(saved_errno),
                           "relay sink write failed"));
      return;  // Relay has been cleaned up by fault handler.
    }
  }

  bool is_persistent = (relay->flags & IREE_ASYNC_RELAY_FLAG_PERSISTENT) != 0;
  if (is_persistent) {
    // Drain source to prevent busy-loops with level-triggered monitoring.
    iree_async_posix_relay_drain_source(relay);
  } else {
    // One-shot: remove from fd_map + event_set, then clean up.
    int fd = relay->source.primitive.value.fd;
    iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
    iree_status_ignore(
        iree_async_posix_event_set_remove(proactor->event_set, fd));

    // Close source fd if we own it.
    if (relay->flags & IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE) {
      close(fd);
    }

    // Unlink and free.
    iree_async_posix_relay_unlink(proactor, relay);
    iree_async_posix_relay_release_resources(relay);
  }
}

//===----------------------------------------------------------------------===//
// Notification-source relay dispatch
//===----------------------------------------------------------------------===//

void iree_async_proactor_posix_dispatch_notification_relays(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification) {
  uint32_t current_epoch = (uint32_t)iree_atomic_load(
      &notification->epoch, iree_memory_order_acquire);

  iree_async_relay_t** previous = &notification->platform.posix.relay_list;
  iree_async_relay_t* relay = notification->platform.posix.relay_list;
  while (relay) {
    iree_async_relay_t* next = relay->platform.posix.notification_relay_next;

    if (current_epoch == relay->wait_epoch) {
      // No epoch advancement — skip this relay.
      previous = &relay->platform.posix.notification_relay_next;
      relay = next;
      continue;
    }

    // Epoch advanced — fire the sink.
    if (!iree_async_posix_relay_fire_sink(relay)) {
      int saved_errno = errno;
      // Remove from the notification relay list before fault handler
      // calls unregister (which would also try to remove from list).
      *previous = next;
      relay->platform.posix.notification_relay_next = NULL;
      // Check if notification should deactivate after list removal.
      if (!iree_async_posix_notification_has_consumers(notification)) {
        iree_async_posix_notification_deactivate(proactor, notification);
      }
      iree_async_posix_relay_fault(
          proactor, relay,
          iree_make_status(iree_status_code_from_errno(saved_errno),
                           "relay sink write failed"));
      relay = next;
      continue;
    }

    bool is_persistent = (relay->flags & IREE_ASYNC_RELAY_FLAG_PERSISTENT) != 0;
    if (is_persistent) {
      // Update wait_epoch for next dispatch cycle.
      relay->wait_epoch = current_epoch;
      previous = &relay->platform.posix.notification_relay_next;
    } else {
      // One-shot: remove from notification relay list.
      *previous = next;
      relay->platform.posix.notification_relay_next = NULL;
      // Check if notification should deactivate after list removal.
      if (!iree_async_posix_notification_has_consumers(notification)) {
        iree_async_posix_notification_deactivate(proactor, notification);
      }
      // Unlink from proactor relay list and free.
      iree_async_posix_relay_unlink(proactor, relay);
      iree_async_posix_relay_release_resources(relay);
    }

    relay = next;
  }
}

//===----------------------------------------------------------------------===//
// Cleanup
//===----------------------------------------------------------------------===//

void iree_async_proactor_posix_destroy_all_relays(
    iree_async_proactor_posix_t* proactor) {
  iree_async_relay_t* relay = proactor->relays;
  while (relay) {
    iree_async_relay_t* next = relay->next;

    if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
      // Remove from fd_map and event_set before close (kqueue safety).
      int fd = relay->source.primitive.value.fd;
      iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
      iree_status_ignore(
          iree_async_posix_event_set_remove(proactor->event_set, fd));
    } else if (relay->source.type ==
               IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
      // Remove from notification's relay list (no need to deactivate —
      // we're destroying everything).
      iree_async_posix_relay_remove_from_notification_list(relay);
    }

    // Close owned fds.
    if ((relay->flags & IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE) &&
        relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
      close(relay->source.primitive.value.fd);
    }

    // Release notifications and free.
    iree_async_posix_relay_release_resources(relay);

    relay = next;
  }
  proactor->relays = NULL;
}
