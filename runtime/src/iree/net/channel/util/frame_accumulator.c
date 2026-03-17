// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/frame_accumulator.h"

#include <string.h>

iree_status_t iree_net_frame_accumulator_initialize(
    iree_net_frame_accumulator_t* accumulator, iree_host_size_t max_frame_size,
    iree_net_frame_length_callback_t frame_length,
    iree_net_frame_complete_callback_t on_frame_complete) {
  memset(accumulator, 0, sizeof(*accumulator));
  accumulator->frame_length = frame_length;
  accumulator->on_frame_complete = on_frame_complete;
  accumulator->buffer_capacity = max_frame_size;
  accumulator->buffer_used = 0;
  return iree_ok_status();
}

void iree_net_frame_accumulator_deinitialize(
    iree_net_frame_accumulator_t* accumulator) {
  memset(accumulator, 0, sizeof(*accumulator));
}

void iree_net_frame_accumulator_reset(
    iree_net_frame_accumulator_t* accumulator) {
  accumulator->buffer_used = 0;
}

iree_status_t iree_net_frame_accumulator_push_lease(
    iree_net_frame_accumulator_t* accumulator, iree_async_buffer_lease_t* lease,
    iree_host_size_t valid_bytes) {
  if (valid_bytes == 0) {
    iree_async_buffer_lease_release(lease);
    return iree_ok_status();
  }

  const uint8_t* data = iree_async_span_ptr(lease->span);
  iree_host_size_t remaining = valid_bytes;

  while (remaining > 0) {
    // Case A: No buffered data - try zero-copy path.
    if (accumulator->buffer_used == 0) {
      iree_const_byte_span_t available =
          iree_make_const_byte_span(data, remaining);
      iree_host_size_t frame_size = accumulator->frame_length.fn(
          accumulator->frame_length.user_data, available);

      if (frame_size > accumulator->buffer_capacity) {
        iree_async_buffer_lease_release(lease);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "frame size %" PRIhsz " exceeds max %" PRIhsz,
                                frame_size, accumulator->buffer_capacity);
      }

      if (frame_size > 0 && frame_size <= remaining) {
        // Zero-copy: complete frame in buffer, pass lease to callback.
        iree_const_byte_span_t frame =
            iree_make_const_byte_span(data, frame_size);
        iree_status_t status = accumulator->on_frame_complete.fn(
            accumulator->on_frame_complete.user_data, frame, lease);
        if (!iree_status_is_ok(status)) {
          // Always release our handle. If callback retained the lease (called
          // iree_async_buffer_lease_retain), it has its own reference.
          iree_async_buffer_lease_release(lease);
          return status;
        }
        data += frame_size;
        remaining -= frame_size;
        continue;  // Check for more frames in same buffer.
      }

      // Frame not complete - need to buffer. Fall through to Case B.
    }

    // Case B: Buffer partial frame data.
    // We copy only what's needed for the CURRENT frame, ensuring buffer_used
    // never exceeds frame_size. This guarantees leftover=0 after delivery,
    // allowing immediate transition to zero-copy for subsequent frames.
    iree_const_byte_span_t buffered = iree_make_const_byte_span(
        accumulator->buffer, accumulator->buffer_used);
    iree_host_size_t frame_size = accumulator->frame_length.fn(
        accumulator->frame_length.user_data, buffered);

    iree_host_size_t bytes_needed;
    if (frame_size == 0) {
      // Don't know frame size yet - copy one byte at a time until we can
      // determine the frame length from the header.
      bytes_needed = accumulator->buffer_used + 1;
    } else {
      bytes_needed = frame_size;
    }

    // Copy only what we need to complete/continue this frame.
    iree_host_size_t need_to_copy = bytes_needed - accumulator->buffer_used;
    iree_host_size_t space_available =
        accumulator->buffer_capacity - accumulator->buffer_used;
    iree_host_size_t to_copy =
        iree_min(iree_min(remaining, space_available), need_to_copy);

    // Detect stuck condition: we have input data (remaining > 0, guaranteed by
    // loop condition) but cannot copy any of it (to_copy == 0) and still don't
    // know the frame size (frame_size == 0). This happens when frame_length_fn
    // needs more header bytes than max_frame_size allows - either a protocol
    // mismatch or malicious input designed to stall the receive pump.
    if (to_copy == 0 && frame_size == 0) {
      iree_async_buffer_lease_release(lease);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "unable to determine frame size within %" PRIhsz
                              " byte buffer",
                              accumulator->buffer_capacity);
    }

    memcpy(accumulator->buffer + accumulator->buffer_used, data, to_copy);
    accumulator->buffer_used += to_copy;
    data += to_copy;
    remaining -= to_copy;

    // Re-check frame size with updated buffer.
    buffered = iree_make_const_byte_span(accumulator->buffer,
                                         accumulator->buffer_used);
    frame_size = accumulator->frame_length.fn(
        accumulator->frame_length.user_data, buffered);

    if (frame_size > accumulator->buffer_capacity) {
      iree_async_buffer_lease_release(lease);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "frame size %" PRIhsz " exceeds max %" PRIhsz,
                              frame_size, accumulator->buffer_capacity);
    }

    if (frame_size > 0 && frame_size <= accumulator->buffer_used) {
      // Complete frame in buffer - deliver with NULL lease (must consume now).
      iree_const_byte_span_t frame =
          iree_make_const_byte_span(accumulator->buffer, frame_size);
      iree_status_t status = accumulator->on_frame_complete.fn(
          accumulator->on_frame_complete.user_data, frame, NULL);
      if (!iree_status_is_ok(status)) {
        iree_async_buffer_lease_release(lease);
        return status;
      }

      // Shift any leftover data to buffer start.
      iree_host_size_t leftover = accumulator->buffer_used - frame_size;
      if (leftover > 0) {
        memmove(accumulator->buffer, accumulator->buffer + frame_size,
                leftover);
      }
      accumulator->buffer_used = leftover;
      // Continue loop - if remaining > 0 and buffer_used == 0, next iteration
      // will try zero-copy path.
    }
  }

  // All data processed, release the lease.
  iree_async_buffer_lease_release(lease);
  return iree_ok_status();
}
