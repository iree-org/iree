// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/tracing.h"

// NOTE: this header in clang only declares the builtins for va_list-related
// things - if it becomes an issue we can easily inline them here.
#include <stdarg.h>

#include "iree/hal/drivers/amdgpu/device/support/queue.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_query_ringbuffer_t
//===----------------------------------------------------------------------===//

iree_hal_amdgpu_trace_execution_query_id_t
iree_hal_amdgpu_device_query_ringbuffer_acquire(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT
        ringbuffer) {
  // Slice off a single value and return it mapped into a query ID.
  return (ringbuffer->write_index++) &
         (IREE_AMDGPU_ARRAYSIZE(ringbuffer->signals) - 1);
}

uint64_t iree_hal_amdgpu_device_query_ringbuffer_acquire_range(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint16_t count) {
  // Slice off another chunk.
  uint64_t base_index = ringbuffer->write_index;
  ringbuffer->write_index += count;
  return base_index;
}

void iree_hal_amdgpu_device_query_ringbuffer_release_range(
    iree_hal_amdgpu_device_query_ringbuffer_t* IREE_AMDGPU_RESTRICT ringbuffer,
    uint16_t count) {
  // Reset all returned signals.
  for (uint32_t i = ringbuffer->read_index; i < ringbuffer->read_index + count;
       ++i) {
    iree_amd_signal_t* signal =
        &ringbuffer
             ->signals[i & (IREE_AMDGPU_ARRAYSIZE(ringbuffer->signals) - 1)];
    signal->value = 1;
    signal->start_ts = 0;
    signal->end_ts = 0;
  }
  ringbuffer->read_index += count;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_trace_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_TRACING_FEATURES

// Initializes the signals in a trace buffer's query ringbuffer.
// The ringbuffer memory must be zero-initialized by the allocator.
// Must be called prior to acquiring any signals.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_trace_buffer_initialize(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer) {
  const size_t i = iree_hal_amdgpu_device_global_id_x();

  // Initialize device trace buffer memory (once).
  if (i == 0) {
    trace_buffer->read_commit_offset = 0;
    trace_buffer->write_reserve_offset = 0;
    trace_buffer->write_commit_offset = 0;
    trace_buffer->query_ringbuffer.read_index = 0;
    trace_buffer->query_ringbuffer.write_index = 0;
  }

  // Initialize signal.
  iree_amd_signal_t* signal = &trace_buffer->query_ringbuffer.signals[i];
  signal->kind = IREE_AMD_SIGNAL_KIND_USER;
  signal->value = 1;
  signal->event_mailbox_ptr = 0;
  signal->event_id = 0;
  signal->reserved1 = 0;
  signal->start_ts = 0;
  signal->end_ts = 0;
  signal->queue_ptr = 0;
  signal->reserved3[0] = 0;
  signal->reserved3[1] = 0;
}

// Reserves |length| bytes from the trace buffer and returns a pointer to it.
// Callers must populate the entire packet prior to calling
// iree_hal_amdgpu_device_trace_commit_range. Multiple reservations can be made
// between commits to batch the commit logic (which usually involves a host
// interrupt to flush the ringbuffer).
static inline void* iree_hal_amdgpu_device_trace_reserve_range(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    size_t length) {
  // Reserve a range of the requested size from the current reservation offset.
  // NOTE: this is only modified on device and on the agent associated with the
  // scheduler that's calling this and as such only has to be at device scope.
  uint64_t write_offset = iree_amdgpu_scoped_atomic_fetch_add(
      &trace_buffer->write_reserve_offset, length,
      iree_amdgpu_memory_order_relaxed, iree_amdgpu_memory_scope_device);

  // Spin until there's capacity in the ringbuffer. We need to wait until the
  // host catches up to our last flush.
  // WARNING: this may lock up forever if we really spill the ring.
  // TODO(benvanik): find a way to fail here, or throw an interrupt.
  // We could use a signal instead of an atomic but there's no good way to park
  // from the current pc.
  if (write_offset + length -
          iree_amdgpu_scoped_atomic_load(&trace_buffer->read_commit_offset,
                                         iree_amdgpu_memory_order_acquire,
                                         iree_amdgpu_memory_scope_system) >
      trace_buffer->ringbuffer_capacity) {
    iree_amdgpu_yield();
  }

  // Calculate base address of the packet within the ringbuffer. Note that it
  // may extend off the end of the base allocation but so long as the length is
  // in bounds it'll be accessing the physical memory through the subsequent
  // virtual address mapping.
  void* packet_ptr =
      (uint8_t*)trace_buffer->ringbuffer_base +
      (write_offset & iree_hal_amdgpu_device_trace_buffer_mask(trace_buffer));

  return packet_ptr;
}

bool iree_hal_amdgpu_device_trace_commit_range(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer) {
  // Bump the commit offset as seen by the host to the reserve offset at the
  // start of this call. The host may immediately begin reading from its last
  // read_commit_offset up to the new write_commit_offset and we cannot
  // overwrite any of that range until the read_commit_offset has been bumped by
  // the host.
  uint64_t last_reserve_offset = iree_amdgpu_scoped_atomic_load(
      &trace_buffer->write_reserve_offset, iree_amdgpu_memory_order_acquire,
      iree_amdgpu_memory_scope_device);
  uint64_t last_commit_offset = iree_amdgpu_scoped_atomic_exchange(
      &trace_buffer->write_commit_offset, last_reserve_offset,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);

  // If the last commit offset matches the last reserve offset then there were
  // no pending writes to commit and the caller does not need to notify the
  // host.
  return last_reserve_offset != last_commit_offset;
}

#else

// Dummy to appease the loader.
// We could find some macro tricks to remove this entirely but we want to allow
// host and device builds to differ in their settings.
IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_trace_buffer_initialize(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer) {}

bool iree_hal_amdgpu_device_trace_commit_range(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer) {
  return false;
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURES

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION)

iree_hal_amdgpu_zone_id_t iree_hal_amdgpu_device_trace_zone_begin(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc) {
  iree_hal_amdgpu_trace_zone_begin_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_begin_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN;
  packet->timestamp = iree_amdgpu_device_timestamp();
  packet->src_loc = src_loc;
  return 1;
}

void iree_hal_amdgpu_device_trace_zone_end(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer) {
  iree_hal_amdgpu_trace_zone_end_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_end_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END;
  packet->timestamp = iree_amdgpu_device_timestamp();
}

void iree_hal_amdgpu_device_trace_zone_append_value_i64(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    int64_t value) {
  iree_hal_amdgpu_trace_zone_value_i64_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_zone_value_i64_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64;
  packet->value = value;
}

void iree_hal_amdgpu_device_trace_zone_append_text_literal(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal) {
  iree_hal_amdgpu_trace_zone_value_text_literal_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_zone_value_text_literal_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL;
  packet->value = value_literal;
}

void iree_hal_amdgpu_device_trace_zone_append_text_dynamic(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    const char* IREE_AMDGPU_RESTRICT value, size_t value_length) {
  const size_t total_size =
      sizeof(iree_hal_amdgpu_trace_zone_value_text_dynamic_t) + value_length;
  iree_hal_amdgpu_trace_zone_value_text_dynamic_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(trace_buffer, total_size);
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC;
  packet->length = (uint32_t)value_length;
  iree_amdgpu_memcpy(packet->value, value, value_length);
}

void iree_hal_amdgpu_device_trace_plot_configure(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal,
    iree_hal_amdgpu_trace_plot_type_t type,
    iree_hal_amdgpu_trace_plot_flags_t flags,
    iree_hal_amdgpu_trace_color_t color) {
  iree_hal_amdgpu_trace_plot_config_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_plot_config_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG;
  packet->plot_type = type;
  packet->plot_flags = flags;
  packet->color = color;
  packet->name = name_literal;
}

void iree_hal_amdgpu_device_trace_plot_value_i64(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, int64_t value) {
  iree_hal_amdgpu_trace_plot_value_i64_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_plot_value_i64_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64;
  packet->plot_name = name_literal;
  packet->timestamp = iree_amdgpu_device_timestamp();
  packet->value = value;
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_INSTRUMENTATION

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_begin(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    iree_hal_amdgpu_trace_src_loc_ptr_t src_loc) {
  iree_hal_amdgpu_trace_execution_zone_begin_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_execution_zone_begin_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->issue_timestamp = iree_amdgpu_device_timestamp();
  packet->src_loc = src_loc;
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_end(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  iree_hal_amdgpu_trace_execution_zone_end_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_execution_zone_end_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->issue_timestamp = iree_amdgpu_device_timestamp();
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

iree_hal_amdgpu_trace_agent_time_range_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_trace_execution_zone_notify_batch(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id_base,
    uint16_t execution_query_count) {
  iree_hal_amdgpu_trace_execution_zone_notify_batch_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_execution_zone_notify_batch_t) +
              execution_query_count *
                  sizeof(iree_hal_amdgpu_trace_agent_time_range_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id_base = execution_query_id_base;
  packet->execution_query_count = execution_query_count;
  return &packet->execution_time_ranges[0];
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)

iree_hsa_signal_t iree_hal_amdgpu_device_trace_execution_zone_dispatch(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_execution_zone_type_t zone_type, uint64_t export_loc,
    iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  iree_hal_amdgpu_trace_execution_zone_dispatch_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer,
          sizeof(iree_hal_amdgpu_trace_execution_zone_dispatch_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH;
  packet->zone_type = zone_type;
  packet->executor_id = trace_buffer->executor_id;
  packet->execution_query_id = execution_query_id;
  packet->export_loc = export_loc;
  packet->issue_timestamp = iree_amdgpu_device_timestamp();
  return iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
      &trace_buffer->query_ringbuffer, execution_query_id);
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING)

void iree_hal_amdgpu_device_trace_memory_alloc(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr,
    uint64_t size) {
  iree_hal_amdgpu_trace_memory_alloc_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_memory_alloc_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC;
  packet->pool = name_literal;
  packet->timestamp = iree_amdgpu_device_timestamp();
  packet->ptr = ptr;
  packet->size = size;
}

void iree_hal_amdgpu_device_trace_memory_free(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t name_literal, uint64_t ptr) {
  iree_hal_amdgpu_trace_memory_free_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_memory_free_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE;
  packet->timestamp = iree_amdgpu_device_timestamp();
  packet->ptr = ptr;
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_ALLOCATION_TRACKING

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES
//===----------------------------------------------------------------------===//

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES)

void iree_hal_amdgpu_device_trace_message_literal(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_color_t color,
    iree_hal_amdgpu_trace_string_literal_ptr_t value_literal) {
  iree_hal_amdgpu_trace_message_literal_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(
          trace_buffer, sizeof(iree_hal_amdgpu_trace_message_literal_t));
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL;
  packet->timestamp = iree_amdgpu_device_timestamp();
  packet->value = value_literal;
}

void iree_hal_amdgpu_device_trace_message_dynamic(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    iree_hal_amdgpu_trace_color_t color, const char* IREE_AMDGPU_RESTRICT value,
    size_t value_length) {
  const size_t total_size =
      sizeof(iree_hal_amdgpu_trace_message_dynamic_t) + value_length;
  iree_hal_amdgpu_trace_message_dynamic_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(trace_buffer, total_size);
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC;
  packet->length = (uint32_t)value_length;
  packet->timestamp = iree_amdgpu_device_timestamp();
  iree_amdgpu_memcpy(packet->value, value, value_length);
}

// WARNING: this implementation is for debugging purposes only and is likely
// both unsafe and incorrect.
static int iree_hal_amdgpu_itoa_uint64(char* IREE_AMDGPU_RESTRICT buffer,
                                       uint64_t value, int base,
                                       char base_hex_char) {
  int length = 0;
  uint64_t d = 1;
  while (value / d >= base) d *= base;
  while (d != 0) {
    int digit = value / d;
    value %= d;
    d /= base;
    if (length > 0 || digit > 0 || d == 0) {
      if (buffer) {
        buffer[length] = digit + (digit < 10 ? '0' : base_hex_char - 10);
      }
      ++length;
    }
  }
  return length;
}

// WARNING: this implementation is for debugging purposes only and is likely
// both unsafe and incorrect.
static int iree_hal_amdgpu_vsprintf(char* IREE_AMDGPU_RESTRICT buffer,
                                    char const* IREE_AMDGPU_RESTRICT format,
                                    va_list vlist) {
  int length = 0;
  while (format[0] != '\0') {
    int c = format[0];
    ++format;
    if (c != '%') {
      if (buffer) buffer[length] = c;
      ++length;
      continue;
    }
    c = format[0];
    ++format;
    bool is_long = false;
    if (c == 'l') {
      c = format[0];
      ++format;
      is_long = true;
    }
    switch (c) {
      case 0: {
        return length;
      }
      case '%': {
        if (buffer) buffer[length] = '%';
        ++length;
        break;
      }
      case 'c': {
        const char value = va_arg(vlist, int);
        if (buffer) buffer[length] = value;
        ++length;
        break;
      }
      case 's': {
        const char* str = va_arg(vlist, const char*);
        if (buffer) {
          for (; *str; ++str, ++length) {
            buffer[length] = *str;
          }
        } else {
          for (; *str; ++str, ++length);
        }
        break;
      }
      case 'u': {
        const uint64_t value = is_long ? va_arg(vlist, unsigned long int)
                                       : va_arg(vlist, unsigned int);
        length += iree_hal_amdgpu_itoa_uint64(buffer ? &buffer[length] : NULL,
                                              value, 10, 0);
        break;
      }
      case 'x':
      case 'X': {
        const uint64_t value = is_long ? va_arg(vlist, unsigned long int)
                                       : va_arg(vlist, unsigned int);
        length += iree_hal_amdgpu_itoa_uint64(buffer ? &buffer[length] : NULL,
                                              value, 16, c == 'X' ? 'A' : 'a');
        break;
      }
      case 'p': {
        if (buffer) {
          buffer[length + 0] = '0';
          buffer[length + 1] = 'x';
        }
        length += 2;
        const uint64_t value = va_arg(vlist, unsigned long int);
        length += iree_hal_amdgpu_itoa_uint64(buffer ? &buffer[length] : NULL,
                                              value, 16, 'A');
        break;
      }
      default: {
        // Unhandled.
        if (buffer) buffer[length] = '?';
        ++length;
        break;
      }
    }
  }
  return length;
}

void iree_hal_amdgpu_device_trace_debug(
    iree_hal_amdgpu_device_trace_buffer_t* IREE_AMDGPU_RESTRICT trace_buffer,
    const char* IREE_AMDGPU_RESTRICT format, ...) {
  // NOTE: we capture the timestamp before doing our string processing - not
  // that this should be used for performance work but having the message as
  // close in time to the originating site is always nice.
  const iree_hal_amdgpu_trace_agent_timestamp_t timestamp =
      iree_amdgpu_device_timestamp();

  va_list vlist0, vlist1;
  va_start(vlist0, format);
  va_start(vlist1, format);

  // Determine total length of the formatted message in characters.
  int required_length = iree_hal_amdgpu_vsprintf(NULL, format, vlist0);

  // Reserve trace buffer space for the message contents.
  const size_t total_size =
      sizeof(iree_hal_amdgpu_trace_message_dynamic_t) + required_length;
  iree_hal_amdgpu_trace_message_dynamic_t* packet =
      iree_hal_amdgpu_device_trace_reserve_range(trace_buffer, total_size);
  packet->event_type = IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC;
  packet->length = (uint32_t)required_length;
  packet->timestamp = timestamp;

  // Print the message into the reserved space.
  iree_hal_amdgpu_vsprintf(packet->value, format, vlist1);

  va_end(vlist0);
  va_end(vlist1);
}

#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_LOG_MESSAGES
