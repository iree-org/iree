// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/trace_buffer.h"

#include "iree/hal/drivers/amdgpu/device/kernels.h"
#include "iree/hal/drivers/amdgpu/device/tracing.h"
#include "iree/hal/drivers/amdgpu/util/kfd.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_trace_buffer_t
//===----------------------------------------------------------------------===//

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE) && \
    IREE_HAL_AMDGPU_TRACING_FEATURES

// TODO(benvanik): maybe move somewhere common? only two uses so far in the same
// code path (queue init) so just pulling from there for now.
void iree_hal_amdgpu_kernel_dispatch(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_queue_t* queue,
    IREE_AMDGPU_DEVICE_PTR const iree_hal_amdgpu_device_kernel_args_t*
        kernel_args,
    const uint32_t grid_size[3], IREE_AMDGPU_DEVICE_PTR void* kernarg_address,
    hsa_signal_t completion_signal);

// Asynchronously dispatches the device-side trace initializer kernel:
// `iree_hal_amdgpu_device_trace_buffer_initialize`.
static void iree_hal_amdgpu_trace_buffer_enqueue_device_initialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer, uint32_t signal_count,
    hsa_queue_t* control_queue,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t completion_signal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_count);

  // Distributed by groups of signals across all signals.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args =
      &trace_buffer->kernels->iree_hal_amdgpu_device_trace_buffer_initialize;
  const uint32_t grid_size[3] = {signal_count, 1, 1};

  // Populate kernargs needed by the dispatch.
  // TODO(benvanik): use an HSA API for copying this? Seems to work as-is.
  kernargs->trace_buffer = trace_buffer->device_buffer;

  // Dispatch the initialization kernel asynchronously.
  iree_hal_amdgpu_kernel_dispatch(trace_buffer->libhsa, control_queue,
                                  kernel_args, grid_size, kernargs,
                                  completion_signal);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_trace_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, int kfd_fd,
    iree_string_view_t executor_name, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_queue_t* control_queue,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t ringbuffer_capacity,
    const iree_hal_amdgpu_device_library_t* device_library,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_buffer_t* out_trace_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_trace_buffer, 0, sizeof(*out_trace_buffer));
  out_trace_buffer->libhsa = libhsa;
  out_trace_buffer->kfd_fd = kfd_fd;
  out_trace_buffer->device_agent = device_agent;
  out_trace_buffer->kernels = kernels;

  // Ringbuffer must be a power-of-two for the indexing tricks we use.
  if (!iree_device_size_is_power_of_two(ringbuffer_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trace ringbuffer capacity must be a power-of-two "
                            "(provided %" PRIdsz ")");
  }

  // Query the UID of the device so that we can communicate with the kfd.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), device_agent,
                              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID,
                              &out_trace_buffer->device_driver_uid));

  // Query device library code ranges used to translate embedded data from
  // pointers on the device to pointers on the host.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_device_library_populate_agent_code_range(
              device_library, device_agent,
              &out_trace_buffer->device_library_code_range));

  // Allocate ringbuffer. Only the device will produce and only the host will
  // consume so we explicitly restrict access.
  hsa_amd_memory_access_desc_t access_descs[2] = {
      (hsa_amd_memory_access_desc_t){
          .agent_handle = device_agent,
          .permissions = HSA_ACCESS_PERMISSION_RW,
      },
      (hsa_amd_memory_access_desc_t){
          .agent_handle = host_agent,
          .permissions = HSA_ACCESS_PERMISSION_RO,
      },
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_vmem_ringbuffer_initialize(
              libhsa, device_agent, memory_pool, ringbuffer_capacity,
              IREE_ARRAYSIZE(access_descs), access_descs,
              &out_trace_buffer->ringbuffer));

  // Allocate device-side trace buffer instance.
  iree_hal_amdgpu_device_trace_buffer_t* device_buffer = NULL;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), memory_pool, sizeof(*device_buffer),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, (void**)&device_buffer);
  out_trace_buffer->device_buffer = device_buffer;
  if (iree_status_is_ok(status)) {
    hsa_agent_t access_agents[2] = {
        host_agent,
        device_agent,
    };
    status = iree_hsa_amd_agents_allow_access(
        IREE_LIBHSA(libhsa), IREE_ARRAYSIZE(access_agents), access_agents, NULL,
        device_buffer);
  }

  // Link back the device to the host trace buffer.
  // Used when the device posts messages to the host in order to route back to
  // this particular instance.
  device_buffer->host_trace_buffer = (uint64_t)out_trace_buffer;

  device_buffer->ringbuffer_base = out_trace_buffer->ringbuffer.ring_base_ptr;
  device_buffer->ringbuffer_capacity = out_trace_buffer->ringbuffer.capacity;

  // Perform initial calibration so we can setup the context.
  // This may change over time due to clock drift.
  uint64_t cpu_timestamp = 0;
  uint64_t gpu_timestamp = 0;
  float timestamp_period = 1.0f;  // maybe?
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_clock_counters_t clock_counters = {0};
    status = iree_hal_amdgpu_kfd_get_clock_counters(
        out_trace_buffer->kfd_fd, out_trace_buffer->device_driver_uid,
        &clock_counters);
    // DO NOT SUBMIT tracy timestamp calibration
    // derive tracy-compatible timestamps
    // store initial counters in the trace buffer as a timebase
    cpu_timestamp = iree_tracing_time();
    gpu_timestamp = clock_counters.gpu_clock_counter;
  }

  // Allocate tracing context used to sink events.
  if (iree_status_is_ok(status)) {
    out_trace_buffer->tracing_context =
        iree_tracing_context_allocate(executor_name.data, executor_name.size);
  }

  // Allocate a tracing context.
  // This is a limited resource today in Tracy (255 max) and if we ever need
  // more we'll have to fix that limit (painfully).
  if (iree_status_is_ok(status)) {
    device_buffer->executor_id =
        (iree_hal_amdgpu_trace_executor_id_t)iree_tracing_gpu_context_allocate(
            IREE_TRACING_GPU_CONTEXT_TYPE_OPENCL, executor_name.data,
            executor_name.size,
            /*is_calibrated=*/false, cpu_timestamp, gpu_timestamp,
            timestamp_period);
  }

  // Asynchronously issue device-side initialization for the ringbuffer.
  //
  // NOTE: this must happen after all other initialization is complete so that
  // the device has all information available for use.
  //
  // If any part of initialization fails the completion signal must be waited
  // on to ensure the device is not still using the trace buffer resources.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_trace_buffer_enqueue_device_initialize(
        out_trace_buffer, IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY,
        control_queue, kernargs, initialization_signal);
  } else {
    iree_hal_amdgpu_trace_buffer_deinitialize(out_trace_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_trace_buffer_deinitialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  if (!trace_buffer->libhsa) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush any remaining information in the buffer to complete the trace.
  // This should only be called when all device-side emitters have terminated
  // and we have exclusive access to the buffer.
  if (trace_buffer->ringbuffer.ring_base_ptr) {
    IREE_IGNORE_ERROR(iree_hal_amdgpu_trace_buffer_flush(trace_buffer));
  }

  if (trace_buffer->device_buffer) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(
        IREE_LIBHSA(trace_buffer->libhsa), trace_buffer->device_buffer));
  }

  iree_hal_amdgpu_vmem_ringbuffer_deinitialize(trace_buffer->libhsa,
                                               &trace_buffer->ringbuffer);

  iree_tracing_context_free(trace_buffer->tracing_context);

  IREE_TRACE_ZONE_END(z0);
}

static void* iree_hal_amdgpu_trace_buffer_translate_ptr(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer, uint64_t device_ptr) {
  const uint64_t device_code_ptr =
      trace_buffer->device_library_code_range.device_ptr;
  if (device_ptr >= device_code_ptr &&
      device_ptr <
          device_code_ptr + trace_buffer->device_library_code_range.size) {
    // Address is in the device code range so translate it to the host copy of
    // the code.
    return (void*)((uint64_t)device_ptr - device_code_ptr +
                   trace_buffer->device_library_code_range.host_ptr);
  } else {
    // Address is outside of the device code range and likely a valid host
    // pointer or some other virtual address the host can access.
    return (void*)device_ptr;
  }
}

static const iree_hal_amdgpu_trace_src_loc_t*
iree_hal_amdgpu_trace_buffer_translate_src_loc(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer,
    iree_hal_amdgpu_trace_src_loc_ptr_t device_ptr) {
  return (const iree_hal_amdgpu_trace_src_loc_t*)
      iree_hal_amdgpu_trace_buffer_translate_ptr(trace_buffer, device_ptr);
}

static const char* iree_hal_amdgpu_trace_buffer_translate_literal(
    const iree_hal_amdgpu_trace_buffer_t* trace_buffer,
    iree_hal_amdgpu_trace_string_literal_ptr_t device_ptr) {
  return (const char*)iree_hal_amdgpu_trace_buffer_translate_ptr(trace_buffer,
                                                                 device_ptr);
}

// Useful debugging tool for tracing:
#define IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(...)
// #define IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(...) fprintf(stderr, __VA_ARGS__)

iree_status_t iree_hal_amdgpu_trace_buffer_flush(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // DO NOT SUBMIT tracy flush timestamp rebasing
  // can use to adjust like GpuAgent::TranslateTime
  // https://sourcegraph.com/github.com/ROCm/ROCR-Runtime@909b82d4632b86dff0faadcb19488a43d2108686/-/blob/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp?L2048
  iree_hal_amdgpu_clock_counters_t clock_counters = {0};
  iree_status_t status = iree_hal_amdgpu_kfd_get_clock_counters(
      trace_buffer->kfd_fd, trace_buffer->device_driver_uid, &clock_counters);

  // start_ts + timeshift
  // (uint64_t) $9 = 1731392576844442103
  // 1731392576844442103/1000000000

  // (end - start) / system_frequency = seconds
  //
  // https://github.com/ROCm/hsa-class/blob/master/src/hsa_rsrc_factory.h#L222

#define COUNT 2
  iree_hal_amdgpu_clock_counters_t counters[COUNT];
  uint64_t clock_times[COUNT];
  const uint32_t steps = COUNT - 1;
  for (int i = 0; i < COUNT; ++i) {
    iree_hal_amdgpu_kfd_get_clock_counters(
        trace_buffer->kfd_fd, trace_buffer->device_driver_uid, &counters[i]);
    clock_times[i] = iree_tracing_time();
  }
  const uint64_t tracy_base = clock_times[0];
  const uint64_t gc_base = counters[0].gpu_clock_counter;
  const uint64_t gerror =
      (counters[COUNT - 1].gpu_clock_counter - counters[0].gpu_clock_counter) /
      (2 * steps);
  (void)gerror;
  uint64_t tracy_accum = 0;
  uint64_t gc_accum = 0;
  for (int i = 0; i < COUNT; ++i) {
    tracy_accum += (clock_times[i] - tracy_base);
    gc_accum += (counters[i].gpu_clock_counter - gc_base);
  }
  uint64_t tracy_v = (tracy_accum / COUNT) + tracy_base;
  uint64_t gc_v = (gc_accum / COUNT) + gc_base;
  double scalar = tracy_accum / (double)gc_accum;

  static uint64_t last_calibration_time = 0;
  if (!last_calibration_time) last_calibration_time = tracy_v;
  // uint64_t cpu_delta = tracy_v - last_calibration_time;
  last_calibration_time = tracy_v;

#define TRANSLATE_TIMESTAMP(ticks) \
  (int64_t) tracy_v - (((int64_t)gc_v - (int64_t)ticks) * scalar)
#define TRANSLATE_TIMESTAMP2(ticks) ticks

  // DO NOT SUBMIT
  // iree_tracing_context_calibrate_executor(trace_buffer->tracing_context,
  //                                         /*executor_id=*/0, cpu_delta,
  //                                         tracy_v, gc_v);

  // DO NOT SUBMIT
  // should memcpy entire buffer locally and reset before processing
  // currently missing cache and touching PCIE every single access

  // Consume all packets from the last flush to the new write offset.
  const uint64_t write_commit_offset =
      iree_atomic_load(&trace_buffer->device_buffer->write_commit_offset,
                       iree_memory_order_acquire);
  const uint64_t read_reserve_offset =
      iree_atomic_exchange(&trace_buffer->last_read_offset, write_commit_offset,
                           iree_memory_order_relaxed);
  const uint8_t* ring_base_ptr = trace_buffer->ringbuffer.ring_base_ptr;
  const uint64_t ring_mask = trace_buffer->ringbuffer.capacity - 1;
  for (uint64_t offset = read_reserve_offset; offset < write_commit_offset;) {
    const uint8_t* packet_ptr = ring_base_ptr + (offset & ring_mask);
    const iree_hal_amdgpu_trace_event_type_t event_type = packet_ptr[0];
    switch (event_type) {
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN: {
        const iree_hal_amdgpu_trace_zone_begin_t* event =
            (const iree_hal_amdgpu_trace_zone_begin_t*)packet_ptr;
        const iree_hal_amdgpu_trace_src_loc_t* src_loc =
            iree_hal_amdgpu_trace_buffer_translate_src_loc(trace_buffer,
                                                           event->src_loc);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_BEGIN %s %s %s %d\n",
            src_loc->name, src_loc->function, src_loc->file, src_loc->line);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_zone_begin(
            trace_buffer->tracing_context, timestamp,
            (const iree_tracing_location_t*)src_loc);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END: {
        const iree_hal_amdgpu_trace_zone_end_t* event =
            (const iree_hal_amdgpu_trace_zone_end_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_END\n");
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_zone_end(trace_buffer->tracing_context, timestamp);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64: {
        const iree_hal_amdgpu_trace_zone_value_i64_t* event =
            (const iree_hal_amdgpu_trace_zone_value_i64_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_I64 %" PRIi64 " %016" PRIx64
            "\n",
            event->value, event->value);
        iree_tracing_context_zone_value_i64(trace_buffer->tracing_context,
                                            event->value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL: {
        const iree_hal_amdgpu_trace_zone_value_text_literal_t* event =
            (const iree_hal_amdgpu_trace_zone_value_text_literal_t*)packet_ptr;
        const char* value = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->value);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_LITERAL `%s`\n",
            value);
        iree_tracing_context_zone_value_text_literal(
            trace_buffer->tracing_context, value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC: {
        const iree_hal_amdgpu_trace_zone_value_text_dynamic_t* event =
            (const iree_hal_amdgpu_trace_zone_value_text_dynamic_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_ZONE_VALUE_TEXT_DYNAMIC `%.*s`\n",
            (int)event->length, event->value);
        iree_tracing_context_zone_value_text_dynamic(
            trace_buffer->tracing_context, event->value, event->length);
        offset += sizeof(*event) + event->length;
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN: {
        const iree_hal_amdgpu_trace_execution_zone_begin_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_begin_t*)packet_ptr;
        const iree_hal_amdgpu_trace_src_loc_t* src_loc =
            iree_hal_amdgpu_trace_buffer_translate_src_loc(trace_buffer,
                                                           event->src_loc);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_BEGIN [%02" PRId32
            "] q=%u %s %s %s %d\n",
            event->execution_query_id, event->execution_query_id, src_loc->name,
            src_loc->function, src_loc->file, src_loc->line);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->issue_timestamp);
        iree_tracing_context_execution_zone_begin(
            trace_buffer->tracing_context, timestamp,
            (const iree_tracing_location_t*)src_loc, event->executor_id,
            event->execution_query_id);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END: {
        const iree_hal_amdgpu_trace_execution_zone_end_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_end_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_END [%02" PRId32
            "] q=%u\n",
            event->execution_query_id, event->execution_query_id);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->issue_timestamp);
        iree_tracing_context_execution_zone_end(trace_buffer->tracing_context,
                                                timestamp, event->executor_id,
                                                event->execution_query_id);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH: {
        const iree_hal_amdgpu_trace_execution_zone_dispatch_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_dispatch_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_DISPATCH "
            "executor=%d query_id=%d export_loc=%" PRIx64 "\n",
            event->executor_id, event->execution_query_id, event->export_loc);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->issue_timestamp);
        const iree_tracing_location_t* src_loc = NULL;
        switch (event->zone_type) {
          case IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_DISPATCH:
          case IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY:
          case IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL:
          case IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_INTERNAL:
            src_loc = (const iree_tracing_location_t*)event->export_loc;
            break;
        }
        iree_tracing_context_execution_zone_begin(
            trace_buffer->tracing_context, timestamp, src_loc,
            event->executor_id, event->execution_query_id);
        iree_tracing_context_execution_zone_end(
            trace_buffer->tracing_context, timestamp + 1, event->executor_id,
            event->execution_query_id | 0x8000u);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH: {
        const iree_hal_amdgpu_trace_execution_zone_notify_batch_t* event =
            (const iree_hal_amdgpu_trace_execution_zone_notify_batch_t*)
                packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_EXECUTION_ZONE_NOTIFY_BATCH\n");
        // NOTE: end timestamps may not be populated and we have to ensure they
        // are always at least the same as the begin timestamp.
        for (uint16_t i = 0; i < event->execution_query_count; ++i) {
          const uint16_t query_id =
              (event->execution_query_id_base + i) &
              (IREE_HAL_AMDGPU_DEVICE_QUERY_RINGBUFFER_CAPACITY - 1);
          const iree_hal_amdgpu_trace_agent_time_range_t time_range =
              event->execution_time_ranges[i];
          const uint64_t begin = TRANSLATE_TIMESTAMP2(time_range.begin);
          const uint64_t end = TRANSLATE_TIMESTAMP2(time_range.end);
          iree_tracing_context_execution_zone_notify(
              trace_buffer->tracing_context, event->executor_id, query_id,
              begin);
          // DO NOT SUBMIT we should only do if it's a split event - need to
          // know somehow - may have to include per event in time ranges
          iree_tracing_context_execution_zone_notify(
              trace_buffer->tracing_context, event->executor_id,
              query_id | 0x8000u, iree_max(end, begin + 1));
        }
        offset += sizeof(*event) + event->execution_query_count *
                                       sizeof(event->execution_time_ranges[0]);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC: {
        const iree_hal_amdgpu_trace_memory_alloc_t* event =
            (const iree_hal_amdgpu_trace_memory_alloc_t*)packet_ptr;
        const char* pool = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->pool);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_ALLOC\n");
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_memory_alloc(trace_buffer->tracing_context,
                                          timestamp, pool, event->ptr,
                                          event->size);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE: {
        const iree_hal_amdgpu_trace_memory_free_t* event =
            (const iree_hal_amdgpu_trace_memory_free_t*)packet_ptr;
        const char* pool = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->pool);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_MEMORY_FREE\n");
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_memory_free(trace_buffer->tracing_context,
                                         timestamp, pool, event->ptr);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL: {
        const iree_hal_amdgpu_trace_message_literal_t* event =
            (const iree_hal_amdgpu_trace_message_literal_t*)packet_ptr;
        const char* value = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->value);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_LITERAL `%s`\n", value);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_message_literal(trace_buffer->tracing_context,
                                             timestamp, value);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC: {
        const iree_hal_amdgpu_trace_message_dynamic_t* event =
            (const iree_hal_amdgpu_trace_message_dynamic_t*)packet_ptr;
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_MESSAGE_DYNAMIC `%.*s`\n",
            (int)event->length, event->value);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_message_dynamic(trace_buffer->tracing_context,
                                             timestamp, event->value,
                                             event->length);
        offset += sizeof(*event) + event->length;
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG: {
        const iree_hal_amdgpu_trace_plot_config_t* event =
            (const iree_hal_amdgpu_trace_plot_config_t*)packet_ptr;
        const char* name = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->name);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_CONFIG %s\n", name);
        const bool step = iree_all_bits_set(
            event->plot_flags, IREE_HAL_AMDGPU_TRACE_PLOT_FLAG_DISCRETE);
        const bool fill = iree_all_bits_set(
            event->plot_flags, IREE_HAL_AMDGPU_TRACE_PLOT_FLAG_FILL);
        iree_tracing_context_plot_config(trace_buffer->tracing_context, name,
                                         event->plot_type, step, fill,
                                         event->color);
        offset += sizeof(*event);
      } break;
      case IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64: {
        const iree_hal_amdgpu_trace_plot_value_i64_t* event =
            (const iree_hal_amdgpu_trace_plot_value_i64_t*)packet_ptr;
        const char* plot_name = iree_hal_amdgpu_trace_buffer_translate_literal(
            trace_buffer, event->plot_name);
        IREE_HAL_AMDGPU_PRINT_TRACE_PACKET(
            "IREE_HAL_AMDGPU_TRACE_EVENT_PLOT_VALUE_I64 %s = %" PRIi64 "\n",
            plot_name, event->value);
        const uint64_t timestamp = TRANSLATE_TIMESTAMP(event->timestamp);
        iree_tracing_context_plot_value_i64(trace_buffer->tracing_context,
                                            timestamp, plot_name, event->value);
        offset += sizeof(*event);
      } break;
      default: {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "invalid trace event type %d; possibly corrupt ringbuffer",
            event_type);
      } break;
    }
  }

  // Notify the device that we've read up to the write offset.
  iree_atomic_store(&trace_buffer->device_buffer->read_commit_offset,
                    write_commit_offset, iree_memory_order_release);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#else

iree_status_t iree_hal_amdgpu_trace_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, int kfd_fd,
    iree_string_view_t executor_name, hsa_agent_t host_agent,
    hsa_agent_t device_agent, hsa_queue_t* control_queue,
    hsa_amd_memory_pool_t memory_pool, iree_device_size_t ringbuffer_capacity,
    const iree_hal_amdgpu_device_library_t* device_library,
    const iree_hal_amdgpu_device_kernels_t* kernels,
    IREE_AMDGPU_DEVICE_PTR iree_hal_amdgpu_device_trace_buffer_kernargs_t*
        kernargs,
    hsa_signal_t initialization_signal, iree_allocator_t host_allocator,
    iree_hal_amdgpu_trace_buffer_t* out_trace_buffer) {
  // No-op.
  return iree_ok_status();
}

void iree_hal_amdgpu_trace_buffer_deinitialize(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  // No-op.
}

iree_status_t iree_hal_amdgpu_trace_buffer_flush(
    iree_hal_amdgpu_trace_buffer_t* trace_buffer) {
  // No-op.
  return iree_ok_status();
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
