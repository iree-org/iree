// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_traces.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/profile_aqlprofile.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/libaqlprofile.h"

//===----------------------------------------------------------------------===//
// Executable trace support tables
//===----------------------------------------------------------------------===//

enum { iree_hal_amdgpu_profile_trace_packets_per_event = 3u };
enum { iree_hal_amdgpu_profile_trace_start_packets_per_event = 2u };
enum { iree_hal_amdgpu_profile_trace_default_buffer_size = 16 * 1024 * 1024 };
enum { iree_hal_amdgpu_profile_trace_default_se_mask = 1u };
enum { iree_hal_amdgpu_profile_trace_default_target_cu = 1u };
enum { iree_hal_amdgpu_profile_trace_default_simd_select = 0xFu };

// Per-queue/per-event-ring-slot mutable aqlprofile ATT capture state.
struct iree_hal_amdgpu_profile_trace_slot_t {
  // Callback context retained for the lifetime of |handle|.
  iree_hal_amdgpu_profile_aqlprofile_memory_context_t memory_context;
  // aqlprofile handle owning PM4 programs and trace output storage.
  iree_hal_amdgpu_aqlprofile_handle_t handle;
  // AQL PM4-IB packet templates referencing |handle|'s immutable PM4 programs.
  iree_hal_amdgpu_aqlprofile_att_control_aql_packets_t packets;
  // aqlprofile handle owning the PM4 program for |code_object_marker_packet|.
  iree_hal_amdgpu_aqlprofile_handle_t code_object_marker_handle;
  // AQL PM4-IB packet template that publishes the loaded code-object marker.
  iree_hsa_amd_aql_pm4_ib_packet_t code_object_marker_packet;
  // Code-object id currently represented by |code_object_marker_packet|.
  uint64_t code_object_marker_id;
  // Producer-local trace id assigned when this slot is reserved for a dispatch.
  uint64_t trace_id;
};

// Logical-device profiling session for selected executable traces.
struct iree_hal_amdgpu_profile_trace_session_t {
  // Host allocator used for session and queue slot storage.
  iree_allocator_t host_allocator;
  // Borrowed HSA API table from the logical device.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Dynamically loaded aqlprofile SDK.
  iree_hal_amdgpu_libaqlprofile_t libaqlprofile;
  // Requested bytes in each per-slot ATT trace output buffer.
  uint64_t trace_buffer_size;
  // Shader-engine mask passed to aqlprofile ATT packet generation.
  uint32_t shader_engine_mask;
  // Compute-unit target passed to aqlprofile ATT packet generation.
  uint32_t target_compute_unit;
  // SIMD lane mask passed to aqlprofile ATT packet generation.
  uint32_t simd_select;
  // Next nonzero producer-local executable trace id.
  iree_atomic_int64_t next_trace_id;
};

// Context threaded through aqlprofile ATT data iteration.
typedef struct iree_hal_amdgpu_profile_trace_collect_context_t {
  // Host queue owning the trace slot.
  iree_hal_amdgpu_host_queue_t* queue;
  // Profile sink receiving trace chunks.
  iree_hal_profile_sink_t* sink;
  // Active profile session id.
  uint64_t session_id;
  // Copied dispatch event record correlated with the trace.
  const iree_hal_profile_dispatch_event_t* event;
  // Trace slot whose handle is being decoded.
  const iree_hal_amdgpu_profile_trace_slot_t* slot;
  // First callback or sink failure encountered during iteration.
  iree_status_t status;
} iree_hal_amdgpu_profile_trace_collect_context_t;

static bool iree_hal_amdgpu_profile_trace_mode_requested(
    const iree_hal_device_profiling_options_t* options) {
  return iree_hal_device_profiling_options_requests_executable_traces(options);
}

static iree_status_t iree_hal_amdgpu_profile_trace_create_packets(
    const iree_hal_amdgpu_profile_trace_session_t* session,
    iree_hal_amdgpu_physical_device_t* physical_device,
    const iree_hal_amdgpu_profile_aqlprofile_memory_context_t* memory_context,
    iree_hal_amdgpu_aqlprofile_handle_t* out_handle,
    iree_hal_amdgpu_aqlprofile_att_control_aql_packets_t* out_packets) {
  iree_hal_amdgpu_aqlprofile_att_parameter_t parameters[6];
  memset(parameters, 0, sizeof(parameters));
  uint32_t parameter_count = 0;
  parameters[parameter_count++] = (iree_hal_amdgpu_aqlprofile_att_parameter_t){
      .parameter_name =
          IREE_HAL_AMDGPU_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET,
      .value = session->target_compute_unit,
  };
  parameters[parameter_count++] = (iree_hal_amdgpu_aqlprofile_att_parameter_t){
      .parameter_name = IREE_HAL_AMDGPU_AQLPROFILE_PARAMETER_NAME_SE_MASK,
      .value = session->shader_engine_mask,
  };
  parameters[parameter_count++] = (iree_hal_amdgpu_aqlprofile_att_parameter_t){
      .parameter_name =
          IREE_HAL_AMDGPU_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION,
      .value = session->simd_select,
  };
  parameters[parameter_count++] = (iree_hal_amdgpu_aqlprofile_att_parameter_t){
      .parameter_name =
          IREE_HAL_AMDGPU_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE,
      .value = (uint32_t)session->trace_buffer_size,
  };
  if ((session->trace_buffer_size >> 32) != 0) {
    parameters[parameter_count++] =
        (iree_hal_amdgpu_aqlprofile_att_parameter_t){
            .parameter_name =
                IREE_HAL_AMDGPU_AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH,
            .value = (uint32_t)(session->trace_buffer_size >> 32),
        };
  }
  parameters[parameter_count++] = (iree_hal_amdgpu_aqlprofile_att_parameter_t){
      .parameter_name =
          IREE_HAL_AMDGPU_AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP,
      .value = IREE_HAL_AMDGPU_AQLPROFILE_ATT_PARAMETER_RT_TIMESTAMP_ENABLE,
  };

  iree_hal_amdgpu_aqlprofile_att_profile_t profile = {
      .agent = physical_device->device_agent,
      .parameters = parameters,
      .parameter_count = parameter_count,
  };
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      &session->libaqlprofile,
      session->libaqlprofile.aqlprofile_att_create_packets(
          out_handle, out_packets, profile,
          iree_hal_amdgpu_profile_aqlprofile_memory_alloc,
          iree_hal_amdgpu_profile_aqlprofile_memory_dealloc,
          iree_hal_amdgpu_profile_aqlprofile_memory_copy,
          (void*)memory_context),
      "creating AMDGPU ATT PM4 packets");
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_trace_create_code_object_marker(
    const iree_hal_amdgpu_profile_trace_session_t* session,
    iree_hal_amdgpu_physical_device_t* physical_device,
    const iree_hal_amdgpu_profile_aqlprofile_memory_context_t* memory_context,
    const iree_hal_profile_executable_code_object_load_record_t* load_record,
    iree_hal_amdgpu_aqlprofile_handle_t* out_handle,
    iree_hsa_amd_aql_pm4_ib_packet_t* out_packet) {
  iree_hal_amdgpu_aqlprofile_att_code_object_data_t code_object_data = {
      .id = load_record->code_object_id,
      .address = (uint64_t)load_record->load_delta,
      .length = load_record->load_size,
      .agent = physical_device->device_agent,
      .is_unload = 0,
      .from_start = 1,
  };
  IREE_RETURN_IF_AQLPROFILE_ERROR(
      &session->libaqlprofile,
      session->libaqlprofile.aqlprofile_att_codeobj_marker(
          out_packet, out_handle, code_object_data,
          iree_hal_amdgpu_profile_aqlprofile_memory_alloc,
          iree_hal_amdgpu_profile_aqlprofile_memory_dealloc,
          (void*)memory_context),
      "creating AMDGPU ATT code-object marker packet");
  return iree_ok_status();
}

static void iree_hal_amdgpu_profile_trace_destroy_packets(
    const iree_hal_amdgpu_libaqlprofile_t* libaqlprofile,
    iree_hal_amdgpu_aqlprofile_handle_t* handle) {
  if (!handle->handle) return;
  libaqlprofile->aqlprofile_att_delete_packets(*handle);
  handle->handle = 0;
}

iree_status_t iree_hal_amdgpu_profile_trace_session_create(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_trace_session_t** out_session) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_session = NULL;

  if (!iree_hal_amdgpu_profile_trace_mode_requested(options)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!options->sink)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU executable trace profiling requires a profile sink");
  }
  if (IREE_UNLIKELY(iree_hal_profile_capture_filter_is_default(
          &options->capture_filter))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU executable trace profiling requires a capture filter; use an "
        "export pattern, command buffer/id, physical device, or queue filter "
        "to avoid tracing every dispatch");
  }

  iree_hal_amdgpu_profile_trace_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*session),
                                (void**)&session));
  memset(session, 0, sizeof(*session));
  session->host_allocator = host_allocator;
  session->libhsa = &logical_device->system->libhsa;
  session->trace_buffer_size =
      iree_hal_amdgpu_profile_trace_default_buffer_size;
  session->shader_engine_mask = iree_hal_amdgpu_profile_trace_default_se_mask;
  session->target_compute_unit =
      iree_hal_amdgpu_profile_trace_default_target_cu;
  session->simd_select = iree_hal_amdgpu_profile_trace_default_simd_select;
  iree_atomic_store(&session->next_trace_id, 1, iree_memory_order_relaxed);

  iree_status_t status = iree_hal_amdgpu_libaqlprofile_initialize(
      session->libhsa, iree_string_view_list_empty(), host_allocator,
      &session->libaqlprofile);
  if (iree_status_is_ok(status) &&
      !iree_hal_amdgpu_libaqlprofile_has_att_support(&session->libaqlprofile)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "loaded AMDGPU aqlprofile library does not export ATT/SQTT packet "
        "generation and data iteration symbols");
  }

  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    iree_hal_amdgpu_libaqlprofile_deinitialize(&session->libaqlprofile);
    iree_allocator_free(host_allocator, session);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_profile_trace_session_destroy(
    iree_hal_amdgpu_profile_trace_session_t* session) {
  if (!session) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = session->host_allocator;
  iree_hal_amdgpu_libaqlprofile_deinitialize(&session->libaqlprofile);
  iree_allocator_free(host_allocator, session);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_amdgpu_profile_trace_session_is_active(
    const iree_hal_amdgpu_profile_trace_session_t* session) {
  return session != NULL;
}

static iree_hal_amdgpu_profile_trace_slot_t*
iree_hal_amdgpu_host_queue_profile_trace_slot(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  const uint32_t event_index =
      (uint32_t)(event_position & queue->profiling.dispatch_event_mask);
  return &queue->profiling.trace_slots[event_index];
}

iree_status_t iree_hal_amdgpu_host_queue_enable_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_trace_session_t* session) {
  if (!iree_hal_amdgpu_profile_trace_session_is_active(session)) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  if (IREE_UNLIKELY(!queue->profiling.dispatch_event_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable trace profiling requires dispatch event storage");
  }
  if (IREE_UNLIKELY(!iree_any_bit_set(
          queue->vendor_packet_capabilities,
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable trace profiling requires AQL PM4-IB packet support");
  }
  if (IREE_UNLIKELY(queue->profiling.trace_slots)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU executable trace profiling is already "
                            "enabled");
  }

  iree_host_size_t slot_storage_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &slot_storage_size,
              IREE_STRUCT_FIELD(queue->profiling.dispatch_event_capacity,
                                iree_hal_amdgpu_profile_trace_slot_t, NULL)));
  iree_hal_amdgpu_profile_trace_slot_t* slots = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, slot_storage_size,
                                (void**)&slots));
  memset(slots, 0, slot_storage_size);

  queue->profiling.trace_session = session;
  queue->profiling.trace_slots = slots;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_disable_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->profiling.trace_slots) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_trace_session_t* session =
      queue->profiling.trace_session;
  for (uint32_t i = 0; i < queue->profiling.dispatch_event_capacity; ++i) {
    iree_hal_amdgpu_profile_trace_destroy_packets(
        &session->libaqlprofile, &queue->profiling.trace_slots[i].handle);
    iree_hal_amdgpu_profile_trace_destroy_packets(
        &session->libaqlprofile,
        &queue->profiling.trace_slots[i].code_object_marker_handle);
  }
  iree_allocator_free(queue->host_allocator, queue->profiling.trace_slots);
  queue->profiling.trace_session = NULL;
  queue->profiling.trace_slots = NULL;

  IREE_TRACE_ZONE_END(z0);
}

uint32_t iree_hal_amdgpu_host_queue_profile_trace_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count || !queue->profiling.trace_session) return 0;
  return reservation.event_count *
         iree_hal_amdgpu_profile_trace_packets_per_event;
}

uint32_t iree_hal_amdgpu_host_queue_profile_trace_start_packet_count(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count || !queue->profiling.trace_session) return 0;
  return reservation.event_count *
         iree_hal_amdgpu_profile_trace_start_packets_per_event;
}

iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  iree_hal_amdgpu_profile_trace_session_t* session =
      queue->profiling.trace_session;
  if (!reservation.event_count || !session) return iree_ok_status();

  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[queue->device_ordinal];
  for (uint32_t event_ordinal = 0; event_ordinal < reservation.event_count;
       ++event_ordinal) {
    const uint64_t event_position =
        reservation.first_event_position + event_ordinal;
    iree_hal_amdgpu_profile_trace_slot_t* slot =
        iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
    if (!slot->handle.handle) {
      slot->memory_context =
          (iree_hal_amdgpu_profile_aqlprofile_memory_context_t){
              .libhsa = session->libhsa,
              .device_agent = physical_device->device_agent,
              .host_memory_pools = &physical_device->host_memory_pools,
              .device_coarse_pool =
                  physical_device->coarse_block_pools.large.memory_pool,
          };
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_trace_create_packets(
          session, physical_device, &slot->memory_context, &slot->handle,
          &slot->packets));
    }
    slot->trace_id = (uint64_t)iree_atomic_fetch_add(&session->next_trace_id, 1,
                                                     iree_memory_order_relaxed);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t executable_id) {
  iree_hal_amdgpu_profile_trace_session_t* session =
      queue->profiling.trace_session;
  if (!session) return iree_ok_status();

  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  if (IREE_UNLIKELY(!slot->handle.handle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable trace slot must be prepared before its code-object "
        "marker");
  }

  const uint32_t physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  iree_hal_profile_executable_code_object_load_record_t load_record;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_lookup_code_object_load(
      &logical_device->profile_metadata, executable_id, physical_device_ordinal,
      &load_record));

  if (slot->code_object_marker_handle.handle &&
      slot->code_object_marker_id == load_record.code_object_id) {
    return iree_ok_status();
  }

  iree_hal_amdgpu_profile_trace_destroy_packets(
      &session->libaqlprofile, &slot->code_object_marker_handle);
  memset(&slot->code_object_marker_packet, 0,
         sizeof(slot->code_object_marker_packet));
  slot->code_object_marker_id = 0;

  iree_hal_amdgpu_physical_device_t* physical_device =
      logical_device->physical_devices[queue->device_ordinal];
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_trace_create_code_object_marker(
      session, physical_device, &slot->memory_context, &load_record,
      &slot->code_object_marker_handle, &slot->code_object_marker_packet));
  slot->code_object_marker_id = load_record.code_object_id;
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_emplace_profile_trace_packet_at(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hsa_amd_aql_pm4_ib_packet_t* source_packet,
    uint64_t first_packet_id, uint32_t packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
      &queue->aql_ring, first_packet_id + packet_index);
  iree_hal_amdgpu_profile_aqlprofile_emplace_pm4_ib_packet(
      source_packet, packet, packet_control, iree_hsa_signal_null(),
      &packet_headers[packet_index], &packet_setups[packet_index]);
}

void iree_hal_amdgpu_host_queue_emplace_profile_trace_start_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_emplace_profile_trace_packet_at(
      queue, &slot->packets.start_packet, first_packet_id, first_packet_index,
      packet_control, packet_headers, packet_setups);
}

void iree_hal_amdgpu_host_queue_emplace_profile_trace_code_object_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_emplace_profile_trace_packet_at(
      queue, &slot->code_object_marker_packet, first_packet_id,
      first_packet_index, packet_control, packet_headers, packet_setups);
}

void iree_hal_amdgpu_host_queue_emplace_profile_trace_stop_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t first_packet_id, uint32_t first_packet_index,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    uint16_t* packet_headers, uint16_t* packet_setups) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_emplace_profile_trace_packet_at(
      queue, &slot->packets.stop_packet, first_packet_id, first_packet_index,
      packet_control, packet_headers, packet_setups);
}

static void iree_hal_amdgpu_host_queue_commit_profile_trace_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hsa_amd_aql_pm4_ib_packet_t* source_packet, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  uint16_t header = 0;
  uint16_t setup = 0;
  iree_hal_amdgpu_profile_aqlprofile_emplace_pm4_ib_packet(
      source_packet, packet, packet_control, iree_hsa_signal_null(), &header,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_profile_trace_start_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_commit_profile_trace_packet(
      queue, &slot->packets.start_packet, packet_id, packet_control);
}

void iree_hal_amdgpu_host_queue_commit_profile_trace_code_object_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_commit_profile_trace_packet(
      queue, &slot->code_object_marker_packet, packet_id, packet_control);
}

void iree_hal_amdgpu_host_queue_commit_profile_trace_stop_packet(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position,
    uint64_t packet_id, iree_hal_amdgpu_aql_packet_control_t packet_control) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  iree_hal_amdgpu_host_queue_commit_profile_trace_packet(
      queue, &slot->packets.stop_packet, packet_id, packet_control);
}

static iree_status_t iree_hal_amdgpu_profile_trace_write_chunk(
    iree_hal_amdgpu_profile_trace_collect_context_t* context,
    uint32_t shader_engine, const void* trace_data,
    iree_host_size_t data_size) {
  const iree_hal_profile_dispatch_event_t* event = context->event;
  const iree_hal_amdgpu_host_queue_t* queue = context->queue;

  iree_hal_profile_executable_trace_record_t record =
      iree_hal_profile_executable_trace_record_default();
  record.format = IREE_HAL_PROFILE_EXECUTABLE_TRACE_FORMAT_AMDGPU_ATT;
  record.flags = IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_DISPATCH_EVENT;
  if (iree_any_bit_set(event->flags,
                       IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER)) {
    record.flags |= IREE_HAL_PROFILE_EXECUTABLE_TRACE_FLAG_COMMAND_OPERATION;
  }
  record.shader_engine = shader_engine;
  record.trace_id = context->slot->trace_id;
  record.dispatch_event_id = event->event_id;
  record.submission_id = event->submission_id;
  record.command_buffer_id = event->command_buffer_id;
  record.executable_id = event->executable_id;
  record.stream_id = iree_hal_amdgpu_host_queue_profile_stream_id(queue);
  record.command_index = event->command_index;
  record.export_ordinal = event->export_ordinal;
  record.physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  record.queue_ordinal =
      iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  record.data_length = data_size;

  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_TRACES;
  metadata.name = iree_make_cstring_view("amdgpu.att");
  metadata.session_id = context->session_id;
  metadata.stream_id = record.stream_id;
  metadata.event_id = event->event_id;
  metadata.executable_id = event->executable_id;
  metadata.command_buffer_id = event->command_buffer_id;
  metadata.physical_device_ordinal = record.physical_device_ordinal;
  metadata.queue_ordinal = record.queue_ordinal;

  iree_const_byte_span_t iovecs[] = {
      iree_make_const_byte_span(&record, sizeof(record)),
      iree_make_const_byte_span(trace_data, data_size),
  };
  return iree_hal_profile_sink_write(context->sink, &metadata,
                                     IREE_ARRAYSIZE(iovecs), iovecs);
}

static hsa_status_t iree_hal_amdgpu_profile_trace_collect_callback(
    uint32_t shader_engine, void* buffer, uint64_t size, void* user_data) {
  iree_hal_amdgpu_profile_trace_collect_context_t* context =
      (iree_hal_amdgpu_profile_trace_collect_context_t*)user_data;
  if (!iree_status_is_ok(context->status)) return HSA_STATUS_ERROR;
  if (IREE_UNLIKELY(size > IREE_HOST_SIZE_MAX)) {
    context->status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "AMDGPU ATT trace byte length %" PRIu64
                         " exceeds host addressable size %" PRIhsz,
                         size, IREE_HOST_SIZE_MAX);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  context->status = iree_hal_amdgpu_profile_trace_write_chunk(
      context, shader_engine, buffer, (iree_host_size_t)size);
  return iree_status_is_ok(context->status) ? HSA_STATUS_SUCCESS
                                            : HSA_STATUS_ERROR;
}

static iree_status_t iree_hal_amdgpu_host_queue_write_profile_trace(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_trace_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id, uint64_t event_position,
    const iree_hal_profile_dispatch_event_t* event) {
  iree_hal_amdgpu_profile_trace_slot_t* slot =
      iree_hal_amdgpu_host_queue_profile_trace_slot(queue, event_position);
  if (IREE_UNLIKELY(!slot->handle.handle || slot->trace_id == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU executable trace slot was not prepared before flush");
  }

  iree_hal_amdgpu_profile_trace_collect_context_t context = {
      .queue = queue,
      .sink = sink,
      .session_id = session_id,
      .event = event,
      .slot = slot,
      .status = iree_ok_status(),
  };
  hsa_status_t hsa_status = session->libaqlprofile.aqlprofile_att_iterate_data(
      slot->handle, iree_hal_amdgpu_profile_trace_collect_callback, &context);
  if (!iree_status_is_ok(context.status)) return context.status;
  return iree_status_from_aqlprofile_status(
      &session->libaqlprofile, __FILE__, __LINE__, hsa_status,
      "aqlprofile_att_iterate_data", "iterating AMDGPU ATT trace data");
}

iree_status_t iree_hal_amdgpu_host_queue_write_profile_traces(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id, uint64_t event_read_position,
    iree_host_size_t event_count,
    const iree_hal_profile_dispatch_event_t* events) {
  if (!sink || !event_count || !queue->profiling.trace_session) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_trace_session_t* session =
      queue->profiling.trace_session;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t event_ordinal = 0;
       event_ordinal < event_count && iree_status_is_ok(status);
       ++event_ordinal) {
    const uint64_t event_position = event_read_position + event_ordinal;
    status = iree_hal_amdgpu_host_queue_write_profile_trace(
        queue, session, sink, session_id, event_position,
        &events[event_ordinal]);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
