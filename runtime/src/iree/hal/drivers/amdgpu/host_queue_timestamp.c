// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"

#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

static iree_hal_amdgpu_pm4_ib_slot_t* iree_hal_amdgpu_host_queue_pm4_ib_slot(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id) {
  return &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
}

void iree_hal_amdgpu_host_queue_commit_timestamp_start(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control, uint64_t* start_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_start(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, start_tick, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_timestamp_end(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* end_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_end(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, completion_signal, end_tick, &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_timestamp_range(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t packet_id,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint64_t* start_tick,
    uint64_t* end_tick) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      iree_hal_amdgpu_host_queue_pm4_ib_slot(queue, packet_id);
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_timestamp_range(
      &packet->pm4_ib, pm4_ib_slot, packet_control,
      queue->pm4_timestamp_strategy, completion_signal, start_tick, end_tick,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}
