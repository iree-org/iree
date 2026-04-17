// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PM4 packet emission helpers. These build indirect-buffer payloads and the
// vendor AQL PM4-IB packet bodies used to jump to those payloads. They do not
// reserve IB storage, commit AQL packet headers, or ring doorbells; queue code
// owns those publication and lifetime rules.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_EMITTER_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_EMITTER_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/queue.h"
#include "iree/hal/drivers/amdgpu/abi/signal.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queue-private PM4 IB slot. The slot count always matches the AQL ring
// capacity, so AQL packet id N uses pm4_ib_slots[N & aql_ring.mask].
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_pm4_ib_slot_t {
  // Encoded PM4 packet words consumed by a PM4-IB AQL packet.
  uint32_t dwords[16];
} iree_hal_amdgpu_pm4_ib_slot_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_pm4_ib_slot_t) == 64,
                          "PM4 IB slot must be exactly one cache line");

enum {
  IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY = 16,
  // PM4-IB AQL envelopes encode the indirect-buffer dword count in 20 bits.
  IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT = 0xFFFFF,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA = 0x37,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER = 0x3F,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA = 0x40,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_RELEASE_MEM = 0x49,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64 = 0x93,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TIMESTAMP = 9 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 = 2 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM = 5 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 = 2 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS = 1 << 16,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION = 1 << 20,
  IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_TYPE_BOTTOM_OF_PIPE_TS = 40 << 0,
  IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_INDEX_END_OF_PIPE = 5 << 8,
  IREE_HAL_AMDGPU_PM4_RELEASE_MEM_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM = 3 << 24,
  IREE_HAL_AMDGPU_PM4_RELEASE_MEM_DATA_SEL_TIMESTAMP = 3u << 29,
  IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 = 2 << 8,
  IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION = 1 << 20,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM = 0,
};

static const uint32_t
    IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE = 0x80000000u;

static inline uint32_t iree_hal_amdgpu_pm4_make_header(uint32_t opcode,
                                                       uint32_t dword_count) {
  return (3u << 30) | (opcode << 8) | ((dword_count - 2u) << 16);
}

// Bounded builder for one queue-private PM4 IB slot.
typedef struct iree_hal_amdgpu_pm4_ib_builder_t {
  // Queue-owned PM4 IB slot being populated.
  iree_hal_amdgpu_pm4_ib_slot_t* slot;
  // Number of dwords already populated in |slot|.
  uint32_t dword_count;
} iree_hal_amdgpu_pm4_ib_builder_t;

// Clears |slot| and initializes |out_builder| for bounded packet appends.
static inline void iree_hal_amdgpu_pm4_ib_builder_initialize(
    iree_hal_amdgpu_pm4_ib_slot_t* slot,
    iree_hal_amdgpu_pm4_ib_builder_t* out_builder) {
  memset(slot, 0, sizeof(*slot));
  out_builder->slot = slot;
  out_builder->dword_count = 0;
}

// Returns the number of dwords populated by |builder|.
static inline uint32_t iree_hal_amdgpu_pm4_ib_builder_dword_count(
    const iree_hal_amdgpu_pm4_ib_builder_t* builder) {
  return builder->dword_count;
}

// Returns the remaining dword capacity in |builder|.
static inline uint32_t iree_hal_amdgpu_pm4_ib_builder_remaining(
    const iree_hal_amdgpu_pm4_ib_builder_t* builder) {
  return IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY - builder->dword_count;
}

// Appends |dword_count| uninitialized dwords and returns their start, or NULL
// if the requested span does not fit.
static inline uint32_t* iree_hal_amdgpu_pm4_ib_builder_append_dwords(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, uint32_t dword_count) {
  if (dword_count > iree_hal_amdgpu_pm4_ib_builder_remaining(builder)) {
    return NULL;
  }
  uint32_t* dwords = &builder->slot->dwords[builder->dword_count];
  builder->dword_count += dword_count;
  return dwords;
}

// Appends one PM4 packet header and reserves |dword_count| packet dwords.
// Returns the packet start, or NULL if the packet is malformed or does not fit.
static inline uint32_t* iree_hal_amdgpu_pm4_ib_builder_append_packet(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, uint32_t opcode,
    uint32_t dword_count) {
  if (dword_count < 2) return NULL;
  uint32_t* packet =
      iree_hal_amdgpu_pm4_ib_builder_append_dwords(builder, dword_count);
  if (!packet) return NULL;
  packet[0] = iree_hal_amdgpu_pm4_make_header(opcode, dword_count);
  return packet;
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_lo(uintptr_t address) {
  return (uint32_t)(address & 0xFFFFFFFCu);
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_lo_8(uintptr_t address) {
  return (uint32_t)(address & 0xFFFFFFF8u);
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_hi(uintptr_t address) {
  return (uint32_t)(address >> 32);
}

static inline uint32_t iree_hal_amdgpu_pm4_ib_addr_hi(uintptr_t address) {
  return (uint32_t)((address >> 32) & 0xFFFFu);
}

// Appends a COPY_DATA timestamp write to |target|. This is the RADV-style
// top-of-pipe/immediate timestamp form and is intended for profiling records,
// not memory copies. The packet form was probed on gfx1100 AQL compute queues;
// callers must select it only for architectures where the physical-device
// capability table says this form is valid.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, void* target) {
  if (!iree_host_ptr_has_alignment(target, 8)) return false;
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA, 6);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TIMESTAMP |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = 0;
  dword[3] = 0;
  dword[4] = (uint32_t)address;
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(address);
  return true;
}

// Appends a RELEASE_MEM bottom-of-pipe timestamp write to |target|. This uses
// only the common timestamp event/data fields and deliberately avoids cache or
// PWS bits whose layout differs across gfx generations. The packet form was
// probed on gfx1100 AQL compute queues; callers must select it only for
// architectures where the physical-device capability table says this form is
// valid.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_release_mem_timestamp_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, void* target) {
  if (!iree_host_ptr_has_alignment(target, 8)) return false;
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_RELEASE_MEM, 8);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_TYPE_BOTTOM_OF_PIPE_TS |
             IREE_HAL_AMDGPU_PM4_RELEASE_MEM_EVENT_INDEX_END_OF_PIPE;
  dword[2] =
      IREE_HAL_AMDGPU_PM4_RELEASE_MEM_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM |
      IREE_HAL_AMDGPU_PM4_RELEASE_MEM_DATA_SEL_TIMESTAMP;
  dword[3] = (uint32_t)address;
  dword[4] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[5] = 0;
  dword[6] = 0;
  dword[7] = 0;
  return true;
}

static inline uint32_t iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
    uint32_t function, uint32_t mem_space, uint32_t operation) {
  return (function & 0x7u) | ((mem_space & 0x3u) << 4) |
         ((operation & 0x3u) << 6);
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_wait_reg_mem64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, iree_hsa_signal_t epoch_signal,
    iree_hsa_signal_value_t compare_value, iree_hsa_signal_value_t mask) {
  memset(slot, 0, sizeof(*slot));
  iree_amd_signal_t* signal_abi = (iree_amd_signal_t*)epoch_signal.handle;
  volatile iree_hsa_signal_value_t* value_address = &signal_abi->value;
  const uintptr_t address = (uintptr_t)value_address;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64, 9);
  dword[1] = iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN,
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM);
  dword[2] = iree_hal_amdgpu_pm4_addr_lo_8(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = (uint32_t)compare_value;
  dword[5] = (uint32_t)((uint64_t)compare_value >> 32);
  dword[6] = (uint32_t)mask;
  dword[7] = (uint32_t)((uint64_t)mask >> 32);
  dword[8] = 4 | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE;
  return 9;
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_write_data32(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, void* target, uint32_t value) {
  memset(slot, 0, sizeof(*slot));
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, 5);
  dword[1] = IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = value;
  return 5;
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_write_data64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, void* target, uint64_t value) {
  memset(slot, 0, sizeof(*slot));
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, 6);
  dword[1] = IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = (uint32_t)value;
  dword[5] = (uint32_t)(value >> 32);
  return 6;
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_copy_data32(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, const void* source, void* target) {
  memset(slot, 0, sizeof(*slot));
  const uintptr_t source_address = (uintptr_t)source;
  const uintptr_t target_address = (uintptr_t)target;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA, 6);
  dword[1] = IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(source_address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(source_address);
  dword[4] = iree_hal_amdgpu_pm4_addr_lo(target_address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  return 6;
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_copy_data64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, const void* source, void* target) {
  memset(slot, 0, sizeof(*slot));
  const uintptr_t source_address = (uintptr_t)source;
  const uintptr_t target_address = (uintptr_t)target;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA, 6);
  dword[1] = IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo_8(source_address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(source_address);
  dword[4] = iree_hal_amdgpu_pm4_addr_lo_8(target_address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  return 6;
}

// Emits an AQL PM4-IB envelope referencing |ib_dwords|. The referenced dword
// storage must remain immutable and live until the AQL packet retires.
static inline uint16_t iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet, const uint32_t* ib_dwords,
    uint32_t ib_dword_count,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  IREE_ASSERT(ib_dword_count <= IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  const uintptr_t ib_address = (uintptr_t)ib_dwords;
  packet->ib_jump_cmd[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, 4);
  packet->ib_jump_cmd[1] = iree_hal_amdgpu_pm4_addr_lo(ib_address);
  packet->ib_jump_cmd[2] = iree_hal_amdgpu_pm4_ib_addr_hi(ib_address);
  packet->ib_jump_cmd[3] =
      (ib_dword_count & IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT) | (1u << 23);
  packet->dw_cnt_remain = 0xA;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packet->reserved); ++i) {
    packet->reserved[i] = 0;
  }
  packet->completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_PM4_IB;
  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
                                         packet_control);
}

static inline uint16_t iree_hal_amdgpu_aql_emit_pm4_ib(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    const iree_hal_amdgpu_pm4_ib_slot_t* ib_slot, uint32_t ib_dword_count,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  return iree_hal_amdgpu_aql_emit_pm4_ib_dwords(packet, ib_slot->dwords,
                                                ib_dword_count, packet_control,
                                                completion_signal, out_setup);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_EMITTER_H_
