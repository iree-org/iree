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
#include "iree/hal/drivers/amdgpu/util/pm4_capabilities.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Queue-private PM4 IB slot. The slot count always matches the AQL ring
// capacity, so AQL packet id N uses pm4_ib_slots[N & aql_ring.mask].
typedef struct IREE_AMDGPU_ALIGNAS(64) iree_hal_amdgpu_pm4_ib_slot_t {
  // Encoded PM4 packet words consumed by a PM4-IB AQL packet.
  uint32_t dwords[32];
} iree_hal_amdgpu_pm4_ib_slot_t;
IREE_AMDGPU_STATIC_ASSERT(sizeof(iree_hal_amdgpu_pm4_ib_slot_t) == 128,
                          "PM4 IB slot must be exactly two cache lines");

enum {
  IREE_HAL_AMDGPU_PM4_IB_SLOT_DWORD_CAPACITY = 32,
  // PM4-IB AQL envelopes encode the indirect-buffer dword count in 20 bits.
  IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT = 0xFFFFF,
  IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT = 6,
  IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT =
      2 * IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT,
  IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT = 2,
  IREE_HAL_AMDGPU_PM4_SET_REGISTER_DWORD_COUNT = 3,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DWORD_COUNT = 6,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA = 0x37,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER = 0x3F,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA = 0x40,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE = 0x46,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG = 0x76,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_UCONFIG_REG = 0x79,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64 = 0x93,
  IREE_HAL_AMDGPU_PM4_REGISTER_OFFSET_MASK = 0x3FFFF,
  IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START = 0x00002C00,
  IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_END = 0x00002FFF,
  IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_START = 0x0000C000,
  IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_END = 0x0000FFFF,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT = 9 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 = 2 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_MEM_MAPPED_REGISTER = 0 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_PERFCOUNTER = 4 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_IMMEDIATE_DATA = 5 << 0,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM_MAPPED_REGISTER = 0 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM = 5 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 = 2 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_PERFCOUNTER = 4 << 8,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM = 1 << 13,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS = 1 << 16,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION = 1 << 20,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_TEMPORAL_LU = 3 << 13,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM = 1 << 25,
  IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_TEMPORAL_LU = 3 << 25,
  IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH = 7 << 0,
  IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH = 4 << 8,
  IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 = 2 << 8,
  IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION = 1 << 20,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM = 0,
};

static const uint32_t
    IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE = 0x80000000u;

typedef enum iree_hal_amdgpu_pm4_register_space_e {
  IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_MEM_MAPPED_REGISTER = 0,
  IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_PERFCOUNTER = 4,
} iree_hal_amdgpu_pm4_register_space_t;

typedef enum iree_hal_amdgpu_pm4_write_confirmation_e {
  IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_NONE = 0,
  IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_WAIT = 1,
} iree_hal_amdgpu_pm4_write_confirmation_t;

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

// Returns the PM4 IB dword count required by one timestamp range emitted with
// |strategy|, or 0 when the strategy has no range packet sequence.
static inline uint32_t iree_hal_amdgpu_pm4_timestamp_range_dword_count(
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM:
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU:
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU:
      return IREE_HAL_AMDGPU_PM4_TIMESTAMP_RANGE_DWORD_COUNT;
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE:
    default:
      return 0;
  }
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

static inline bool iree_hal_amdgpu_pm4_register_space_is_valid(
    iree_hal_amdgpu_pm4_register_space_t register_space) {
  return register_space ==
             IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_MEM_MAPPED_REGISTER ||
         register_space == IREE_HAL_AMDGPU_PM4_REGISTER_SPACE_PERFCOUNTER;
}

static inline bool iree_hal_amdgpu_pm4_write_confirmation_is_valid(
    iree_hal_amdgpu_pm4_write_confirmation_t write_confirmation) {
  return write_confirmation == IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_NONE ||
         write_confirmation == IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_WAIT;
}

static inline uint32_t iree_hal_amdgpu_pm4_copy_data_source_register_space(
    iree_hal_amdgpu_pm4_register_space_t register_space) {
  return (uint32_t)register_space << 0;
}

static inline uint32_t iree_hal_amdgpu_pm4_copy_data_target_register_space(
    iree_hal_amdgpu_pm4_register_space_t register_space) {
  return (uint32_t)register_space << 8;
}

static inline uint32_t iree_hal_amdgpu_pm4_copy_data_write_confirmation(
    iree_hal_amdgpu_pm4_write_confirmation_t write_confirmation) {
  return write_confirmation == IREE_HAL_AMDGPU_PM4_WRITE_CONFIRMATION_WAIT
             ? IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION
             : 0;
}

// Appends an EVENT_WRITE CS_PARTIAL_FLUSH packet. This is the queue-local
// wait-idle building block around counter programming; stronger
// cache-management packets should remain separate helpers.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_event_write_cs_partial_flush(
    iree_hal_amdgpu_pm4_ib_builder_t* builder) {
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_EVENT_WRITE,
      IREE_HAL_AMDGPU_PM4_EVENT_WRITE_DWORD_COUNT);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_TYPE_CS_PARTIAL_FLUSH |
             IREE_HAL_AMDGPU_PM4_EVENT_WRITE_EVENT_INDEX_CS_PARTIAL_FLUSH;
  return true;
}

// Appends a SET_SH_REG packet for persistent shader registers.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_set_sh_reg(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, uint32_t register_address,
    uint32_t value) {
  if (register_address < IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START ||
      register_address > IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_END) {
    return false;
  }
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_SH_REG,
      IREE_HAL_AMDGPU_PM4_SET_REGISTER_DWORD_COUNT);
  if (!dword) return false;
  dword[1] = register_address - IREE_HAL_AMDGPU_PM4_PERSISTENT_SPACE_START;
  dword[2] = value;
  return true;
}

// Appends a SET_UCONFIG_REG packet for user configuration registers.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_set_uconfig_reg(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, uint32_t register_address,
    uint32_t value) {
  if (register_address < IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_START ||
      register_address > IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_END) {
    return false;
  }
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_SET_UCONFIG_REG,
      IREE_HAL_AMDGPU_PM4_SET_REGISTER_DWORD_COUNT);
  if (!dword) return false;
  dword[1] = register_address - IREE_HAL_AMDGPU_PM4_UCONFIG_SPACE_START;
  dword[2] = value;
  return true;
}

// Appends a COPY_DATA packet that writes an immediate 32-bit value into a
// memory-mapped register or perfcounter register address.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_copy_immediate32_to_register(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_register_space_t register_space,
    uint32_t register_address, uint32_t value,
    iree_hal_amdgpu_pm4_write_confirmation_t write_confirmation) {
  if (!iree_hal_amdgpu_pm4_register_space_is_valid(register_space) ||
      !iree_hal_amdgpu_pm4_write_confirmation_is_valid(write_confirmation) ||
      register_address > IREE_HAL_AMDGPU_PM4_REGISTER_OFFSET_MASK) {
    return false;
  }
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      IREE_HAL_AMDGPU_PM4_COPY_DATA_DWORD_COUNT);
  if (!dword) return false;
  dword[1] =
      IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_IMMEDIATE_DATA |
      iree_hal_amdgpu_pm4_copy_data_target_register_space(register_space) |
      iree_hal_amdgpu_pm4_copy_data_write_confirmation(write_confirmation);
  dword[2] = value;
  dword[3] = 0;
  dword[4] = register_address;
  dword[5] = 0;
  return true;
}

// Appends a COPY_DATA packet that copies a 32-bit register or perfcounter value
// into memory.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_copy_register32_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_register_space_t register_space,
    uint32_t register_address, void* target,
    iree_hal_amdgpu_pm4_write_confirmation_t write_confirmation) {
  if (!iree_hal_amdgpu_pm4_register_space_is_valid(register_space) ||
      !iree_hal_amdgpu_pm4_write_confirmation_is_valid(write_confirmation) ||
      register_address > IREE_HAL_AMDGPU_PM4_REGISTER_OFFSET_MASK ||
      !iree_host_ptr_has_alignment(target, 4)) {
    return false;
  }
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      IREE_HAL_AMDGPU_PM4_COPY_DATA_DWORD_COUNT);
  if (!dword) return false;
  dword[1] =
      iree_hal_amdgpu_pm4_copy_data_source_register_space(register_space) |
      IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
      iree_hal_amdgpu_pm4_copy_data_write_confirmation(write_confirmation);
  dword[2] = register_address;
  dword[3] = 0;
  dword[4] = iree_hal_amdgpu_pm4_addr_lo(address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(address);
  return true;
}

// Returns the COPY_DATA control dword for the GPU-clock write selected by
// |strategy|, or 0 when the strategy has no packet encoding.
static inline uint32_t iree_hal_amdgpu_pm4_copy_timestamp_control(
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy) {
  // Timestamp records are consumed after the enclosing AQL packet completes, so
  // per-COPY_DATA write confirmation would only add queue-local readback cost.
  const uint32_t common =
      IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_GPU_CLOCK_COUNT |
      IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS;
  switch (strategy) {
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM:
      return common | IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_CACHE_POLICY_STREAM |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_MEM |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_CACHE_POLICY_STREAM;
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU:
      return common | IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2;
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU:
      return common | IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_TEMPORAL_LU |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_TEMPORAL_LU;
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE:
    default:
      return 0;
  }
}

// Appends a COPY_DATA GPU-clock write to |target|. This is intended for
// profiling records, not memory copies.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* target) {
  const uint32_t control = iree_hal_amdgpu_pm4_copy_timestamp_control(strategy);
  if (control == 0 || !iree_host_ptr_has_alignment(target, 8)) return false;
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA,
      IREE_HAL_AMDGPU_PM4_COPY_TIMESTAMP_DWORD_COUNT);
  if (!dword) return false;
  dword[1] = control;
  dword[2] = 0;
  dword[3] = 0;
  dword[4] = iree_hal_amdgpu_pm4_addr_lo_8(address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(address);
  return true;
}

// Appends the COPY_DATA end-timestamp form selected by |strategy|.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* target) {
  if (!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(strategy) ||
      !iree_host_ptr_has_alignment(target, 8)) {
    return false;
  }
  return iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      builder, strategy, target);
}

// Appends the COPY_DATA start-timestamp form selected by |strategy|.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* target) {
  if (!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(strategy)) {
    return false;
  }
  return iree_hal_amdgpu_pm4_ib_builder_emit_copy_timestamp_to_memory(
      builder, strategy, target);
}

// Appends a timestamp range around subsequent queue work. The helper preflights
// all space, strategy, and alignment requirements so it either appends the
// complete range marker or leaves |builder| unchanged.
static inline bool
iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
    iree_hal_amdgpu_pm4_ib_builder_t* builder,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, void* start_target,
    void* end_target) {
  const uint32_t dword_count =
      iree_hal_amdgpu_pm4_timestamp_range_dword_count(strategy);
  if (dword_count == 0 || !iree_host_ptr_has_alignment(start_target, 8) ||
      !iree_host_ptr_has_alignment(end_target, 8)) {
    return false;
  }
  if (iree_hal_amdgpu_pm4_ib_builder_remaining(builder) < dword_count) {
    return false;
  }
  iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
      builder, strategy, start_target);
  iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(builder, strategy,
                                                              end_target);
  return true;
}

static inline uint32_t iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
    uint32_t function, uint32_t mem_space, uint32_t operation) {
  return (function & 0x7u) | ((mem_space & 0x3u) << 4) |
         ((operation & 0x3u) << 6);
}

// Appends a WRITE_DATA packet that writes an immediate 32-bit value into
// memory visible through TC L2.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_write_data32(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, void* target, uint32_t value) {
  if (!iree_host_ptr_has_alignment(target, 4)) return false;
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, 5);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = value;
  return true;
}

// Appends a WRITE_DATA packet that writes an immediate 64-bit value into
// memory visible through TC L2.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_write_data64(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, void* target, uint64_t value) {
  if (!iree_host_ptr_has_alignment(target, 4)) return false;
  const uintptr_t address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, 6);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = (uint32_t)value;
  dword[5] = (uint32_t)(value >> 32);
  return true;
}

// Appends a COPY_DATA packet that copies a 32-bit memory value through TC L2.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_copy_data32(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, const void* source,
    void* target) {
  if (!iree_host_ptr_has_alignment(source, 4) ||
      !iree_host_ptr_has_alignment(target, 4)) {
    return false;
  }
  const uintptr_t source_address = (uintptr_t)source;
  const uintptr_t target_address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA, 6);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo(source_address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(source_address);
  dword[4] = iree_hal_amdgpu_pm4_addr_lo(target_address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  return true;
}

// Appends a COPY_DATA packet that copies a 64-bit memory value through TC L2.
static inline bool iree_hal_amdgpu_pm4_ib_builder_emit_copy_data64(
    iree_hal_amdgpu_pm4_ib_builder_t* builder, const void* source,
    void* target) {
  if (!iree_host_ptr_has_alignment(source, 8) ||
      !iree_host_ptr_has_alignment(target, 8)) {
    return false;
  }
  const uintptr_t source_address = (uintptr_t)source;
  const uintptr_t target_address = (uintptr_t)target;
  uint32_t* dword = iree_hal_amdgpu_pm4_ib_builder_append_packet(
      builder, IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_COPY_DATA, 6);
  if (!dword) return false;
  dword[1] = IREE_HAL_AMDGPU_PM4_COPY_DATA_SRC_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_DST_SEL_TC_L2 |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_COUNT_SEL_64_BITS |
             IREE_HAL_AMDGPU_PM4_COPY_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  dword[2] = iree_hal_amdgpu_pm4_addr_lo_8(source_address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(source_address);
  dword[4] = iree_hal_amdgpu_pm4_addr_lo_8(target_address);
  dword[5] = iree_hal_amdgpu_pm4_addr_hi(target_address);
  return true;
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
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_write_data32(&builder, target, value);
  IREE_ASSERT(did_emit, "PM4 WRITE_DATA32 must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder);
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_write_data64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, void* target, uint64_t value) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_write_data64(&builder, target, value);
  IREE_ASSERT(did_emit, "PM4 WRITE_DATA64 must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder);
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_copy_data32(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, const void* source, void* target) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_copy_data32(&builder, source, target);
  IREE_ASSERT(did_emit, "PM4 COPY_DATA32 must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder);
}

static inline uint32_t iree_hal_amdgpu_pm4_emit_copy_data64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, const void* source, void* target) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_copy_data64(&builder, source, target);
  IREE_ASSERT(did_emit, "PM4 COPY_DATA64 must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder);
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
  packet->ib_jump_cmd[3] = ib_dword_count | (1u << 23);
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

// Emits an AQL PM4-IB packet that writes a start timestamp to |start_tick|. The
// caller owns packet publication.
static inline uint16_t iree_hal_amdgpu_aql_emit_timestamp_start(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    iree_hal_amdgpu_pm4_ib_slot_t* ib_slot,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy, uint64_t* start_tick,
    uint16_t* out_setup) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
          &builder, strategy, start_tick);
  IREE_ASSERT(did_emit, "PM4 start timestamp must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_aql_emit_pm4_ib(
      packet, ib_slot, iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
      packet_control, iree_hsa_signal_null(), out_setup);
}

// Emits an AQL PM4-IB packet that writes an end timestamp to |end_tick|. The
// caller owns packet publication.
static inline uint16_t iree_hal_amdgpu_aql_emit_timestamp_end(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    iree_hal_amdgpu_pm4_ib_slot_t* ib_slot,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy,
    iree_hsa_signal_t completion_signal, uint64_t* end_tick,
    uint16_t* out_setup) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
          &builder, strategy, end_tick);
  IREE_ASSERT(did_emit, "PM4 end timestamp must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_aql_emit_pm4_ib(
      packet, ib_slot, iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
      packet_control, completion_signal, out_setup);
}

// Emits one AQL PM4-IB packet that writes both start and end timestamp fields.
// The caller owns packet publication.
static inline uint16_t iree_hal_amdgpu_aql_emit_timestamp_range(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    iree_hal_amdgpu_pm4_ib_slot_t* ib_slot,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy,
    iree_hsa_signal_t completion_signal, uint64_t* start_tick,
    uint64_t* end_tick, uint16_t* out_setup) {
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
          &builder, strategy, start_tick, end_tick);
  IREE_ASSERT(did_emit, "PM4 timestamp range must fit PM4 IB slot");
  (void)did_emit;
  return iree_hal_amdgpu_aql_emit_pm4_ib(
      packet, ib_slot, iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
      packet_control, completion_signal, out_setup);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_EMITTER_H_
