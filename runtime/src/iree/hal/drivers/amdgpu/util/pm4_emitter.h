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

#include "iree/base/alignment.h"
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
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA = 0x37,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER = 0x3F,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64 = 0x93,
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

static inline uint16_t iree_hal_amdgpu_aql_emit_pm4_ib(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    const iree_hal_amdgpu_pm4_ib_slot_t* ib_slot, uint32_t ib_dword_count,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  const uintptr_t ib_address = (uintptr_t)ib_slot->dwords;
  packet->ib_jump_cmd[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, 4);
  packet->ib_jump_cmd[1] = iree_hal_amdgpu_pm4_addr_lo(ib_address);
  packet->ib_jump_cmd[2] = iree_hal_amdgpu_pm4_ib_addr_hi(ib_address);
  packet->ib_jump_cmd[3] = (ib_dword_count & 0xFFFFFu) | (1u << 23);
  packet->dw_cnt_remain = 0xA;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packet->reserved); ++i) {
    packet->reserved[i] = 0;
  }
  packet->completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_PM4_IB;
  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
                                         packet_control);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_EMITTER_H_
