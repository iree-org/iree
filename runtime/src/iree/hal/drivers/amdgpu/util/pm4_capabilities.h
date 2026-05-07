// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_CAPABILITIES_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_CAPABILITIES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Hardware mechanism used for cross-queue epoch waits after wait resolution
// proves that a dependency has already been submitted by a local peer queue.
typedef enum iree_hal_amdgpu_wait_barrier_strategy_e {
  // No device-side 64-bit epoch wait is known for this agent; unresolved
  // cross-queue waits must use software deferral.
  IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER = 0,
  // AMD vendor AQL BARRIER_VALUE packet.
  IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE = 1,
  // AMD vendor AQL PM4-IB packet executing a WAIT_REG_MEM64 PM4 packet.
  IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64 = 2,
} iree_hal_amdgpu_wait_barrier_strategy_t;

// PM4 timestamp packet sequence used to bracket queue-device profiling ranges.
typedef enum iree_hal_amdgpu_pm4_timestamp_strategy_e {
  // No queue-local PM4 timestamp sequence is selected.
  IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE = 0,
  // COPY_DATA reads the GPU clock into memory using the gfx9/aqlprofile
  // MEMORY destination with STREAM cache policy.
  IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM = 1,
  // COPY_DATA reads the GPU clock into memory using the gfx10/gfx11
  // aqlprofile TC_L2 destination with LRU cache policy.
  IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU = 2,
  // COPY_DATA reads the GPU clock into memory using the gfx12/aqlprofile TC_L2
  // destination with last-use temporal policy.
  IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU = 3,
} iree_hal_amdgpu_pm4_timestamp_strategy_t;

// Returns true if |strategy| can emit complete queue-device timestamp ranges.
static inline bool iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
    iree_hal_amdgpu_pm4_timestamp_strategy_t strategy) {
  switch (strategy) {
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_MEMORY_STREAM:
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LRU:
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_COPY_CLOCK_TC_L2_LU:
      return true;
    case IREE_HAL_AMDGPU_PM4_TIMESTAMP_STRATEGY_NONE:
    default:
      return false;
  }
}

// AMD vendor-packet and PM4 packet-family capabilities available on a physical
// device.
enum iree_hal_amdgpu_vendor_packet_capability_bits_t {
  // AMD vendor AQL PM4-IB packets can jump to device-visible PM4 programs.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB = 1u << 0,
  // AMD vendor AQL BARRIER_VALUE packets can wait on arbitrary signal values.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE = 1u << 1,
  // PM4 WAIT_REG_MEM64 packets can perform 64-bit memory comparisons.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 = 1u << 2,
  // PM4 EVENT_WRITE can emit compute-pipeline events.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE = 1u << 3,
  // PM4 SET_SH_REG can program persistent shader registers.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG = 1u << 4,
  // PM4 SET_UCONFIG_REG can program user configuration registers.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG = 1u << 5,
  // PM4 COPY_DATA can read register values into memory.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK = 1u << 6,
  // PM4 COPY_DATA can read performance-counter values into memory.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK = 1u << 7,
  // PM4 COPY_DATA can write immediate values into registers/perfcounters.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE = 1u << 8,
  // PM4 WRITE_DATA can write immediate values into memory through TC_L2.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY = 1u << 9,
  // PM4 COPY_DATA can copy memory through TC_L2 into memory through TC_L2.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY = 1u << 10,
  // PM4 ACQUIRE_MEM can perform explicit cache-management operations.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM = 1u << 11,
  // PM4 DISPATCH_DIRECT can launch compute work from a PM4 command stream.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT = 1u
                                                                         << 12,
  // PM4 DISPATCH_INDIRECT can launch indirect compute work from a PM4 command
  // stream.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_INDIRECT =
      1u << 13,
  // PM4 ACQUIRE_MEM uses the gfx9 CP_COHER_CNTL packet layout.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9 = 1u << 14,
  // PM4 ACQUIRE_MEM uses the gfx10+ GCR_CNTL packet layout.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10 = 1u << 15,
};
typedef uint32_t iree_hal_amdgpu_vendor_packet_capability_flags_t;

// Returns true if the device can emit queue-private PM4 WRITE_DATA memory
// writes.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_write_data(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_WRITE_DATA_MEMORY);
}

// Returns true if the device can emit queue-private PM4 COPY_DATA memory
// copies.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_copy_data(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_DATA_MEMORY);
}

// Returns true if the device can emit PM4 compute DISPATCH_DIRECT streams.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_compute_dispatch_direct(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT);
}

// Returns true if the device can emit PM4 compute DISPATCH_INDIRECT streams.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_compute_dispatch_indirect(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_INDIRECT);
}

// Returns true if the device can run the initial dispatch-only PM4
// command-buffer path with conservative in-stream barriers.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_dispatch_command_buffers(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  if (!iree_all_bits_set(
          capabilities,
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM |
              IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COMPUTE_DISPATCH_DIRECT)) {
    return false;
  }
  const iree_hal_amdgpu_vendor_packet_capability_flags_t acquire_mem_layouts =
      capabilities &
      (IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9 |
       IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10);
  return acquire_mem_layouts ==
             IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9 ||
         acquire_mem_layouts ==
             IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10;
}

// Returns true if the device can emit the gfx10+ packet families needed for
// queue-local PMC start/read/stop programs.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_gfx10_pmc_programs(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_SH_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_SET_UCONFIG_REG |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_REGISTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_PERFCOUNTER_READBACK |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_IMMEDIATE_WRITE);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_CAPABILITIES_H_
