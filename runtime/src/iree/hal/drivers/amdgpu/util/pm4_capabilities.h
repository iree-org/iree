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

// AMD vendor-packet and PM4 packet-family capabilities available on a physical
// device.
enum iree_hal_amdgpu_vendor_packet_capability_bits_t {
  // AMD vendor AQL PM4-IB packets can jump to device-visible PM4 programs.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB = 1u << 0,
  // AMD vendor AQL BARRIER_VALUE packets can wait on arbitrary signal values.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_BARRIER_VALUE = 1u << 1,
  // PM4 WAIT_REG_MEM64 packets can perform 64-bit memory comparisons.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_WAIT_REG_MEM64 = 1u << 2,
  // PM4 COPY_DATA can copy the immediate timestamp counter to memory.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_TIMESTAMP = 1u << 3,
  // PM4 RELEASE_MEM can write a bottom-of-pipe timestamp to memory.
  IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_RELEASE_MEM_TIMESTAMP = 1u << 4,
};
typedef uint32_t iree_hal_amdgpu_vendor_packet_capability_flags_t;

// Returns true if the device can emit queue-private PM4 timestamp ranges using
// a COPY_DATA start timestamp and RELEASE_MEM end timestamp.
static inline bool
iree_hal_amdgpu_vendor_packet_capabilities_support_timestamp_range(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities) {
  return iree_all_bits_set(
      capabilities,
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_AQL_PM4_IB |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_COPY_TIMESTAMP |
          IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_RELEASE_MEM_TIMESTAMP);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_PM4_CAPABILITIES_H_
