// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_AFFINITY_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_AFFINITY_H_

#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_affinity_t
//===----------------------------------------------------------------------===//

// A bitmap indicating physical device affinity.
// Each bit represents a physical device in the topology with an ordinal
// corresponding to the bit index. For example 0b10 indicates physical device 1
// in the system and 0b110 indicates physical devices 1 and 2.
//
// Currently represents a maximum of 64 devices within the topology (not in the
// machine) as that makes life easy. Could be extended to support more ala
// CPU_SET but >64 unique physical devices in one machine managed by a single
// HAL logical device is not a reasonable use of the system.
typedef uint64_t iree_hal_amdgpu_device_affinity_t;

#define iree_hal_amdgpu_device_affinity_count(affinity) \
  iree_math_count_ones_u64(affinity)
#define iree_hal_amdgpu_device_affinity_find_first_set(affinity) \
  iree_math_count_trailing_zeros_u64(affinity)
#define iree_hal_amdgpu_device_affinity_shr(affinity, amount) \
  iree_shr((affinity), (amount))
#define iree_hal_amdgpu_device_affinity_and_into(inout_affinity, \
                                                 mask_affinity)  \
  (inout_affinity) = ((inout_affinity) & (mask_affinity))
#define iree_hal_amdgpu_device_affinity_or_into(inout_affinity, mask_affinity) \
  (inout_affinity) = ((inout_affinity) | (mask_affinity))

// Loops over each physical device in the given |device_affinity| bitmap.
//
// The following variables are available within the loop:
//     device_count: total number of devices used
//     device_index: loop index (0 to device_count)
//   device_ordinal: physical device ordinal in the topology
//
// Example:
//  IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(my_device_affinity) {
//    compact_device_list[device_index];     // 0 to my_device_affinity count
//    physical_device_list[device_ordinal];  // 0 to available devices
//  }
#define IREE_HAL_AMDGPU_FOR_PHYSICAL_DEVICE(device_affinity)                   \
  iree_hal_amdgpu_device_affinity_t _device_bits = (device_affinity);          \
  for (int device_index = 0, _device_ordinal_base = 0,                         \
           device_count = iree_hal_amdgpu_device_affinity_count(_device_bits), \
           _bit_offset = 0,                                                    \
           device_ordinal =                                                    \
               iree_hal_amdgpu_device_affinity_find_first_set(_device_bits);   \
       device_index < device_count;                                            \
       ++device_index, _device_ordinal_base += _bit_offset + 1,                \
           _device_bits = iree_hal_amdgpu_device_affinity_shr(                 \
               _device_bits, _bit_offset + 1),                                 \
           _bit_offset =                                                       \
               iree_hal_amdgpu_device_affinity_find_first_set(_device_bits),   \
           device_ordinal = _device_ordinal_base + _bit_offset)

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_AFFINITY_H_
