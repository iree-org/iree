// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_SYMS_H_
#define IREE_HAL_DRIVERS_VULKAN_SYMS_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

// Internal representation of the public opaque iree_hal_vulkan_syms_t handle.
typedef struct iree_hal_vulkan_syms_t {
  // Reference count for shared ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for the wrapper.
  iree_allocator_t host_allocator;

  // Retained Vulkan loader entry points.
  iree_hal_vulkan_libvulkan_t libvulkan;
} iree_hal_vulkan_syms_t;

#endif  // IREE_HAL_DRIVERS_VULKAN_SYMS_H_
