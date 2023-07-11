// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_VMA_IMPL_H_
#define IREE_HAL_DRIVERS_VULKAN_VMA_IMPL_H_

#include "iree/hal/drivers/vulkan/vulkan_headers.h"

// Force all Vulkan calls to go through an indirect pVulkanFunctions interface.
// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/configuration.html
#define VMA_STATIC_VULKAN_FUNCTIONS 0

// Prevent VMA from querying for dynamic functions we may not have provided.
// We want to be able to print nice errors or decide whether something is ok
// to be omitted and not have VMA poking around where it shouldn't.
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <vk_mem_alloc.h>  // IWYU pragma: export

#endif  // IREE_HAL_DRIVERS_VULKAN_VMA_IMPL_H_
