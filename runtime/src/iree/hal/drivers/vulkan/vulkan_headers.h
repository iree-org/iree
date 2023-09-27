// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_VULKAN_HEADERS_H_
#define IREE_HAL_DRIVERS_VULKAN_VULKAN_HEADERS_H_

#include "iree/base/api.h"

// We exclusively use Vulkan via queried function pointers. To ensure that there
// are no accidental calls to the linker-loaded implicit functions we just
// compile them all out.
//
// Code under iree/hal/drivers/vulkan/ *MUST NOT* directly include vulkan.h or
// any header that includes it without this first being set. This means that
// this iree/hal/drivers/vulkan/vulkan_headers.h file must usually be included
// first in all files using it.
//
// From there, use iree/hal/drivers/vulkan/dynamic_symbols.h to plumb the
// dynamically resolved symbols to any code that may need to make Vulkan calls.
// See that header for more information: in general we try to keep our required
// set of symbols minimal to avoid binary size/runtime memory/linker time so
// symbols are only added as needed.
//
// Other non-core code can choose not to disable the prototypes if they want.
// I don't suggest it though for anything beyond samples.
//
// There's a bunch of reasons to dynamically link against Vulkan like supporting
// platforms without Vulkan or with differing Vulkan versions where all symbols
// may not be available.
//
// See this article for more information:
// https://djang86.blogspot.com/2019/01/what-is-vknoprototypes.html
#define VK_NO_PROTOTYPES 1

#if defined(IREE_PLATFORM_ANDROID)
#define VK_USE_PLATFORM_ANDROID_KHR 1
#elif defined(IREE_PLATFORM_WINDOWS)
#define VK_USE_PLATFORM_WIN32_KHR 1
#endif  // IREE_PLATFORM_*

#include "vulkan/vulkan.h"  // IWYU pragma: export

#ifdef IREE_PLATFORM_APPLE
#include "vulkan/vulkan_beta.h"  // IWYU pragma: export
#endif

#endif  // IREE_HAL_DRIVERS_VULKAN_VULKAN_HEADERS_H_
