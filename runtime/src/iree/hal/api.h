// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See iree/base/api.h for documentation on the API conventions used.

#ifndef IREE_HAL_API_H_
#define IREE_HAL_API_H_

#include "iree/hal/allocator.h"         // IWYU pragma: export
#include "iree/hal/buffer.h"            // IWYU pragma: export
#include "iree/hal/buffer_view.h"       // IWYU pragma: export
#include "iree/hal/buffer_view_util.h"  // IWYU pragma: export
#include "iree/hal/channel.h"           // IWYU pragma: export
#include "iree/hal/channel_provider.h"  // IWYU pragma: export
#include "iree/hal/command_buffer.h"    // IWYU pragma: export
#include "iree/hal/device.h"            // IWYU pragma: export
#include "iree/hal/driver.h"            // IWYU pragma: export
#include "iree/hal/driver_registry.h"   // IWYU pragma: export
#include "iree/hal/event.h"             // IWYU pragma: export
#include "iree/hal/executable.h"        // IWYU pragma: export
#include "iree/hal/executable_cache.h"  // IWYU pragma: export
#include "iree/hal/fence.h"             // IWYU pragma: export
#include "iree/hal/pipeline_layout.h"   // IWYU pragma: export
#include "iree/hal/resource.h"          // IWYU pragma: export
#include "iree/hal/semaphore.h"         // IWYU pragma: export
#include "iree/hal/string_util.h"       // IWYU pragma: export

#endif  // IREE_HAL_API_H_
