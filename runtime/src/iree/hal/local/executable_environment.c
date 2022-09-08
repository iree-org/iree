// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_environment.h"

#include "iree/base/internal/cpu.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_executable_environment_*_t
//===----------------------------------------------------------------------===//

void iree_hal_executable_environment_initialize(
    iree_allocator_t temp_allocator,
    iree_hal_executable_environment_v0_t* out_environment) {
  IREE_ASSERT_ARGUMENT(out_environment);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_environment, 0, sizeof(*out_environment));

  // Force CPU initialization.
  // TODO(benvanik): move this someplace better? Technically not thread-safe
  // but should be enough for usage within the HAL.
  iree_cpu_initialize(temp_allocator);

  // Will fill all of the required fields and zero any extras.
  iree_cpu_read_data(IREE_HAL_PROCESSOR_DATA_CAPACITY_V0,
                     &out_environment->processor.data[0]);

  IREE_TRACE_ZONE_END(z0);
}
