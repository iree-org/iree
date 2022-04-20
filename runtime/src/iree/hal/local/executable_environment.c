// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_environment.h"

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_processor_*_t
//===----------------------------------------------------------------------===//

void iree_hal_processor_query(iree_allocator_t temp_allocator,
                              iree_hal_processor_v0_t* out_processor) {
  IREE_ASSERT_ARGUMENT(out_processor);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_processor, 0, sizeof(*out_processor));

  // TODO(benvanik): define processor features we want to query for each arch.
  // This needs to be baked into the executable library API and made consistent
  // with the compiler side producing the executables that access it.

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_executable_environment_*_t
//===----------------------------------------------------------------------===//

void iree_hal_executable_environment_initialize(
    iree_allocator_t temp_allocator,
    iree_hal_executable_environment_v0_t* out_environment) {
  IREE_ASSERT_ARGUMENT(out_environment);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_environment, 0, sizeof(*out_environment));
  iree_hal_processor_query(temp_allocator, &out_environment->processor);
  IREE_TRACE_ZONE_END(z0);
}
