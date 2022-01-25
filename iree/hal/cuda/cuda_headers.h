// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CUDA_CUDA_HEADERS_H_
#define IREE_HAL_CUDA_CUDA_HEADERS_H_

#include "cuda.h"  // IWYU pragma: export

#if IREE_ENABLE_CUPTI  // CUPTI is only used for tracing.

/* Activity, callback, event and metric APIs */
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_events.h>
#include <cupti_metrics.h>

/* Runtime, driver, and nvtx function identifiers */
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>

#endif  // IREE_TRACING_FEATURES

#endif  // IREE_HAL_CUDA_CUDA_HEADERS_H_
