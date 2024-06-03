// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_HIP_HEADERS_H_
#define IREE_HAL_DRIVERS_HIP_HIP_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error "32-bit not supported on HIP backend"
#endif  // defined(IREE_PTR_SIZE_32)

#define __HIP_PLATFORM_AMD__
// Order matters here--hip_deprecated.h depends on hip_runtime_api.h. So turn
// off clang-format.
//
// We need to pull in this hip_deprecated.h for the old hipDeviceProp_t struct
// definition, hipDeviceProp_tR0000. HIP 6.0 release changes the struct in the
// middle. The hipDeviceProp_t struct would need to use the matching
// hipGetDevicePropertiesR0600() API to query it. We want to also support HIP
// 5.x versions so use the old hipGetDeviceProperties() API with its matching
// struct.

// clang-format off
#include "hip/hip_runtime_api.h"  // IWYU pragma: export
#include "hip/hip_deprecated.h"   // IWYU pragma: export
// clang-format on

#endif  // IREE_HAL_DRIVERS_HIP_HIP_HEADERS_H_
