// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_ROCM_HEADERS_H_
#define IREE_HAL_ROCM_ROCM_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error 32-bit not supported on ROCm
#endif  // defined(IREE_PTR_SIZE_32)

#define __HIP_PLATFORM_AMD__
#include "hip/hip_runtime.h"  // IWYU pragma: export

#endif  // IREE_HAL_ROCM_ROCM_HEADERS_H_
