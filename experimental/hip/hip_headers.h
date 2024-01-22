// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_HIP_HEADERS_H_
#define IREE_EXPERIMENTAL_HIP_HIP_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error "32-bit not supported on HIP backend"
#endif  // defined(IREE_PTR_SIZE_32)

#define __HIP_PLATFORM_AMD__
#include "hip/hip_runtime_api.h"  // IWYU pragma: export

#endif  // IREE_EXPERIMENTAL_HIP_HIP_HEADERS_H_
