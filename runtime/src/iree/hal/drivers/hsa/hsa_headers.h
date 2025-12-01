// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_HSA_HEADERS_H_
#define IREE_HAL_DRIVERS_HSA_HSA_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error "32-bit not supported on HSA backend"
#endif  // defined(IREE_PTR_SIZE_32)

// HSA runtime headers
#include "hsa/hsa.h"                  // IWYU pragma: export
#include "hsa/hsa_ext_amd.h"          // IWYU pragma: export
#include "hsa/hsa_ext_finalize.h"     // IWYU pragma: export

#endif  // IREE_HAL_DRIVERS_HSA_HSA_HEADERS_H_

