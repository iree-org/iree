
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_HEADERS_H_
#define IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error 32-bit not supported on level zero
#endif  // defined(IREE_PTR_SIZE_32)

#include "ze_api.h"  // IWYU pragma: export

#endif  // IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_HEADERS_H_
