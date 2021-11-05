// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mmt4d.h"

#if !defined(__aarch64__)

MMT4D_GENERIC(8, 4, 8, int8_t, int8_t, int32_t);

#endif  // arch exclusions
