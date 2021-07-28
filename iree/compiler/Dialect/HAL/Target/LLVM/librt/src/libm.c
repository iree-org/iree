// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libm.h"

// https://en.cppreference.com/w/c/numeric/math/fma
LIBRT_EXPORT float fmaf(float x, float y, float z) {
  // TODO(*): a real implementation :)
  return (x * y) + z;
}
