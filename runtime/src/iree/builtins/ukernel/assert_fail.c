// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standard implementation of iree_uk_assert_impl.
// Microkernel code needs to be stand-alone, not including the standard library
// (see comment in common.h). But it's hard to implement assertion failure
// without the standard library. So this implementation is kept in a separate
// file, so that:
// 1. The standard library dependency is a "hidden detail" of this library.
// 2. Each user can choose to either link this or provide their own replacement.

#include <stdio.h>
#include <stdlib.h>

#include "iree/builtins/ukernel/common.h"

#ifdef IREE_UK_ENABLE_ASSERTS
void iree_uk_assert_fail(const char* file, int line, const char* function,
                         const char* condition) {
  fflush(stdout);
  fprintf(stderr, "%s:%d: %s: assertion failed: %s\n", file, line, function,
          condition);
  fflush(stderr);
  abort();
}
#endif  // IREE_UK_ENABLE_ASSERTS