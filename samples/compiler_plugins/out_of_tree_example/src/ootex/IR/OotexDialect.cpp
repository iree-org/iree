// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ootex/IR/OotexDialect.h"

#include "ootex/IR/OotexOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "ootex/IR/OotexOps.cpp.inc"

namespace ootex {

void OotexDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ootex/IR/OotexOps.cpp.inc"
      >();
}

}  // namespace ootex
