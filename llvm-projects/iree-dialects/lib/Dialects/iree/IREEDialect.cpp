// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialects/iree/IREEDialect.h"

#include "iree-dialects/Dialects/iree/IREEOps.h"

using namespace mlir;
using namespace mlir::iree;

#include "iree-dialects/Dialects/iree/IREEOpsDialect.cpp.inc"

void IREEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialects/iree/IREEOps.cpp.inc"
      >();
}
