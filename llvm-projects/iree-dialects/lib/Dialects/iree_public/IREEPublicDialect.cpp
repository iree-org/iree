// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialects/iree_public/IREEPublicDialect.h"

#include "iree-dialects/Dialects/iree_public/IREEPublicOps.h"

using namespace mlir;
using namespace mlir::iree_public;

#include "iree-dialects/Dialects/iree_public/IREEPublicOpsDialect.cpp.inc"

void IREEPublicDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialects/iree_public/IREEPublicOps.cpp.inc"
      >();
}
