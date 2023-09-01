// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "simple_io_sample/IR/SimpleIODialect.h"

#include "simple_io_sample/IR/SimpleIOOps.h"

namespace mlir::iree_compiler::IREE::SimpleIO {

void SimpleIODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "simple_io_sample/IR/SimpleIOOps.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::SimpleIO

#include "simple_io_sample/IR/SimpleIODialect.cpp.inc"
