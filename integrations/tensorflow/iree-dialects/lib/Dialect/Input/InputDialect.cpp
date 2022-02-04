// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/Input/InputDialect.h"

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::Input;

#include "iree-dialects/Dialect/Input/InputDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"

void IREEInputDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-dialects/Dialect/Input/InputTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "iree-dialects/Dialect/Input/InputOps.cpp.inc"
      >();
}
