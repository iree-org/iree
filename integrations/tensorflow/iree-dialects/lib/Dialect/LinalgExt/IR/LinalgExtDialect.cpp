// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

void IREELinalgExtDialect::initialize() {
  // TODO(hanchung): Add interface to the dialect.
  // addInterfaces<IREEInlinerInterface>();
#define GET_OP_LIST
  addOperations<
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();
}

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.cpp.inc"
