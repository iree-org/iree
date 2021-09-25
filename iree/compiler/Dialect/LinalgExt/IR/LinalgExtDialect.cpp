// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOpsDialect.cpp.inc"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

void LinalgExtDialect::initialize() {
  // TODO(hanchung): Add interface to the dialect.
  // addInterfaces<IREEInlinerInterface>();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
