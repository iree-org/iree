// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/MIPS/IR/MIPSDialect.h"

#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::MIPS;

//===----------------------------------------------------------------------===//
// Inliner interface — allow MIPS ops to be inlined unconditionally.
//===----------------------------------------------------------------------===//

namespace {
struct MIPSInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect initialize
//===----------------------------------------------------------------------===//

void MIPSDialect::initialize() {
  addInterfaces<MIPSInlinerInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/MIPS/IR/MIPSOps.cpp.inc"
  >();
}

#include "iree/compiler/Dialect/MIPS/IR/MIPSDialect.cpp.inc"
