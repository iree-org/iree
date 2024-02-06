// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/IR/IndexingDialect.h"

#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::Indexing {

// Used for custom printing support.
struct IndexingOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    return AliasResult::NoAlias;
  }
};

void IndexingDialect::registerAttributes() {}
void IndexingDialect::registerTypes() {}

IndexingDialect::IndexingDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<IndexingDialect>()) {
  addInterfaces<IndexingOpAsmInterface>();
  registerAttributes();
  registerTypes();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Indexing/IR/IndexingOps.cpp.inc"
      >();
}

void IndexingDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {}

} // namespace mlir::iree_compiler::IREE::Indexing
