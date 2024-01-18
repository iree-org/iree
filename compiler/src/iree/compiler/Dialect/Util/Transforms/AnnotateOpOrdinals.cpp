// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

class AnnotateOpOrdinalsPass
    : public AnnotateOpOrdinalsBase<AnnotateOpOrdinalsPass> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto attrName = StringAttr::get(context, "util.ordinal");
    auto indexType = IndexType::get(context);
    int64_t globalOrdinal = 0;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      op->setAttr(attrName, IntegerAttr::get(indexType, globalOrdinal++));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateOpOrdinalsPass() {
  return std::make_unique<AnnotateOpOrdinalsPass>();
}

} // namespace mlir::iree_compiler::IREE::Util
