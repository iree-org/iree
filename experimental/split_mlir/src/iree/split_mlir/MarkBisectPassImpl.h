// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include <memory>

#include "iree/split_mlir/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree {
namespace split_mlir {

#define GEN_PASS_DEF_MARKBISECT
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

namespace {

void markRangeFirst(Operation& op, OpBuilder& builder) {
  op.setAttr("outline_range_first", builder.getUnitAttr());
}

void markRangeLast(Operation& op, OpBuilder& builder) {
  op.setAttr("outline_range_last", builder.getUnitAttr());
}

struct MarkBisectPass : public impl::MarkBisectBase<MarkBisectPass> {
  using MarkBisectBase::MarkBisectBase;

  LogicalResult initialize(MLIRContext* context) override {
    functionsSet.insert(functions.begin(), functions.end());
    return LogicalResult::success();
  }

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    if (!functionsSet.contains(funcOp.getSymName())) {
      return;
    }
    if (funcOp.getBody().getBlocks().size() > 1) {
      return signalPassFailure();
    }
    Block& entryBlock = funcOp.getBody().front();
    if (entryBlock.getOperations().size() < 3) {
      // Degenerate case. Needs at least 1 op for each half + the return op.
      return;
    }
    size_t opsCount = entryBlock.getOperations().size();
    size_t cutOpIndex = (opsCount - 1) / 2;
    OpBuilder builder(&getContext());
    // Ranges are inclusive, [first, last].
    auto firstHalfLastOp = entryBlock.begin();
    std::advance(firstHalfLastOp, cutOpIndex - 1);
    markRangeFirst(entryBlock.front(), builder);
    markRangeLast(*firstHalfLastOp, builder);
    auto secondHalfFirstOp = firstHalfLastOp;
    std::advance(secondHalfFirstOp, 1);
    markRangeFirst(*secondHalfFirstOp, builder);
    auto secondHalfLastOp = entryBlock.end();
    // Take operation that is just before the return operation.
    std::advance(secondHalfLastOp, -2);
    markRangeLast(*secondHalfLastOp, builder);
  }

 private:
  llvm::SmallSet<llvm::StringRef, 3> functionsSet;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createMarkBisectPass() {
  return std::make_unique<MarkBisectPass>();
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
