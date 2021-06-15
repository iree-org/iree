// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// If the value is a threadID return the range [0, workgroupSize-1].
static Optional<std::pair<AffineExpr, AffineExpr>> threadIdMinMax(
    Value value, SmallVectorImpl<Value> &dims, SmallVectorImpl<Value> &symbols,
    ArrayRef<int32_t> workgroupSize) {
  if (auto idOp = value.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = StringSwitch<unsigned>(idOp.dimension())
                         .Case("x", 0)
                         .Case("y", 1)
                         .Case("z", 2);
    OpBuilder b(value.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  return {};
}

namespace {

class LinalgToLLVMGPURemoveSingleIterationLoopPass
    : public LinalgToLLVMGPURemoveSingleIterationLoopBase<
          LinalgToLLVMGPURemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    std::array<int32_t, 3> workgroupSize;
    for (auto it : llvm::enumerate(funcOp->getAttr("llvmgpu_workgroup_size")
                                       .cast<DenseIntElementsAttr>()
                                       .getIntValues())) {
      workgroupSize[it.index()] = it.value().getZExtValue();
    }
    auto getThreadIdMinMax = [&workgroupSize](Value value,
                                              SmallVectorImpl<Value> &dims,
                                              SmallVectorImpl<Value> &symbols) {
      return threadIdMinMax(value, dims, symbols, workgroupSize);
    };
    MLIRContext *context = funcOp->getContext();
    OwningRewritePatternList patterns(context);
    populateRemoveSingleIterationLoopPattern(patterns, getThreadIdMinMax);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMGPURemoveSingleIterationLoopPass() {
  return std::make_unique<LinalgToLLVMGPURemoveSingleIterationLoopPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
