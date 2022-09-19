// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-alloc"

namespace mlir {
namespace iree_compiler {

/// Filter to decide which ops need allocations.
static bool filter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;
  // Can't promote dynamic shapes.
  if (linalgOp.hasDynamicShape()) return false;
  SmallVector<unsigned> dims;
  linalgOp.getParallelDims(dims);
  SmallVector<int64_t, 4> shapes = linalgOp.getStaticLoopRanges();
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  return numNonUnitParallelLoop > 1 && linalg::isaContractionOpInterface(op) &&
         linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

namespace {
struct LLVMGPUTensorAllocPass
    : public LLVMGPUTensorAllocBase<LLVMGPUTensorAllocPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Tile the reduction first to reduce the alloc size.
    if (failed(tileReduction(funcOp))) {
      return signalPassFailure();
    }

    SmallVector<Operation *> opsToPromote;
    funcOp.walk([&](Operation *op) {
      if (filter(op)) opsToPromote.push_back(op);
    });
    for (Operation *op : opsToPromote) {
      OpBuilder builder(op);
      auto linalgOp = cast<linalg::LinalgOp>(op);
      bufferization::BufferizationOptions options;
      // Promote all the input operands.
      for (auto operand : linalgOp.getInputOperands()) {
        FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
            builder, op->getLoc(), operand->get(), false, options, true);
        if (failed(ret)) {
          return signalPassFailure();
        }
        Value v = ret.getValue();
        operand->get().replaceAllUsesExcept(v, v.getDefiningOp());
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorAlloc() {
  return std::make_unique<LLVMGPUTensorAllocPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
