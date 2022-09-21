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

/// Filter to decide which contract ops need allocations.
static bool contractOpFilter(Operation *op) {
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

/// Filter to decide which transpose ops need allocations.
static bool transposeOpFilter(Operation *op) {
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp) return false;
  // Can't promote dynamic shapes.
  if (genericOp.hasDynamicShape()) return false;
  return genericOp.getNumParallelLoops() >= 2 &&
         genericOp.getNumParallelLoops() <= 3;
}

/// Returns true if the index map represents a transpose that benefits from
/// shared mem.
static bool isSharedMemTranspose(AffineMap indexMap) {
  if (!indexMap.isEmpty() && indexMap.isPermutation()) {
    // Ensure that the fasted moving dimension (the last one) is permuted,
    // Otherwise shared memory promotion will not benefit the operation.
    if (indexMap.getDimPosition(indexMap.getNumDims() - 1) !=
        indexMap.getNumDims() - 1) {
      return true;
    }
  }
  return false;
}

namespace {
struct LLVMGPUTensorAllocPass
    : public LLVMGPUTensorAllocBase<LLVMGPUTensorAllocPass> {
 private:
  GPUPromoteSharedMemPattern promoteSharedMemPattern =
      GPUPromoteSharedMemPattern::ContractionOpPattern;

 public:
  LLVMGPUTensorAllocPass(GPUPromoteSharedMemPattern promoteSharedMemPattern)
      : promoteSharedMemPattern(promoteSharedMemPattern) {}
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
      switch (promoteSharedMemPattern) {
        case GPUPromoteSharedMemPattern::ContractionOpPattern:
          if (contractOpFilter(op)) opsToPromote.push_back(op);
          break;
        case GPUPromoteSharedMemPattern::TransposeOpPattern:
          if (transposeOpFilter(op)) opsToPromote.push_back(op);
          break;
      }
    });
    for (Operation *op : opsToPromote) {
      OpBuilder builder(op);
      auto linalgOp = cast<linalg::LinalgOp>(op);
      bufferization::BufferizationOptions options;
      // Promote all the input operands for contract op or transpose operands
      // for shared mem transpose.
      for (auto operand : linalgOp.getInputOperands()) {
        if (promoteSharedMemPattern ==
            GPUPromoteSharedMemPattern::TransposeOpPattern) {
          if (!isSharedMemTranspose(linalgOp.getTiedIndexingMap(operand))) {
            continue;
          }
        }
        FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
            builder, op->getLoc(), operand->get(), false, options, true);
        if (failed(ret)) {
          return signalPassFailure();
        }
        Value v = ret.getValue();
        operand->set(v);
      }
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorAlloc(
    GPUPromoteSharedMemPattern promoteSharedMemPattern) {
  return std::make_unique<LLVMGPUTensorAllocPass>(promoteSharedMemPattern);
}

}  // namespace iree_compiler
}  // namespace mlir
