// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUPasses.h"
#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-alloc"

namespace mlir {
namespace iree_compiler {

// For optimal performance we always want to copy 128 bits
static constexpr int copyVectorNumBits = 128;

/// Filter to decide which contract ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;

  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }

  // The workgroup specialization already makes static shapes available for the
  // main tile part and makes the partial tile computation small, so promoting
  // to shared memory for the partial tile actually hurts the performance.
  if (linalgOp.hasDynamicShape()) return false;

  // Check if the shape is tile-distributable. The leading dimension must be a
  // multiple of the target vector size, which is 128b / the element bit width.
  auto isTileDistributable = [&](OpOperand *v) {
    ShapedType ty = v->get().getType().cast<ShapedType>();
    unsigned bitWidth = ty.getElementTypeBitWidth();
    int targetVectorSize = copyVectorNumBits / bitWidth;
    return ty.getShape().back() % targetVectorSize == 0;
  };

  if (!llvm::all_of(linalgOp.getDpsInputOperands(), isTileDistributable)) {
    return false;
  }

  if (!llvm::all_of(linalgOp.getDpsInitOperands(), isTileDistributable)) {
    return false;
  }

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
  return numNonUnitParallelLoop > 1 && linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

/// Filter to decide which transpose ops need allocations.
static bool transposeOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return false;
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
  return opInfo.isTranspose();
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
    registry.insert<bufferization::BufferizationDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Tile the reduction first to reduce the alloc size.
    if (failed(tileToSerialLoops(funcOp))) {
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
      switch (promoteSharedMemPattern) {
        case GPUPromoteSharedMemPattern::ContractionOpPattern:
          // Promote all the input operands
          for (auto operand : linalgOp.getDpsInputOperands()) {
            FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
                builder, op->getLoc(), operand->get(), false, options, true);
            if (failed(ret)) {
              return signalPassFailure();
            }
            Value v = ret.value();
            operand->set(v);
          }
          break;

        case GPUPromoteSharedMemPattern::TransposeOpPattern:
          LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

          for (auto operand : opInfo.getTransposeOperands()) {
            FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
                builder, op->getLoc(), operand->get(), false, options, true);
            if (failed(ret)) {
              return signalPassFailure();
            }
            Value v = ret.value();
            operand->set(v);
          }
          break;
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
