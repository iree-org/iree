// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include <optional>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-lower-to-global-loads"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOGLOBALLOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static constexpr int kNumBitsPerCopy = 32;

static LogicalResult
distributeLinalgCopyToThreads(RewriterBase &rewriter, linalg::CopyOp copy,
                              ArrayRef<int64_t> workgroupSize,
                              int64_t subgroupSize) {
  LDBG("==== distributing op: ");
  LDBG(*copy);
  Location loc = copy.getLoc();

  // The linalg.copy we are dealing with represents a region we need to copy to
  // workgroup memory. Assume there are N threads in the workgroup, then there
  // are `num_subgroups = N / gpu.subgroup_size` subgroups in the workgroup.
  //
  // So we are slicing up the target memref into `num_subgroups` consecutive
  // slices, and threads in the same subgroup will copy their slice to workgroup
  // memory slice.

  // Get the copy size:
  auto copyMemRefType = cast<MemRefType>(copy.getOperand(1).getType());
  if (!memref::isStaticShapeAndContiguousRowMajor(copyMemRefType)) {
    return rewriter.notifyMatchFailure(copy,
                                       "Copy to non-static or non-contiguous, "
                                       "non-row major memref.");
  }
  int64_t rank = copyMemRefType.getRank();
  SmallVector<OpFoldResult> tileSize(rank - 1, rewriter.getIndexAttr(1));

  int64_t elementBitWidth = copyMemRefType.getElementTypeBitWidth();
  if (kNumBitsPerCopy % elementBitWidth != 0) {
    return rewriter.notifyMatchFailure(copy, "Copy size is not a multiple of "
                                             "element bit width.");
  }
  int64_t elementsPerCopy = kNumBitsPerCopy / elementBitWidth;

  // Divide the copy by subgroup, and load linearly.
  assert(workgroupSize[0] % subgroupSize == 0);

  int64_t numSubgroups = workgroupSize[0] / subgroupSize;
  int64_t totalCopySize = copyMemRefType.getNumElements();
  int64_t totalCopySizePerSubgroup = totalCopySize / numSubgroups;
  int64_t numCopiesPerThread =
      (totalCopySizePerSubgroup / elementsPerCopy) / subgroupSize;
  int64_t residualElements =
      totalCopySizePerSubgroup % (subgroupSize * elementsPerCopy);

  LDBG("-- elementsPerCopy: " << elementsPerCopy);
  LDBG("-- workgroupSize: " << workgroupSize[0]);
  LDBG("-- numSubgroups: " << numSubgroups);
  LDBG("-- totalCopySize: " << totalCopySize);
  LDBG("-- totalCopySizePerSubgroup: " << totalCopySizePerSubgroup);
  LDBG("-- numCopiesPerThread: " << numCopiesPerThread);
  LDBG("-- residualElements: " << residualElements);

  if (residualElements != 0) {
    return rewriter.notifyMatchFailure(
        copy, "Cannot proceed: cannot handle copying residual elements.");
  }

  Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, nullptr);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, nullptr);

  auto sourceType = cast<MemRefType>(copy.getOperand(0).getType());
  auto localType = cast<MemRefType>(copy.getOutputs().front().getType());

  auto getGlobalGatherIndex = [&](Value sgIdVal, Value lIdVal,
                                  Value indVar) -> Value {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return rewriter.create<affine::AffineLinearizeIndexOp>(
        loc, ValueRange{sgIdVal, indVar, lIdVal, zero},
        ArrayRef<int64_t>{numSubgroups, numCopiesPerThread, subgroupSize,
                          elementsPerCopy},
        /*disjoint=*/true);
  };

  auto getSubgroupStoreBaseIndex = [&](Value sgIdVal, Value indVar) -> Value {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    return getGlobalGatherIndex(sgIdVal, zero, indVar);
  };

  // Build a for loop skeleton:
  auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto upperBound =
      rewriter.create<arith::ConstantIndexOp>(loc, numCopiesPerThread);
  auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  scf::ForOp forOp =
      rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

  auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
    return rewriter.create<affine::AffineDelinearizeIndexOp>(loc, index, shape)
        .getMultiIndex();
  };

  // For loop body:
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    auto inductionVar = forOp.getInductionVar();
    Value linearizedGatherIndices =
        getGlobalGatherIndex(subgroupId, laneId, inductionVar);
    ValueRange delinearizedGlobalIndices =
        delinearizeIndex(linearizedGatherIndices, sourceType.getShape());
    Value linearizedBaseIndices =
        getSubgroupStoreBaseIndex(subgroupId, inductionVar);
    ValueRange delinearizedLocalIndices =
        delinearizeIndex(linearizedBaseIndices, localType.getShape());
    rewriter.create<IREE::GPU::GlobalLoadDMAOp>(
        loc, copy.getOperand(0), delinearizedGlobalIndices,
        copy.getOutputs()[0], delinearizedLocalIndices);
  }

  // Sync at the end of the loop across threads.
  rewriter.replaceOpWithNewOp<gpu::BarrierOp>(copy);
  return success();
}

static LogicalResult isEligibleForGlobalDMA(linalg::CopyOp copy) {
  // Source must be global address and target must be workgroup address.
  auto sourceType = cast<MemRefType>(copy.getOperand(0).getType());
  auto targetType = cast<MemRefType>(copy.getOutputs().front().getType());

  if (!getLoweringConfig<IREE::GPU::UseGlobalLoadDMAAttr>(copy)) {
    LDBG("-- Op: " << *copy);
    LDBG("-- does not have `use_global_load_dma` attribute, skipping.");
    return failure();
  }

  if (!hasGlobalMemoryAddressSpace(sourceType) ||
      !hasSharedMemoryAddressSpace(targetType)) {
    LDBG("-- Op: " << *copy);
    LDBG("-- incompatible source or target memory address space.");
    return failure();
  }

  // TODO: check that the copy's target memref is not a subview: a subview
  // cannot guarantee contiguity of dest memory region.
  return success();
}

struct LowerToDMAPattern : public OpRewritePattern<linalg::CopyOp> {
  LowerToDMAPattern(MLIRContext *context, ArrayRef<int64_t> workgroupSize,
                    int64_t subgroupSize)
      : OpRewritePattern<linalg::CopyOp>(context), workgroupSize(workgroupSize),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(linalg::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    if (failed(isEligibleForGlobalDMA(copy))) {
      return failure();
    }
    return distributeLinalgCopyToThreads(rewriter, copy, workgroupSize,
                                         subgroupSize);
  }

private:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
};

namespace {
struct GPULowerToGlobalLoadsPass final
    : impl::GPULowerToGlobalLoadsPassBase<GPULowerToGlobalLoadsPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    std::optional<SmallVector<int64_t>> workgroupSize =
        mlir::iree_compiler::getWorkgroupSize(funcOp);
    if (!workgroupSize) {
      funcOp.emitOpError(
          "unimplemented: Distribution with dynamic workgroup size.");
      return signalPassFailure();
    }
    auto subgroupSize = mlir::iree_compiler::getSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp.emitOpError(
          "unimplemented: Distribution with dynamic subgroup size.");
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<LowerToDMAPattern>(context, *workgroupSize, *subgroupSize);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }
};
} // namespace
} // namespace mlir::iree_compiler
