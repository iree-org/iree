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
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-lower-coalesced-dma-to-global-loads"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERCOALESCEDDMATOGLOBALLOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static LogicalResult verifyThreadMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    return failure();
  }

  // Verify that all mappings are thread mappings
  for (Attribute attr : *mappingAttr) {
    if (!isa<gpu::GPUThreadMappingAttr>(attr)) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyMemoryLayout(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                        PatternRewriter &rewriter) {
  // Check that destination memref is contiguous.
  auto destMemRefType = cast<MemRefType>(dmaOp.getInit().getType());

  if (!destMemRefType.areTrailingDimsContiguous(1)) {
    return rewriter.notifyMatchFailure(
        dmaOp,
        "destination memref does not have contiguous trailing dimension");
  }

  auto sourceType = cast<MemRefType>(dmaOp.getSource().getType());
  auto targetType = cast<MemRefType>(dmaOp.getInit().getType());

  bool hasGlobalSource = hasGlobalMemoryAddressSpace(sourceType);
  bool hasSharedTarget = hasSharedMemoryAddressSpace(targetType);

  if (!hasGlobalSource || !hasSharedTarget) {
    return rewriter.notifyMatchFailure(
        dmaOp, "incompatible source or target memory address space");
  }

  return success();
}

static LogicalResult
isEligibleForGlobalDMA(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                       PatternRewriter &rewriter) {
  scf::ForallOp forallOp = dmaOp->getParentOfType<scf::ForallOp>();

  // Verify that the forall has the required GPU thread mapping.
  if (failed(verifyThreadMapping(forallOp))) {
    return failure();
  }

  if (failed(verifyMemoryLayout(dmaOp, rewriter))) {
    return failure();
  }

  return success();
}

// Lowers iree_gpu.coalesced_gather_dma operations within scf.forall loops
// with thread mapping to amdgpu.gather_to_lds operations inside scf.for loops.
//
// In short, looks for patterns like this:
//   scf.forall (...) in (1, 32) {
//     %1 = iree_gpu.coalesced_gather_dma %source into %dest
//   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
//
// And replaces the DMA op with a loop that slices the source using the largest
// DMA size from the target's dma_sizes attribute:
//   scf.for %iv = %c0 to %num_dmas step %c1 {
//     %offset = arith.muli %iv, %elements_per_dma
//     amdgpu.gather_to_lds %source[0, %offset], %dest[0, 0]
//   }
struct LowerCoalescedGatherDMAPattern
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> workgroupSize,
                                 int64_t subgroupSize,
                                 ArrayRef<int64_t> targetDmaSizes)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        workgroupSize(workgroupSize), subgroupSize(subgroupSize),
        targetDmaSizes(targetDmaSizes) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(
          dmaOp, "coalesced_gather_dma not inside scf.forall");
    }

    if (failed(verifyThreadMapping(forallOp))) {
      return failure();
    }

    if (failed(verifyMemoryLayout(dmaOp, rewriter))) {
      LDBG() << "  - Memory layout verification failed";
      return failure();
    }

    Location loc = dmaOp.getLoc();

    Value source = dmaOp.getSource();
    Value dest = dmaOp.getInit();

    auto sourceType = cast<MemRefType>(source.getType());
    Type transferType = sourceType.getElementType();
    int64_t elementBits = sourceType.getElementTypeBitWidth();

    ArrayRef<int64_t> dmaSizes = targetDmaSizes;
    if (dmaSizes.empty()) {
      return rewriter.notifyMatchFailure(dmaOp,
                                         "target does not specify dma_sizes");
    }

    int64_t maxDmaSizeBits =
        *std::max_element(dmaSizes.begin(), dmaSizes.end());
    int64_t elementsPerDma = maxDmaSizeBits / elementBits;
    int64_t totalElements = sourceType.getShape().back();
    int64_t numDmas = (totalElements + elementsPerDma - 1) / elementsPerDma;

    rewriter.setInsertionPoint(dmaOp);

    // Create the for loop
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numDmasVal = rewriter.create<arith::ConstantIndexOp>(loc, numDmas);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto forOp = rewriter.create<scf::ForOp>(loc, c0, numDmasVal, c1);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value iv = forOp.getInductionVar();

    Value offsetVal = rewriter.create<arith::MulIOp>(
        loc, iv, rewriter.create<arith::ConstantIndexOp>(loc, elementsPerDma));

    // For source, use the original source memref with offset indices
    SmallVector<Value> srcIndices;
    for (int64_t i = 0; i < sourceType.getRank() - 1; ++i) {
      srcIndices.push_back(zero);
    }
    srcIndices.push_back(offsetVal);

    // For destination, always use [0, 0, ...] indices
    auto destType = cast<MemRefType>(dest.getType());
    SmallVector<Value> dstIndices(destType.getRank(), zero);

    rewriter.create<amdgpu::GatherToLDSOp>(
        loc, source, srcIndices, dest, dstIndices, TypeAttr::get(transferType));

    rewriter.replaceOp(dmaOp, dest);

    return success();
  }

private:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
  ArrayRef<int64_t> targetDmaSizes;
};

namespace {
struct GPULowerCoalescedDMAToGlobalLoadsPass final
    : impl::GPULowerCoalescedDMAToGlobalLoadsPassBase<
          GPULowerCoalescedDMAToGlobalLoadsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    int count = 0;
    funcOp.walk([&count](IREE::GPU::CoalescedGatherDMAOp op) {
      LDBG() << "Found CoalescedGatherDMAOp: " << op;
      ++count;
    });

    std::optional<SmallVector<int64_t>> workgroupSize =
        mlir::iree_compiler::getWorkgroupSize(funcOp);
    auto subgroupSize = mlir::iree_compiler::getSubgroupSize(funcOp);
    if (!subgroupSize) {
      funcOp.emitOpError(
          "unimplemented: Distribution with dynamic subgroup size.");
      return signalPassFailure();
    }

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    if (!target) {
      funcOp.emitOpError("Missing GPU target attribute");
      return signalPassFailure();
    }

    // Get dma_sizes from target attribute
    ArrayRef<int64_t> dmaSizes;
    if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }
    if (dmaSizes.empty()) {
      funcOp.emitOpError("Target does not specify dma_sizes");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, *workgroupSize,
                                                 *subgroupSize, dmaSizes);

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};
} // namespace

} // namespace mlir::iree_compiler
