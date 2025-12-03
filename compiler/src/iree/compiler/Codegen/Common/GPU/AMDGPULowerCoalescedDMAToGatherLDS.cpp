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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-lower-coalesced-dma-to-gather-lds"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPULOWERCOALESCEDDMATOGATHERLDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

static LogicalResult verifyThreadMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    return failure();
  }

  // Verify that all mappings are thread mappings.
  return success(llvm::all_of(
      *mappingAttr,
      llvm::IsaPred<gpu::GPUThreadMappingAttr, IREE::GPU::LaneIdAttr>));
}

static LogicalResult
verifyMemoryLayoutContiguous(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                             PatternRewriter &rewriter) {
  // Important: check that destination memref is contiguous.
  auto destMemRefType = cast<MemRefType>(dmaOp.getInit().getType());

  if (!destMemRefType.areTrailingDimsContiguous(1)) {
    return rewriter.notifyMatchFailure(
        dmaOp,
        "destination memref does not have contiguous trailing dimension");
  }

  return success();
}

struct LowerCoalescedGatherDMAPattern final
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using Base::Base;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> targetDmaSizes)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        targetDmaSizes(targetDmaSizes) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    LDBG() << "Processing CoalescedGatherDMAOp: " << dmaOp;

    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(
          dmaOp, "coalesced_gather_dma not inside scf.forall");
    }

    if (failed(verifyThreadMapping(forallOp))) {
      return rewriter.notifyMatchFailure(
          dmaOp, "forall does not have proper thread mapping");
    }

    if (failed(verifyMemoryLayoutContiguous(dmaOp, rewriter))) {
      return failure();
    }

    Value source = dmaOp.getSource();
    Value dest = dmaOp.getInit();
    OperandRange indicesRange = dmaOp.getIndices();

    auto sourceType = cast<MemRefType>(source.getType());
    auto destType = cast<MemRefType>(dest.getType());

    if (!indicesRange.empty()) {
      Value indices = indicesRange.front();
      auto indicesType = cast<MemRefType>(indices.getType());
      ArrayRef<int64_t> indicesShape = indicesType.getShape();
      ArrayRef<int64_t> destShape = destType.getShape();

      // Verify that indices dimensions match dest dimensions.
      if (indicesShape != destShape) {
        return rewriter.notifyMatchFailure(
            dmaOp, "indices shape does not match dest shape");
      }
    }

    Type elementType = sourceType.getElementType();
    int64_t elementBits = sourceType.getElementTypeBitWidth();
    LDBG() << "  Element type: " << elementType;
    LDBG() << "  Element bits: " << elementBits;

    int64_t innermostDimSize = sourceType.getShape().back();
    int64_t transferSizeBits = innermostDimSize * elementBits;
    LDBG() << "  Source innermost dimension size: " << innermostDimSize;
    LDBG() << "  Transfer size in bits: " << transferSizeBits;

    std::optional<int64_t> subgroupSize =
        getSubgroupSize(dmaOp->getParentOfType<FunctionOpInterface>());
    if (!subgroupSize.has_value()) {
      return rewriter.notifyMatchFailure(
          dmaOp, "unable to determine subgroup size from forall");
    }
    LDBG() << "Subgroup size: " << *subgroupSize;

    // Check that transfer size matches one of the target DMA sizes.
    int64_t transferSizePerLane = transferSizeBits / *subgroupSize;
    LDBG() << "Transfer size per lane: " << transferSizePerLane << " bits";

    if (!targetDmaSizes.empty() &&
        !llvm::is_contained(targetDmaSizes, transferSizePerLane)) {
      return rewriter.notifyMatchFailure(
          dmaOp, "transfer size does not match any target DMA size");
    }
    LDBG() << "Transfer size matches target DMA sizes";

    // TODO: Handle indices properly - for now we skip the explicit indices
    // check The lane parameter is always present but we handle it differently.

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    LDBG() << "Source rank: " << sourceShape.size();
    if (sourceShape.size() < 2) {
      return rewriter.notifyMatchFailure(
          dmaOp, "source must have at least 2 dimensions");
    }

    int64_t secondInnermostDimSize = sourceShape[sourceShape.size() - 2];
    LDBG() << "  Source second innermost dimension size: "
           << secondInnermostDimSize;

    int64_t elementsPerTransfer = innermostDimSize / *subgroupSize;

    auto transferType = VectorType::get({elementsPerTransfer}, elementType);

    // Actually create the GatherToLDS ops to perform the transfer.
    rewriter.setInsertionPoint(dmaOp);

    TypedValue<IndexType> laneId = dmaOp.getLane();

    Location loc = dmaOp.getLoc();
    Value laneOffset = arith::MulIOp::create(
        rewriter, loc, laneId,
        arith::ConstantIndexOp::create(rewriter, loc, elementsPerTransfer));

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange({secondInnermostDimSize}, {1})) {
      Value iVal = arith::ConstantIndexOp::create(rewriter, loc, offsets[0]);

      SmallVector<Value> srcIndices(sourceType.getRank(), zero);
      srcIndices[sourceType.getRank() - 2] = iVal;
      srcIndices[sourceType.getRank() - 1] = laneOffset;

      SmallVector<Value> dstIndices(destType.getRank(), zero);
      dstIndices[destType.getRank() - 2] = iVal;

      amdgpu::GatherToLDSOp::create(rewriter, loc, source, srcIndices, dest,
                                    dstIndices, TypeAttr::get(transferType));
    }

    rewriter.eraseOp(dmaOp);
    return success();
  }

private:
  ArrayRef<int64_t> targetDmaSizes;
};

namespace {
struct AMDGPULowerCoalescedDMAToGatherLDSPass final
    : impl::AMDGPULowerCoalescedDMAToGatherLDSPassBase<
          AMDGPULowerCoalescedDMAToGatherLDSPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    LDBG() << "GPU Target attribute: " << target;
    if (!target) {
      LDBG() << "Missing GPU target attribute, pass will fail";
      // Don't fail if no target attribute - just skip the pass.
      return;
    }

    ArrayRef<int64_t> dmaSizes;
    if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }

    // dma_sizes is optional - if not specified, skip the size validation.
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, dmaSizes);

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};
} // namespace

} // namespace mlir::iree_compiler
