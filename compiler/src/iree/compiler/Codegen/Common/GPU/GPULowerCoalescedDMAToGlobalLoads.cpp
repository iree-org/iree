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
  // Accept both gpu::GPUThreadMappingAttr and iree_gpu.lane_id
  for (Attribute attr : *mappingAttr) {
    if (!isa<gpu::GPUThreadMappingAttr, IREE::GPU::LaneIdAttr>(attr)) {
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

  /*
  auto sourceType = cast<MemRefType>(dmaOp.getSource().getType());
  auto targetType = cast<MemRefType>(dmaOp.getInit().getType());

  bool hasGlobalSource = hasGlobalMemoryAddressSpace(sourceType);
  bool hasSharedTarget = hasSharedMemoryAddressSpace(targetType);

  if (!hasGlobalSource || !hasSharedTarget) {
    return rewriter.notifyMatchFailure(
        dmaOp, "incompatible source or target memory address space");
  }
  */

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

static bool matchesTargetDmaSize(int64_t transferSizeBytes,
                                 ArrayRef<int64_t> targetDmaSizes) {
  for (int64_t dmaSize : targetDmaSizes) {
    if (transferSizeBytes == dmaSize) {
      return true;
    }
  }
  return false;
}

struct LowerCoalescedGatherDMAPattern
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> targetDmaSizes)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        targetDmaSizes(targetDmaSizes) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(LDBG() << "Processing CoalescedGatherDMAOp: " << dmaOp << "\n");

    // Verify the DMA op is inside a scf.forall
    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
      LLVM_DEBUG(LDBG() << "FAILED - not inside scf.forall\n");
      return rewriter.notifyMatchFailure(
          dmaOp, "coalesced_gather_dma not inside scf.forall");
    }
    LLVM_DEBUG(LDBG() << "Found parent forall\n");

    // Verify thread mapping
    if (failed(verifyThreadMapping(forallOp))) {
      LLVM_DEBUG(LDBG() << "FAILED - thread mapping verification failed\n");
      return rewriter.notifyMatchFailure(
          dmaOp, "forall does not have proper thread mapping");
    }
    LLVM_DEBUG(LDBG() << "Thread mapping verified\n");

    // Verify memory layout
    if (failed(verifyMemoryLayout(dmaOp, rewriter))) {
      LLVM_DEBUG(LDBG() << "FAILED - memory layout verification failed\n");
      return failure();
    }
    LLVM_DEBUG(LDBG() << "Memory layout verified\n");

    [[maybe_unused]] Location loc = dmaOp.getLoc();
    Value source = dmaOp.getSource();
    Value dest = dmaOp.getInit();

    LDBG() << "  Source: " << source;
    LDBG() << "  Dest: " << dest;

    // Get all operands
    auto indicesRange = dmaOp.getIndices();

    auto sourceType = cast<MemRefType>(source.getType());
    auto destType = cast<MemRefType>(dest.getType());

    // Check if indices operand exists
    if (!indicesRange.empty()) {
      Value indices = indicesRange.front();
      LDBG() << "  Indices: " << indices;
      auto indicesType = cast<MemRefType>(indices.getType());
      ArrayRef<int64_t> indicesShape = indicesType.getShape();
      ArrayRef<int64_t> destShape = destType.getShape();

      // Verify that indices dimensions are a prefix of dest dimensions
      if (indicesShape.size() > destShape.size()) {
        return rewriter.notifyMatchFailure(dmaOp,
                                           "indices rank exceeds dest rank");
      }

      for (size_t i = 0; i < indicesShape.size(); ++i) {
        if (indicesShape[i] != destShape[i]) {
          return rewriter.notifyMatchFailure(
              dmaOp, "indices shape is not a prefix of dest shape");
        }
      }

      // Get the trailing missing dimension size (innermost dimension only)
      [[maybe_unused]] int64_t trailingDims =
          destShape.size() - indicesShape.size();
      LDBG() << "  Trailing dimensions: " << trailingDims;

      // Get only the innermost dimension size
      [[maybe_unused]] int64_t trailingSize = destShape.back();
      LDBG() << "  Trailing size (innermost): " << trailingSize;
    }

    // Get element size
    Type elementType = sourceType.getElementType();
    int64_t elementBits = sourceType.getElementTypeBitWidth();
    LDBG() << "  Element type: " << elementType;
    LDBG() << "  Element bits: " << elementBits;

    // Get innermost dimension size of SOURCE and check against target DMA sizes
    int64_t innermostDimSize = sourceType.getShape().back();
    int64_t transferSizeBits = innermostDimSize * elementBits;
    LDBG() << "  Source innermost dimension size: " << innermostDimSize;
    LDBG() << "  Transfer size in bits: " << transferSizeBits;

    std::optional<int64_t> subgroupSize =
        getSubgroupSize(dmaOp->getParentOfType<FunctionOpInterface>());
    if (!subgroupSize.has_value()) {
      LLVM_DEBUG(LDBG() << "FAILED - unable to determine subgroup size\n");
      return rewriter.notifyMatchFailure(
          dmaOp, "unable to determine subgroup size from forall");
    }
    LLVM_DEBUG(LDBG() << "Subgroup size: " << *subgroupSize << "\n");

    // Check that transfer size matches one of the target DMA sizes (if
    // specified) Note: dma_sizes are in bits
    int64_t transferSizePerLane = transferSizeBits / *subgroupSize;
    LLVM_DEBUG({
      LDBG() << "Transfer size per lane: " << transferSizePerLane << " bits\n";
      LDBG() << "Target DMA sizes: [";
      for (int64_t dmaSize : targetDmaSizes) {
        llvm::dbgs() << dmaSize << " ";
      }
      llvm::dbgs() << "]\n";
    });

    if (!targetDmaSizes.empty() &&
        !matchesTargetDmaSize(transferSizePerLane, targetDmaSizes)) {
      LLVM_DEBUG(LDBG() << "FAILED - transfer size " << transferSizePerLane
                        << " does not match any target DMA size\n");
      return rewriter.notifyMatchFailure(
          dmaOp, "transfer size does not match any target DMA size");
    }
    LLVM_DEBUG(LDBG() << "Transfer size matches target DMA sizes\n");

    // TODO: Handle indices properly - for now we skip the explicit indices
    // check The lane parameter is always present but we handle it differently

    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    LLVM_DEBUG(LDBG() << "Source rank: " << sourceShape.size() << "\n");
    if (sourceShape.size() < 2) {
      LLVM_DEBUG(LDBG() << "FAILED - source rank < 2\n");
      return rewriter.notifyMatchFailure(
          dmaOp, "source must have at least 2 dimensions");
    }
    LLVM_DEBUG(LDBG() << "Source shape check passed\n");

    int64_t secondInnermostDimSize = sourceShape[sourceShape.size() - 2];
    LDBG() << "  Source second innermost dimension size: "
           << secondInnermostDimSize;

    int64_t elementsPerTransfer = innermostDimSize / *subgroupSize;

    auto transferType = VectorType::get({elementsPerTransfer}, elementType);

    rewriter.setInsertionPoint(dmaOp);

    auto laneId = dmaOp.getLane();

    auto laneOffset = arith::MulIOp::create(
        rewriter, loc, laneId,
        arith::ConstantIndexOp::create(rewriter, loc, elementsPerTransfer));

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (int64_t i = 0; i < secondInnermostDimSize; ++i) {
      Value iVal = arith::ConstantIndexOp::create(rewriter, loc, i);

      // Build source indices [i, %lane]
      SmallVector<Value> srcIndices;
      for (int64_t dim = 0; dim < sourceType.getRank() - 2; ++dim) {
        srcIndices.push_back(zero);
      }
      srcIndices.push_back(iVal);
      srcIndices.push_back(laneOffset);

      // Build dest indices [i, 0]
      SmallVector<Value> dstIndices;
      for (int64_t dim = 0; dim < destType.getRank() - 2; ++dim) {
        dstIndices.push_back(zero);
      }
      dstIndices.push_back(iVal);
      dstIndices.push_back(zero);

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
    LDBG() << "Total CoalescedGatherDMAOps found: " << count;
    if (count == 0) {
      LDBG() << "No CoalescedGatherDMAOps found, exiting early";
      return;
    }

    IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
    LDBG() << "GPU Target attribute: " << target;
    if (!target) {
      LDBG() << "Missing GPU target attribute, pass will fail";
      // Don't fail if no target attribute - just skip the pass
      return;
    }

    ArrayRef<int64_t> dmaSizes;
    if (auto dmaSizesAttr = target.getWgp().getDmaSizes()) {
      dmaSizes = dmaSizesAttr.asArrayRef();
    }
    // dma_sizes is optional - if not specified, skip the size validation

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, dmaSizes);

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};
} // namespace

} // namespace mlir::iree_compiler
