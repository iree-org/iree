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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-lower-coalesced-dma-to-global-loads"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERCOALESCEDDMATOGLOBALLOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

// Verifies that the mapping attributes match the expected pattern:
// [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>].
static LogicalResult verifyWarpMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    LDBG() << "No mapping attribute found";
    return failure();
  }
  if (mappingAttr->size() != 2) {
    LDBG() << "Mapping attribute size is " << mappingAttr->size()
           << ", expected 2";
    return failure();
  }

  auto warp0 = dyn_cast<gpu::GPUWarpMappingAttr>((*mappingAttr)[0]);
  auto warp1 = dyn_cast<gpu::GPUWarpMappingAttr>((*mappingAttr)[1]);
  if (!warp0 || !warp1) {
    LDBG() << "  - Mapping attributes are not GPUWarpMappingAttr";
    return failure();
  }

  if (warp0.getWarp() != gpu::MappingId::LinearDim1 ||
      warp1.getWarp() != gpu::MappingId::LinearDim0) {
    LDBG() << "  - Mapping mismatch (expected linear_dim_1, linear_dim_0)";
    return failure();
  }

  return success();
}

// Verifies that destination memref is contiguous and source/target memory
// address spaces are compatible.
static LogicalResult verifyMemoryLayout(IREE::GPU::CoalescedGatherDMAOp dmaOp) {
  // Check that destination memref is contiguous.
  auto destMemRefType = cast<MemRefType>(dmaOp.getInit().getType());
  LDBG() << "  - Destination type: " << destMemRefType;

  if (!destMemRefType.getLayout().isIdentity()) {
    LDBG() << "  - Layout is not identity, checking for strided layout";
    // Check if it's a strided layout with contiguous innermost dimension.
    auto stridedLayout =
        dyn_cast<StridedLayoutAttr>(destMemRefType.getLayout());
    if (!stridedLayout) {
      LDBG() << "  - Layout is not strided";
      return failure();
    }
    auto strides = stridedLayout.getStrides();
    if (strides.empty() || strides.back() != 1) {
      LDBG() << "  - Innermost stride is not 1";
      return failure();
    }
  }
  LDBG() << "  - Destination memref is contiguous";

  // Check memory address spaces.
  auto sourceType = cast<MemRefType>(dmaOp.getSource().getType());
  auto targetType = cast<MemRefType>(dmaOp.getInit().getType());

  bool hasGlobalSource = hasGlobalMemoryAddressSpace(sourceType);
  bool hasSharedTarget = hasSharedMemoryAddressSpace(targetType);

  if (!hasGlobalSource || !hasSharedTarget) {
    LDBG() << "-- Op: " << *dmaOp;
    LDBG() << "-- incompatible source or target memory address space.";
    return failure();
  }

  return success();
}

// Checks if a CoalescedGatherDMAOp is eligible for lowering to global loads:
// * the surrounding scf.forall must with 2D GPU warp mapping
// * The DMA op must be the only operation in the forall body
// * Destination memref must be contiguous
// * Source must be in global memory and destination in shared memory
static LogicalResult
isEligibleForGlobalDMA(IREE::GPU::CoalescedGatherDMAOp dmaOp) {
  scf::ForallOp forallOp = dmaOp->getParentOfType<scf::ForallOp>();

  // Verify that the forall has the required GPU warp mapping.
  if (failed(verifyWarpMapping(forallOp))) {
    return failure();
  }

  Block &body = forallOp.getRegion().front();

  // For now, the coalesced dma op must be the only op in the forall body
  // (excluding the terminator).
  if (body.getOperations().size() != 2) {
    LDBG() << "Forall body has " << body.getOperations().size()
           << " operations, expected 2 (dma + terminator)";
    return failure();
  }
  if (!isa<IREE::GPU::CoalescedGatherDMAOp>(body.front())) {
    LDBG() << "First operation is not a CoalescedGatherDMAOp";
    return failure();
  }

  if (failed(verifyMemoryLayout(dmaOp))) {
    return failure();
  }

  return success();
}

// Compute the number of global loads per thread based on the target's
// max load instruction size.
static int64_t getNumOfGlobalLoadsPerThread(MemRefType destType,
                                            int64_t subgroupSize,
                                            int64_t maxLoadInstructionBits) {
  // First, compute total number of bytes to load.
  int64_t elementSize = destType.getElementTypeBitWidth();
  int64_t totalBits = destType.getNumElements() * elementSize;

  // Then, compute number of bytes each thread should load.
  assert(totalBits % subgroupSize == 0 &&
         "Total bits must be divisible by subgroup size");
  int64_t bitsPerThread = totalBits / subgroupSize;

  assert(bitsPerThread % maxLoadInstructionBits == 0 &&
         "Bits per thread must be divisible by max load instruction bits");
  int64_t copiesPerThread = bitsPerThread / maxLoadInstructionBits;
  return copiesPerThread;
}

// Lowers iree_gpu.coalesced_gather_dma operations within scf.forall loops
// with warp mapping to amdgpu.gather_to_lds operations inside scf.for loops.
//
// In short, looks for patterns like this:
//   scf.forall (...) in (32, 1) {
//     %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest
//   } {mapping = [#gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0>]}
//
// And replaces with:
//   %subgroup_id = gpu.subgroup_id
//   %lane_id = gpu.lane_id
//   scf.for %iv = %c0 to %num_copies step %c1 {
//     %thread_index = affine.delinearize_index(%subgroup_id, ...)
//     %store_index = affine.delinearize_index(%lane_id, ...)
//     amdgpu.gather_to_lds %source[%thread_index] into %dest[%store_index]
//   }
//   gpu.barrier
struct LowerCoalescedGatherDMAPattern
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> workgroupSize,
                                 int64_t subgroupSize,
                                 int64_t maxLoadInstructionBits)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        workgroupSize(workgroupSize), subgroupSize(subgroupSize),
        maxLoadInstructionBits(maxLoadInstructionBits) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dmaOp.getLoc();

    LDBG() << "=== matchAndRewrite called for: " << dmaOp;

    // Check if this DMA op is eligible for transformation.
    if (failed(isEligibleForGlobalDMA(dmaOp))) {
      LDBG() << "Op is not eligible for transformation";
      return failure();
    }

    scf::ForallOp forallOp = dmaOp->getParentOfType<scf::ForallOp>();

    // Replace the entire scf.forall with amdgpu.gather_to_lds.
    rewriter.setInsertionPoint(forallOp);

    auto destMemRefType = dyn_cast<MemRefType>(dmaOp.getInit().getType());
    int64_t numCopiesPerThread = getNumOfGlobalLoadsPerThread(
        destMemRefType, subgroupSize, maxLoadInstructionBits);

    Value indices = dmaOp.getIndices();
    auto indicesType = cast<MemRefType>(indices.getType());

    Value dest = dmaOp.getInit();
    auto destType = cast<MemRefType>(dest.getType());

    Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(
        loc, rewriter.getIndexType(), nullptr);
    Value laneId =
        rewriter.create<gpu::LaneIdOp>(loc, rewriter.getIndexType(), nullptr);

    // Build a for loop skeleton.
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, numCopiesPerThread);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp forOp =
        rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // Now build the `scf.for` loop body.
    auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
      return rewriter
          .create<affine::AffineDelinearizeIndexOp>(loc, index, shape)
          .getMultiIndex();
    };

    auto getLinearizedGatherIndex = [&](Value sgIdVal, Value lIdVal,
                                        Value indVar) -> Value {
      int64_t numSubgroups = workgroupSize[0] / subgroupSize;
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      return rewriter.create<affine::AffineLinearizeIndexOp>(
          loc, ValueRange{sgIdVal, indVar, lIdVal, zero},
          ArrayRef<int64_t>{numSubgroups, numCopiesPerThread, subgroupSize,
                            maxLoadInstructionBits /
                                destType.getElementTypeBitWidth()},
          /*disjoint=*/true);
    };

    auto getSubgroupStoreBaseIndex = [&](Value indVar) -> Value {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      return getLinearizedGatherIndex(subgroupId, zero, indVar);
    };

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value inductionVar = forOp.getInductionVar();

      // Compute load address from `indices`.
      Value linearizedGatherIndices =
          getLinearizedGatherIndex(subgroupId, laneId, inductionVar);
      ValueRange delinearizedGlobalIndices =
          delinearizeIndex(linearizedGatherIndices, indicesType.getShape());

      // Load the actual thread gather index from `indices` memref.
      Value threadGatherIndex = rewriter.create<memref::LoadOp>(
          loc, indices, delinearizedGlobalIndices);

      // Compute the base index in dest memref.
      Value linearizedBaseIndices = getSubgroupStoreBaseIndex(inductionVar);
      ValueRange delinearizedLocalIndices =
          delinearizeIndex(linearizedBaseIndices, destType.getShape());

      // Determine the transfer type based on the destination element type and
      // the max load instruction size.
      int64_t elementBits = destType.getElementTypeBitWidth();
      int64_t elementsPerTransfer = maxLoadInstructionBits / elementBits;

      Type transferType;
      if (elementsPerTransfer == 1) {
        transferType = destType.getElementType();
      } else {
        transferType =
            VectorType::get({elementsPerTransfer}, destType.getElementType());
      }

      // Create the amdgpu.gather_to_lds operation.
      rewriter.create<amdgpu::GatherToLDSOp>(
          loc, dmaOp.getSource(), ValueRange{threadGatherIndex}, dest,
          delinearizedLocalIndices, TypeAttr::get(transferType));
    }

    rewriter.setInsertionPointAfter(forOp);
    rewriter.create<gpu::BarrierOp>(loc);

    rewriter.replaceOp(dmaOp, dmaOp.getInit());

    // Remove the scf.forall.
    rewriter.eraseOp(forallOp);
    return success();
  }

private:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
  int64_t maxLoadInstructionBits;
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

    std::optional<int32_t> maxLoadBits =
        target.getWgp().getMaxLoadInstructionBits();
    if (!maxLoadBits) {
      funcOp.emitOpError("Target does not specify max load instruction bits");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, *workgroupSize,
                                                 *subgroupSize, *maxLoadBits);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
