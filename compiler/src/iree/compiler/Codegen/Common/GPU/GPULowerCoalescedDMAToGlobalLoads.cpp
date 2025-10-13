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

// Verifies that the mapping attributes match the expected pattern:
// [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>].
static LogicalResult verifyThreadMapping(scf::ForallOp forallOp) {
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

  LDBG() << "  - Mapping[0]: " << (*mappingAttr)[0];
  LDBG() << "  - Mapping[1]: " << (*mappingAttr)[1];
  auto thread0 = dyn_cast<gpu::GPUThreadMappingAttr>((*mappingAttr)[0]);
  auto thread1 = dyn_cast<gpu::GPUThreadMappingAttr>((*mappingAttr)[1]);
  if (!thread0 || !thread1) {
    LDBG() << "  - Mapping attributes are not GPUThreadMappingAttr";
    return failure();
  }

  if (thread0.getThread() != gpu::MappingId::LinearDim1 ||
      thread1.getThread() != gpu::MappingId::LinearDim0) {
    LDBG() << "  - Mapping mismatch (expected linear_dim_1, linear_dim_0)";
    LDBG() << "    thread0: " << thread0.getThread()
           << " thread1: " << thread1.getThread();
    return failure();
  }

  LDBG() << "  - Thread mapping verified successfully";
  return success();
}

// Verifies that destination memref is contiguous and source/target memory
// address spaces are compatible.
static LogicalResult verifyMemoryLayout(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                        PatternRewriter &rewriter) {
  // Check that destination memref is contiguous.
  auto destMemRefType = cast<MemRefType>(dmaOp.getInit().getType());
  LDBG() << "  - Destination type: " << destMemRefType;

  if (!destMemRefType.areTrailingDimsContiguous(1)) {
    return rewriter.notifyMatchFailure(
        dmaOp,
        "destination memref does not have contiguous trailing dimension");
  }

  // Check memory address spaces.
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

// Checks if a CoalescedGatherDMAOp is eligible for lowering to global loads:
// * the surrounding scf.forall must with 2D GPU thread mapping
// * The DMA op must be the only operation in the forall body
// * Destination memref must be contiguous
// * Source must be in global memory and destination in shared memory
static LogicalResult
isEligibleForGlobalDMA(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                       PatternRewriter &rewriter) {
  scf::ForallOp forallOp = dmaOp->getParentOfType<scf::ForallOp>();

  // Verify that the forall has the required GPU thread mapping.
  if (failed(verifyThreadMapping(forallOp))) {
    return failure();
  }

  Block &body = forallOp.getRegion().front();

  // For now, the coalesced dma op must be the only op in the forall body
  // (excluding the terminator).
  if (body.getOperations().size() != 2) {
    return rewriter.notifyMatchFailure(
        dmaOp, "forall body must have exactly 2 operations (dma + terminator)");
  }
  if (!isa<IREE::GPU::CoalescedGatherDMAOp>(body.front())) {
    return rewriter.notifyMatchFailure(
        dmaOp, "first operation in forall body is not a CoalescedGatherDMAOp");
  }

  if (failed(verifyMemoryLayout(dmaOp, rewriter))) {
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

  int64_t copiesPerThread = bitsPerThread / maxLoadInstructionBits;
  return copiesPerThread;
}

// Lowers iree_gpu.coalesced_gather_dma operations within scf.forall loops
// with thread mapping to amdgpu.gather_to_lds operations inside scf.for loops.
//
// In short, looks for patterns like this:
//   scf.forall (...) in (32, 1) {
//     %1 = iree_gpu.coalesced_gather_dma %indices, %source into %dest
//   } {mapping = [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]}
//
// And replaces the DMA op with:
//   %lane_id = gpu.lane_id
//   scf.for %iv = %c0 to %num_copies step %c1 {
//     %thread_index = affine.delinearize_index(%subgroup_id, ...)
//     %store_index = affine.delinearize_index(%lane_id, ...)
//     amdgpu.gather_to_lds %source[%thread_index] into %dest[%store_index]
//   }
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
    LDBG() << "Checking coalesced_gather_dma op: " << dmaOp;

    // Find the parent forall op
    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
    if (!forallOp) {
      return rewriter.notifyMatchFailure(
          dmaOp, "coalesced_gather_dma not inside scf.forall");
    }

    // Verify that the forall has the required GPU thread mapping.
    if (failed(verifyThreadMapping(forallOp))) {
      return failure();
    }

    // Ensure this is not nested inside another thread-mapped forall
    if (forallOp->getParentOfType<scf::ForallOp>()) {
      auto parentForall = forallOp->getParentOfType<scf::ForallOp>();
      if (parentForall.getMapping()) {
        auto parentMapping = *parentForall.getMapping();
        if (!parentMapping.empty() &&
            isa<gpu::GPUThreadMappingAttr>(parentMapping[0])) {
          return rewriter.notifyMatchFailure(
              forallOp, "nested thread-mapped forall not supported");
        }
      }
    }

    // Verify memory layout
    if (failed(verifyMemoryLayout(dmaOp, rewriter))) {
      LDBG() << "  - Memory layout verification failed";
      return failure();
    }

    LDBG() << "  - Memory layout verified, starting transformation";
    Location loc = dmaOp.getLoc();

    // Set insertion point at the dma op to create replacement ops
    rewriter.setInsertionPoint(dmaOp);

    auto destMemRefType = dyn_cast<MemRefType>(dmaOp.getInit().getType());
    int64_t numCopiesPerThread = getNumOfGlobalLoadsPerThread(
        destMemRefType, subgroupSize, maxLoadInstructionBits);

    Value indices = dmaOp.getIndices();
    Value dest = dmaOp.getInit();
    Value source = dmaOp.getSource();
    auto indicesType = cast<MemRefType>(indices.getType());
    auto destType = cast<MemRefType>(dest.getType());

    auto sourceType = cast<MemRefType>(source.getType());
    if (!hasGlobalMemoryAddressSpace(sourceType)) {
      return rewriter.notifyMatchFailure(
          dmaOp, "source must have global memory address space for "
                 "amdgpu.gather_to_lds");
    }

    int64_t numIndices = indicesType.getNumElements();

    // Account for parallelism: multiply numIndices by the total number of
    // parallel iterations in the forall loop.
    auto staticUpperBound = forallOp.getStaticUpperBound();
    int64_t parallelIterations = 1;
    for (int64_t bound : staticUpperBound) {
      parallelIterations *= bound;
    }
    numIndices *= parallelIterations;

    int64_t totalBytesOfCopy =
        (destType.getNumElements() * destType.getElementTypeBitWidth()) / 8;
    LDBG() << "  - numIndices: " << numIndices
           << ", totalBytesOfCopy: " << totalBytesOfCopy;
    if (totalBytesOfCopy % numIndices != 0) {
      return rewriter.notifyMatchFailure(
          dmaOp, "total bytes to copy is not divisible by number of indices");
    }

    int64_t bytesPerIndexCopy = totalBytesOfCopy / numIndices;
    LDBG() << "  - bytesPerIndexCopy: " << bytesPerIndexCopy;

    numCopiesPerThread = numIndices / subgroupSize;

    // TODO: change this to a better query.
    if (!dmaOp.isLegalLoadWidthInBytes(bytesPerIndexCopy)) {
      return rewriter.notifyMatchFailure(
          dmaOp,
          "bytes per index copy does not match max load instruction bits");
    }

    Value laneId =
        rewriter.create<gpu::LaneIdOp>(loc, rewriter.getIndexType(), nullptr);

    // Use the subgroup size immediate value from the config.
    Value subgroupSizeValue =
        rewriter.create<arith::ConstantIndexOp>(loc, this->subgroupSize);

    // Build a for loop skeleton.
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, numCopiesPerThread);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    scf::ForOp forOp =
        rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

    // helper functions
    auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
      return rewriter
          .create<affine::AffineDelinearizeIndexOp>(loc, index, shape)
          .getMultiIndex();
    };

    // Determine the transfer type based on the destination element type and
    // the max load instruction size.
    int64_t bytesPerElement = destType.getElementTypeBitWidth() / 8;
    int64_t elementsPerTransfer = bytesPerIndexCopy / bytesPerElement;
    Type transferType =
        VectorType::get({elementsPerTransfer}, destType.getElementType());

    // Build the loop body:
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value inductionVar = forOp.getInductionVar();

      Value linearizedBaseIndex =
          rewriter.create<arith::MulIOp>(loc, inductionVar, subgroupSizeValue);

      // Compute the base index in dest memref.
      ValueRange delinearizedBaseIndices =
          delinearizeIndex(linearizedBaseIndex, destType.getShape());

      // Compute load address from `indices`.
      Value linearizedGatherIndex = rewriter.create<arith::AddIOp>(
          loc, laneId,
          linearizedBaseIndex); // copy_id * subgroup
      ValueRange delinearizedGlobalIndices =
          delinearizeIndex(linearizedGatherIndex, indicesType.getShape());

      // Load the actual thread gather index from `indices` memref.
      Value threadGatherIndex = rewriter.create<memref::LoadOp>(
          loc, indices, delinearizedGlobalIndices);

      // Create the amdgpu.gather_to_lds operation.
      rewriter.create<amdgpu::GatherToLDSOp>(
          loc, source, ValueRange{threadGatherIndex}, dest,
          delinearizedBaseIndices, TypeAttr::get(transferType));
    }

    // Erase the DMA op since we've replaced it with the for loop
    rewriter.eraseOp(dmaOp);
    LDBG() << "  - Transformation completed successfully";
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

    walkAndApplyPatterns(funcOp, std::move(patterns));
  }
};
} // namespace

} // namespace mlir::iree_compiler
