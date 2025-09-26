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

static LogicalResult
isEligibleForGlobalDMA(IREE::GPU::CoalescedGatherDMAOp dmaOp) {
  LLVM_DEBUG(llvm::dbgs() << "Checking eligibility for: " << dmaOp << "\n");

  // check that the surrounding scf.forall:
  // 1. loop bounds are (subgroup_size, 1)
  // 2. mapping is [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
  auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();
  if (!forallOp) {
    LLVM_DEBUG(llvm::dbgs() << "  - Not in scf.forall\n");
    // If not in scf.forall, don't transform
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Found parent scf.forall\n");

  // Verify that the forall has the required loop bounds (32, 1)
  auto upperBounds = forallOp.getStaticUpperBound();
  LLVM_DEBUG(llvm::dbgs() << "  - Loop bounds: ");
  LLVM_DEBUG({
    for (auto bound : upperBounds) {
      llvm::dbgs() << bound << " ";
    }
    llvm::dbgs() << "\n";
  });
  if (upperBounds.size() != 2 || upperBounds[0] != 32 || upperBounds[1] != 1) {
    LLVM_DEBUG(llvm::dbgs() << "  - Loop bounds mismatch (expected 32, 1)\n");
    return failure();
  }

  // Verify that the forall has the required GPU thread mapping
  auto mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    LLVM_DEBUG(llvm::dbgs() << "  - No mapping attribute found\n");
    return failure();
  }
  if (mappingAttr->size() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "  - Mapping attribute size is "
                            << mappingAttr->size() << ", expected 2\n");
    return failure();
  }

  // Check that the mapping attributes match the expected pattern:
  // [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
  llvm::SmallString<64> mapping0Str, mapping1Str;
  llvm::raw_svector_ostream os0(mapping0Str), os1(mapping1Str);
  (*mappingAttr)[0].print(os0);
  (*mappingAttr)[1].print(os1);

  LLVM_DEBUG(llvm::dbgs() << "  - Mapping[0]: " << mapping0Str << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  - Mapping[1]: " << mapping1Str << "\n");

  if (mapping0Str != "#gpu.thread<linear_dim_1>" ||
      mapping1Str != "#gpu.thread<linear_dim_0>") {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Mapping mismatch (expected #gpu.thread<linear_dim_1>, "
                  "#gpu.thread<linear_dim_0>)\n");
    return failure();
  }

  // Check that the forall body contains a CoalescedGatherDMAOp
  // scf.forall has an implicit terminator (scf.forall.in_parallel)
  Block &body = forallOp.getRegion().front();
  LLVM_DEBUG(llvm::dbgs() << "  - Forall body has "
                          << body.getOperations().size() << " operations\n");

  // Print all operations for debugging
  LLVM_DEBUG({
    int opIdx = 0;
    for (auto &op : body.getOperations()) {
      llvm::dbgs() << "    Op[" << opIdx++ << "]: " << op.getName() << "\n";
    }
  });

  // The forall body should have exactly 2 ops: the DMA op and the terminator
  if (body.getOperations().size() != 2) {
    LLVM_DEBUG(llvm::dbgs() << "  - Expected exactly 2 operations in forall "
                               "body (DMA + terminator)\n");
    return failure();
  }

  // Verify the first operation is a CoalescedGatherDMAOp
  if (!isa<IREE::GPU::CoalescedGatherDMAOp>(body.front())) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - First operation is not a CoalescedGatherDMAOp\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Forall body check passed\n");

  // Check that destination memref is contiguous
  auto destMemRefType = llvm::dyn_cast<MemRefType>(dmaOp.getInit().getType());
  if (!destMemRefType) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Failed to cast destination to MemRefType\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Destination type: " << destMemRefType
                          << "\n");

  // For contiguity check, we need to ensure the innermost dimension has
  // stride 1
  if (!destMemRefType.getLayout().isIdentity()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Layout is not identity, checking for strided layout\n");
    // Check if it's a strided layout with contiguous innermost dimension
    auto stridedLayout =
        llvm::dyn_cast<StridedLayoutAttr>(destMemRefType.getLayout());
    if (!stridedLayout) {
      LLVM_DEBUG(llvm::dbgs() << "  - Layout is not strided\n");
      return failure();
    }
    auto strides = stridedLayout.getStrides();
    if (strides.empty() || strides.back() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "  - Innermost stride is not 1\n");
      return failure();
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "  - Destination memref is contiguous\n");

  // Check memory address spaces
  auto sourceType = llvm::dyn_cast<MemRefType>(dmaOp.getSource().getType());
  auto targetType = llvm::dyn_cast<MemRefType>(dmaOp.getInit().getType());
  if (!sourceType || !targetType) {
    LLVM_DEBUG(llvm::dbgs()
               << "  - Failed to cast source or target to MemRefType\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "  - Source type: " << sourceType << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  - Target type: " << targetType << "\n");

  bool hasGlobalSource = hasGlobalMemoryAddressSpace(sourceType);
  bool hasSharedTarget = hasSharedMemoryAddressSpace(targetType);
  LLVM_DEBUG(llvm::dbgs() << "  - Source has global memory: " << hasGlobalSource
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  - Target has shared memory: " << hasSharedTarget
                          << "\n");

  if (!hasGlobalSource || !hasSharedTarget) {
    LLVM_DEBUG(llvm::dbgs() << "-- Op: " << *dmaOp << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "-- incompatible source or target memory address space.\n");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "  - All eligibility checks passed!\n");
  return success();
}

constexpr int64_t kNumBitsPerCopy = 128;

// Here we assume each load is 128 bit, so
static int64_t getNumOfGlobalLoadsPerThread(MemRefType destType,
                                            int64_t subgroupSize) {
  // first, compute total number of bytes to load
  int64_t elementSize = destType.getElementTypeBitWidth();
  int64_t totalBits = destType.getNumElements() * elementSize;

  // Then, compute number of bytes each thread should load
  assert(totalBits % subgroupSize == 0 &&
         "total bits must be divisible by subgroup size");
  int64_t bitsPerThread = totalBits / subgroupSize;

  assert(bitsPerThread % kNumBitsPerCopy == 0 &&
         "bits per thread must be divisible by 128");
  int64_t copiesPerThread = bitsPerThread / kNumBitsPerCopy;
  return copiesPerThread;
}

struct LowerCoalescedGatherDMAPattern
    : public OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerCoalescedGatherDMAPattern(MLIRContext *context,
                                 ArrayRef<int64_t> workgroupSize,
                                 int64_t subgroupSize)
      : OpRewritePattern<IREE::GPU::CoalescedGatherDMAOp>(context),
        workgroupSize(workgroupSize), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(IREE::GPU::CoalescedGatherDMAOp dmaOp,
                                PatternRewriter &rewriter) const override {
    auto loc = dmaOp.getLoc();

    LLVM_DEBUG(llvm::dbgs()
               << "=== matchAndRewrite called for: " << dmaOp << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  workgroupSize: " << workgroupSize[0] << ", "
               << workgroupSize[1] << ", " << workgroupSize[2] << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  subgroupSize: " << subgroupSize << "\n");

    // Check if this DMA op is eligible for transformation
    if (failed(isEligibleForGlobalDMA(dmaOp))) {
      LLVM_DEBUG(llvm::dbgs() << "DMA op is not eligible for transformation\n");
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "DMA op is eligible for transformation\n");

    auto forallOp = dmaOp->getParentOfType<scf::ForallOp>();

    // Replace the entire scf.forall with the amdgpu.gather_to_lds operation
    rewriter.setInsertionPoint(forallOp);

    auto destMemRefType = llvm::dyn_cast<MemRefType>(dmaOp.getInit().getType());
    int64_t numCopiesPerThread =
        getNumOfGlobalLoadsPerThread(destMemRefType, subgroupSize);

    auto indices = dmaOp.getIndices();
    auto indicesType = llvm::cast<MemRefType>(indices.getType());

    auto dest = dmaOp.getInit();
    auto destType = llvm::cast<MemRefType>(dest.getType());

    Value subgroupId = gpu::SubgroupIdOp::create(rewriter, loc, nullptr);
    Value laneId = gpu::LaneIdOp::create(rewriter, loc, nullptr);

    // Build a for loop skeleton:
    auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto upperBound =
        arith::ConstantIndexOp::create(rewriter, loc, numCopiesPerThread);
    auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
    scf::ForOp forOp =
        scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);

    // Now build the `scf.for` loop body:
    auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
      return affine::AffineDelinearizeIndexOp::create(rewriter, loc, index,
                                                      shape)
          .getMultiIndex();
    };

    auto getLinearizedGatherIndex = [&](Value sgIdVal, Value lIdVal,
                                        Value indVar) -> Value {
      int64_t numSubgroups = workgroupSize[0] / subgroupSize;
      auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      return affine::AffineLinearizeIndexOp::create(
          rewriter, loc, ValueRange{sgIdVal, indVar, lIdVal, zero},
          ArrayRef<int64_t>{numSubgroups, numCopiesPerThread, subgroupSize,
                            kNumBitsPerCopy /
                                destType.getElementTypeBitWidth()},
          /*disjoint=*/true);
    };

    auto getSubgroupStoreBaseIndex = [&](Value indVar) -> Value {
      auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      return getLinearizedGatherIndex(subgroupId, zero, indVar);
    };

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      auto inductionVar = forOp.getInductionVar();

      // Compute load address from `indices`:
      Value linearizedGatherIndices =
          getLinearizedGatherIndex(subgroupId, laneId, inductionVar);
      ValueRange delinearizedGlobalIndices =
          delinearizeIndex(linearizedGatherIndices, indicesType.getShape());

      // Load the actual thread gather index from `indices` memref:
      Value threadGatherIndex = rewriter.create<memref::LoadOp>(
          loc, indices, delinearizedGlobalIndices);

      // Compute the base index in dest memref:
      Value linearizedBaseIndices = getSubgroupStoreBaseIndex(inductionVar);
      ValueRange delinearizedLocalIndices =
          delinearizeIndex(linearizedBaseIndices, destType.getShape());

      // Determine the transfer type based on the destination element type and
      // the number of bits per copy (128 bits)
      int64_t elementBits = destType.getElementTypeBitWidth();
      int64_t elementsPerTransfer = kNumBitsPerCopy / elementBits;

      Type transferType;
      if (elementsPerTransfer == 1) {
        transferType = destType.getElementType();
      } else {
        transferType =
            VectorType::get({elementsPerTransfer}, destType.getElementType());
      }

      // Create the amdgpu.gather_to_lds operation
      rewriter.create<amdgpu::GatherToLDSOp>(
          loc, dmaOp.getSource(), ValueRange{threadGatherIndex}, dest,
          delinearizedLocalIndices, TypeAttr::get(transferType));
    }

    // Insert a barrier after the loop
    rewriter.setInsertionPointAfter(forOp);
    rewriter.create<gpu::BarrierOp>(loc);

    // Replace the DMA op result with its init value (dest memref)
    rewriter.replaceOp(dmaOp, dmaOp.getInit());

    // Remove the scf.forall
    rewriter.eraseOp(forallOp);
    return success();
  }

private:
  ArrayRef<int64_t> workgroupSize;
  int64_t subgroupSize;
};

namespace {
struct GPULowerCoalescedDMAToGlobalLoadsPass final
    : impl::GPULowerCoalescedDMAToGlobalLoadsPassBase<
          GPULowerCoalescedDMAToGlobalLoadsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();

    // Debug: Print that the pass is running
    LLVM_DEBUG(llvm::dbgs()
               << "Running GPULowerCoalescedDMAToGlobalLoadsPass on "
               << funcOp.getName() << "\n");

    // Walk the function to see if there are any CoalescedGatherDMAOp ops
    int count = 0;
    funcOp.walk([&count](IREE::GPU::CoalescedGatherDMAOp op) {
      LLVM_DEBUG(llvm::dbgs() << "Found CoalescedGatherDMAOp: " << op << "\n");
      count++;
    });
    LLVM_DEBUG(llvm::dbgs()
               << "Total CoalescedGatherDMAOp found: " << count << "\n");

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

    LLVM_DEBUG(llvm::dbgs()
               << "Workgroup size: " << (*workgroupSize)[0] << ", "
               << (*workgroupSize)[1] << ", " << (*workgroupSize)[2] << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Subgroup size: " << *subgroupSize << "\n");

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerCoalescedGatherDMAPattern>(context, *workgroupSize,
                                                 *subgroupSize);

    LLVM_DEBUG(llvm::dbgs() << "Applying patterns...\n");

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern application failed\n");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "Pattern application succeeded\n");
  }
};
} // namespace

} // namespace mlir::iree_compiler
