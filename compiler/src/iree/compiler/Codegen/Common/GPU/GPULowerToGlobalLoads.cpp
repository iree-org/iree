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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

#define DEBUG_TYPE "iree-llvmgpu-lower-to-global-loads"

#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOGLOBALLOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
//====---------------------------------------------------------------------===//
// TODO: Move this to a common place.
//====---------------------------------------------------------------------===//

static constexpr int kNumBitsPerCopy = 32;

// Slice the tensor into numParts parts, and return the i-th part.
static std::optional<Value> sliceTensor(RewriterBase &rewriter, Value tensor,
                                        size_t numParts, Value index) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto outermostDimSize = tensorType.getDimSize(0);
  if (outermostDimSize % numParts != 0) {
    return std::nullopt;
  }

  // Create an extract slice
  SmallVector<int64_t> newShape(tensorType.getShape());
  newShape[0] = outermostDimSize / numParts;

  auto loc = tensor.getLoc();

  SmallVector<Value> offsets = {rewriter.create<arith::MulIOp>(
      loc, index, rewriter.create<arith::ConstantIndexOp>(loc, numParts))};

  SmallVector<int64_t> strides =
      llvm::to_vector(llvm::reverse(tensorType.getShape().drop_back()));
  strides.push_back(1);

  auto newSliceType =
      RankedTensorType::get(newShape, tensorType.getElementType());
  auto noVals = ValueRange{};
  return rewriter.create<tensor::ExtractSliceOp>(
      loc, newSliceType, tensor, offsets, noVals, noVals,
      /*static_offset =*/ArrayRef<int64_t>{INT64_MIN, 0},
      /*static_shape =*/newShape, /*static_strides =*/strides);
}

static std::optional<int64_t> getMemRefTypeNumElements(MemRefType memRefType) {
  auto shape = memRefType.getShape();
  if (shape.empty()) {
    return std::nullopt;
  }
  int64_t numElements = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return std::nullopt;
    }
    numElements *= dim;
  }
  return numElements;
}

static bool distributeLinalgCopyToThreads(RewriterBase &rewriter,
                                          linalg::CopyOp copy,
                                          ArrayRef<int64_t> workgroupSize,
                                          ArrayRef<int64_t> subgroupSize) {
  LDBG("==== distributing op:\n"
       << "\t" << *copy << "\n");

  Location loc = copy.getLoc();
  MLIRContext *context = rewriter.getContext();

  OpBuilder::InsertionGuard guard(rewriter);

  auto getTotalSize = [](ArrayRef<int64_t> sizes) {
    return std::accumulate(sizes.begin(), sizes.end(), 1,
                           std::multiplies<int64_t>());
  };

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
    copy.emitOpError("Cannot proceed: copy to non-static or non-contiguous, "
                     "non-row major memref.");
    return false;
  }
  auto rank = copyMemRefType.getRank();
  SmallVector<OpFoldResult> tileSize(rank - 1, rewriter.getIndexAttr(1));

  size_t elementBitWidth = copyMemRefType.getElementTypeBitWidth();
  if (kNumBitsPerCopy % elementBitWidth != 0) {
    copy.emitOpError("Cannot proceed: preferred copy size is not a multiple of "
                     "element bit width.");
    return false;
  }
  size_t elementsPerCopy = kNumBitsPerCopy / elementBitWidth;

  // divide the copy by subgroup:
  assert(subgroupSize.size() == 1); // only 1-D
  assert(workgroupSize[0] % subgroupSize[0] == 0);

  auto numSubgroups = workgroupSize[0] / subgroupSize[0];
  auto totalCopySize = getTotalSize(copyMemRefType.getShape());
  auto totalCopySizePerSubgroup = totalCopySize / numSubgroups;
  auto numCopiesPerThread =
      (totalCopySizePerSubgroup / elementsPerCopy) / subgroupSize[0];
  auto residualElements =
      totalCopySizePerSubgroup % (subgroupSize[0] * elementsPerCopy);

  LDBG("-- elementsPerCopy: " << elementsPerCopy << "\n");
  LDBG("-- workgroupSize: " << workgroupSize[0] << "\n");
  LDBG("-- numSubgroups: " << numSubgroups << "\n");
  LDBG("-- totalCopySize: " << totalCopySize << "\n");
  LDBG("-- totalCopySizePerSubgroup: " << totalCopySizePerSubgroup << "\n");
  LDBG("-- numCopiesPerThread: " << numCopiesPerThread << "\n");
  LDBG("-- residualElements: " << residualElements << "\n");

  if (residualElements != 0) {
    copy.emitOpError(
        "Cannot proceed: Cannot handle copying residual elements.");
    return false;
  }

  // build For loop
  SmallVector<Attribute> mapping;
  for (int64_t i = 0, e = rank; i < e; ++i) {
    unsigned mappingId = static_cast<unsigned>(gpu::MappingId::LinearDim0) + i;
    mapping.push_back(gpu::GPUThreadMappingAttr::get(
        context, static_cast<gpu::MappingId>(mappingId)));
  }
  std::reverse(mapping.begin(), mapping.end());

  Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, nullptr);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, nullptr);

  // TODO: make it multidimensional.

  // compute number of loads per thread
  auto sourceType = cast<MemRefType>(copy.getOperand(0).getType());
  MemRefType localType = cast<MemRefType>(copy.getOutputs().front().getType());

  auto getGlobalGatherIndex = [&](Value sgIdVal, Value lIdVal,
                                  Value indVar) -> Value {
    auto laneIdExpr = rewriter.getAffineSymbolExpr(0);
    auto subgroupIdExpr = rewriter.getAffineSymbolExpr(1);
    auto iterationExpr = rewriter.getAffineSymbolExpr(2);

    AffineExpr strideExpr =
        rewriter.getAffineConstantExpr(subgroupSize[0] * elementsPerCopy);
    AffineExpr totalCopySizeExpr =
        rewriter.getAffineConstantExpr(totalCopySizePerSubgroup);

    // [subgroupStartOffset + i * gatherStride + laneId * elementsPerCopy]
    AffineExpr gatherOffsetExpr = subgroupIdExpr * totalCopySizeExpr +
                                  iterationExpr * strideExpr +
                                  laneIdExpr * elementsPerCopy;
    return affine::makeComposedAffineApply(rewriter, loc, gatherOffsetExpr,
                                           {lIdVal, sgIdVal, indVar})
        .getResult();
  };

  auto getSubgroupStoreBaseIndex = [&](Value sgIdVal, Value indVar) -> Value {
    auto subgroupIdExpr = rewriter.getAffineSymbolExpr(0);
    auto iterationExpr = rewriter.getAffineSymbolExpr(1);

    AffineExpr strideExpr =
        rewriter.getAffineConstantExpr(subgroupSize[0] * elementsPerCopy);
    AffineExpr totalCopySizeExpr =
        rewriter.getAffineConstantExpr(totalCopySizePerSubgroup);

    // [subgroupStartOffset + i * gatherStride]
    AffineExpr baseOffsetExpr =
        subgroupIdExpr * totalCopySizeExpr + iterationExpr * strideExpr;
    return affine::makeComposedAffineApply(rewriter, loc, baseOffsetExpr,
                                           {sgIdVal, indVar})
        .getResult();
  };

  // Build for loop skeleton:
  scf::ForOp forOp = rewriter.create<scf::ForOp>(
      loc, /*lb=*/rewriter.create<arith::ConstantIndexOp>(loc, 0),
      /*ub=*/rewriter.create<arith::ConstantIndexOp>(loc, numCopiesPerThread),
      /*steps=*/rewriter.create<arith::ConstantIndexOp>(loc, 1));

  auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
    return rewriter.create<affine::AffineDelinearizeIndexOp>(loc, index, shape)
        .getMultiIndex();
  };

  // sync at the end of the loop across threads
  rewriter.create<gpu::BarrierOp>(loc);

  // For loop body:
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    auto inductionVar = forOp.getInductionVar();
    auto linearizedGatherIndices =
        getGlobalGatherIndex(subgroupId, laneId, inductionVar);
    auto delinearizedGlobalIndices =
        delinearizeIndex(linearizedGatherIndices, sourceType.getShape());
    auto linearizedBaseIndices =
        getSubgroupStoreBaseIndex(subgroupId, inductionVar);
    auto delinearizedLocalIndices =
        delinearizeIndex(linearizedBaseIndices, localType.getShape());
    rewriter.create<IREE::GPU::GlobalLoadDMAOp>(
        loc, copy.getOperand(0), delinearizedGlobalIndices,
        copy.getOutputs()[0], delinearizedLocalIndices);
  }

  copy->erase();
  return true;
}

} // namespace

static bool checkEligibilityForGlobalLoadDMA(linalg::CopyOp copy) {
  // source must be global address and target must be workgroup address.
  auto sourceType = cast<MemRefType>(copy.getOperand(0).getType());
  auto targetType = cast<MemRefType>(copy.getOutputs().front().getType());
  if (!hasGlobalMemoryAddressSpace(sourceType)) {
    LDBG("-- Op: "
         << *copy
         << "\n-- has source memory address space other than global.\n");
    return false;
  }
  if (targetType.getMemorySpace() !=
      gpu::AddressSpaceAttr::get(copy->getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace())) {
    LDBG("-- Op: "
         << *copy
         << "\n-- has target memory address space other than workgroup.\n");
    return false;
  }
  // TODO: check that the copy's target memref is not a subview.
  return true;
}

namespace {
struct GPULowerToGlobalLoadsPass final
    : impl::GPULowerToGlobalLoadsPassBase<GPULowerToGlobalLoadsPass> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // we will do this before/at the time of tiling, we need:
    // 1. tiling configurations, to compute how many to generate.

    SmallVector<linalg::CopyOp> copies;
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Walk in PreOrder so that parent operations are visited before children,
      // thus allowing all operations contained within thread/warp/lane foralls
      // to be skipped.
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        // Skip ops contained within forall ops with thread/warp/lane mappings.
        if (forallOpHasMappingType<IREE::GPU::LaneIdAttr,
                                   gpu::GPUWarpMappingAttr,
                                   gpu::GPUThreadMappingAttr>(forallOp)) {
          return WalkResult::skip();
        }
      }
      if (auto copy = dyn_cast<linalg::CopyOp>(op)) {
        if (checkEligibilityForGlobalLoadDMA(copy)) {
          copies.push_back(copy);
        } else {
          LDBG("Skipping copy op: "
               << *copy
               << " because it is not eligible for global load DMA.\n");
        }
      }
      return WalkResult::advance();
    });

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

    IRRewriter rewriter(context);
    for (auto copy : copies) {
      rewriter.setInsertionPoint(copy);
      if (!distributeLinalgCopyToThreads(rewriter, copy, *workgroupSize,
                                         *subgroupSize)) {
        copy.emitOpError("failed to lower linalg.copy to global loads");
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
