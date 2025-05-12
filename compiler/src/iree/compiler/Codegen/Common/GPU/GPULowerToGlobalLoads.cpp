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

#define DEBUG_TYPE "iree-codegen-gpu-lower-to-global-loads"

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
  SmallVector<Value> newShapeOfSlice;
  for (auto dim : newShape) {
    newShapeOfSlice.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, dim));
  }

  SmallVector<Value> offsets = {rewriter.create<arith::MulIOp>(
      loc, index, rewriter.create<arith::ConstantIndexOp>(loc, numParts))};

  SmallVector<int64_t> strides;
  for (auto i = tensorType.getRank() - 1; i >= 0; --i) {
    if (i == 0) {
      strides.push_back(1);
    } else {
      strides.push_back(tensorType.getDimSize(i));
    }
  }

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
  LLVM_DEBUG(llvm::dbgs() << "==== distributing op:\n");
  LLVM_DEBUG(llvm::dbgs() << "  " << *copy << "\n");

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

  LLVM_DEBUG(llvm::dbgs() << "-- elementsPerCopy: " << elementsPerCopy << "\n");
  LLVM_DEBUG(llvm::dbgs() << "-- workgroupSize: " << workgroupSize[0] << "\n");
  LLVM_DEBUG(llvm::dbgs() << "-- numSubgroups: " << numSubgroups << "\n");
  LLVM_DEBUG(llvm::dbgs() << "-- totalCopySize: " << totalCopySize << "\n");
  LLVM_DEBUG(llvm::dbgs() << "-- totalCopySizePerSubgroup: "
                          << totalCopySizePerSubgroup << "\n");
  LLVM_DEBUG(llvm::dbgs() << "-- numCopiesPerThread: " << numCopiesPerThread
                          << "\n");

  // build For loop
  SmallVector<Attribute> mapping;
  int idx = 0;
  for (int64_t i = 0, e = rank; i < e; ++i) {
    unsigned mappingId =
        static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
    mapping.push_back(gpu::GPUThreadMappingAttr::get(
        context, static_cast<gpu::MappingId>(mappingId)));
  }
  mapping = llvm::to_vector(llvm::reverse(mapping));

  Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, nullptr);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, nullptr);

  // TODO: make it multidimensional.

  // compute number of loads per thread
  auto sourceType = cast<MemRefType>(copy.getOperand(0).getType());
  MemRefType localType = cast<MemRefType>(copy.getOutputs().front().getType());

  auto getGlobalGatherIndex = [&](Value sgIdVal, Value lIdVal,
                                  Value indVar) -> Value {
    SmallVector<AffineExpr> symbols(3);
    bindSymbolsList(rewriter.getContext(), MutableArrayRef{symbols});
    auto laneIdExpr = symbols[0];
    auto subgroupIdExpr = symbols[1];
    auto iterationExpr = symbols[2];

    AffineExpr strideExpr =
        rewriter.getAffineConstantExpr(subgroupSize[0] * elementsPerCopy);
    AffineExpr totalCopySizeExpr =
        rewriter.getAffineConstantExpr(totalCopySizePerSubgroup);

    // [subgroupStartOffset + i * gatherStride + laneId]
    AffineExpr gatherOffsetExpr = subgroupIdExpr * totalCopySizeExpr +
                                  iterationExpr * strideExpr + laneIdExpr;
    OpFoldResult result = affine::makeComposedFoldedAffineApply(
        rewriter, loc, gatherOffsetExpr, {lIdVal, sgIdVal, indVar});
    return cast<Value>(result);
  };

  auto getSubgroupStoreBaseIndex = [&](Value sgIdVal, Value indVar) -> Value {
    SmallVector<AffineExpr> symbols(2);
    bindSymbolsList(rewriter.getContext(), MutableArrayRef{symbols});
    auto subgroupIdExpr = symbols[0];
    auto iterationExpr = symbols[1];

    AffineExpr strideExpr = rewriter.getAffineConstantExpr(subgroupSize[0]);
    AffineExpr totalCopySizeExpr =
        rewriter.getAffineConstantExpr(totalCopySizePerSubgroup);

    // [subgroupStartOffset + i * gatherStride]
    AffineExpr baseOffsetExpr =
        subgroupIdExpr * totalCopySizeExpr + iterationExpr * strideExpr;

    OpFoldResult result = affine::makeComposedFoldedAffineApply(
        rewriter, loc, baseOffsetExpr, {sgIdVal, indVar});
    return cast<Value>(result);
  };

  // Build for loop skeleton:
  scf::ForOp forOp = rewriter.create<scf::ForOp>(
      loc, /*lb = */ rewriter.create<arith::ConstantIndexOp>(loc, 0),
      /*ub = */
      rewriter.create<arith::ConstantIndexOp>(loc, numCopiesPerThread),
      /*steps = */
      rewriter.create<arith::ConstantIndexOp>(loc, 1),
      /*outputs=*/ValueRange{});

  auto delinearizeIndex = [&](Value index, ArrayRef<int64_t> shape) {
    return rewriter.create<affine::AffineDelinearizeIndexOp>(loc, index, shape)
        .getMultiIndex();
  };

  bool residualElements = totalCopySizePerSubgroup % subgroupSize[0];
  if (residualElements != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "-- has residualElements: " << residualElements << "\n");
    auto laneIdCmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, laneId,
        rewriter.create<arith::ConstantIndexOp>(loc, residualElements));
    // statically we know where to load the last elements from.
    rewriter.create<scf::IfOp>(
        loc, laneIdCmp, [&](OpBuilder &builder, Location loc) {
          auto numCopies =
              builder.create<arith::ConstantIndexOp>(loc, numCopiesPerThread);
          auto residualStoreBase =
              getSubgroupStoreBaseIndex(subgroupId, numCopies);
          auto delinearizedBaseIndices =
              delinearizeIndex(residualStoreBase, sourceType.getShape());
          auto laneGatherOffset =
              getGlobalGatherIndex(subgroupId, laneId, numCopies);
          auto delinearizedGatherIndices =
              delinearizeIndex(laneGatherOffset, sourceType.getShape());
          builder.create<IREE::GPU::GlobalLoadDMAOp>(
              loc, copy.getOperand(0), delinearizedGatherIndices,
              copy.getOutputs()[0], delinearizedBaseIndices);
          builder.create<scf::YieldOp>(loc);
        });
  }

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
    LLVM_DEBUG(llvm::dbgs()
               << "-- Op: " << *copy
               << "\n-- has source memory address space other than global.\n");
    return false;
  }
  if (targetType.getMemorySpace() !=
      gpu::AddressSpaceAttr::get(copy->getContext(),
                                 gpu::GPUDialect::getWorkgroupAddressSpace())) {
    LLVM_DEBUG(llvm::dbgs()
               << "-- Op: " << *copy
               << "\n-- has target memory address space other than workgroup.\n");
    return false;
  }
  return true;
}

namespace {
struct GPULowerToGlobalLoadsPass final
    : impl::GPULowerToGlobalLoadsPassBase<GPULowerToGlobalLoadsPass> {

  SmallVector<linalg::CopyOp> collectCopies(Operation *funcOp) {
    SmallVector<linalg::CopyOp> copies;
    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within thread/warp/lane foralls
    // to be skipped.
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
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
          LLVM_DEBUG(llvm::dbgs()
                     << "Skipping copy op: " << *copy
                     << " because it is not eligible for global load DMA.\n");
        }
      }
      return WalkResult::advance();
    });
    return copies;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // we will do this before/at the time of tiling, we need:
    // 1. tiling configurations, to compute how many to generate.
    SmallVector<linalg::CopyOp> copies = collectCopies(funcOp);

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
      // TODO: check that the copy's target memref is not a subview.
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
