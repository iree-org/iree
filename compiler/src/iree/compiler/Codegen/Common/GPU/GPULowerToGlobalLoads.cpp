// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/BuiltinAttributes.h"

#define DEBUG_TYPE "iree-codegen-gpu-lower-to-global-loads"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPULOWERTOGLOBALLOADSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
//====---------------------------------------------------------------------===//
// TODO: Move this to a common place.
//====---------------------------------------------------------------------===//

// For optimal performance we always want to copy 128 bits
static constexpr int kPreferredCopyNumBits = 128;

// Slice the tensor into numParts parts, and return the i-th part.
static std::optional<Value> sliceTensor(RewriterBase &rewriter, Value tensor,
                                        size_t numParts, ValueRange inductionVars) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto outermostDim = tensorType.getRank() - 1;
  auto outermostDimSize = tensorType.getDimSize(outermostDim);
  if (outermostDimSize % numParts != 0) {
    return std::nullopt;
  }

  auto loc = tensor.getLoc();

  // Create an extract slice
  SmallVector<int64_t> newShape(tensorType.getShape());
  newShape[outermostDim] = outermostDimSize / numParts;
  
  // hack:
  if (newShape[0] == 64) {
    // [64, 64]
    newShape = {256, 16};
  } else if (newShape[0] == 256) {

  }
  
  // turn newShape into ValueRange:
  SmallVector<Value> newShapeOfSlice;
    LLVM_DEBUG(llvm::dbgs() << "printing shapes dim for:\n");
  LLVM_DEBUG(llvm::dbgs() << "tensor: " << tensor << "\n");
  for (auto dim : newShape) {
    LLVM_DEBUG(
        llvm::dbgs() << "newShapeOfSlice dim: " << dim << "\n");
    newShapeOfSlice.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, dim));
  }
  
  // offset
  //SmallVector<Value> offsets(tensorType.getRank(),
  //                          rewriter.create<arith::ConstantIndexOp>(loc, 0));
  //offsets[outermostDim] = rewriter.create<arith::MulIOp>(loc, inductionVars[0],
  //    rewriter.create<arith::ConstantIndexOp>(loc, outermostDimSize / numParts));
  SmallVector<Value> offsets = {
      rewriter.create<arith::MulIOp>(
          loc, inductionVars[0],
          rewriter.create<arith::ConstantIndexOp>(loc, outermostDimSize / numParts))
  };

  // strides of 1:
  SmallVector<Value> stridesValue;
  // Hack:
  SmallVector<int64_t> strides = {64, 1};
  for (auto dim : strides) {
    stridesValue.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, dim));
  }
  
  auto newSliceType = RankedTensorType::get(newShape, tensorType.getElementType());
  auto noVals = ValueRange{};
  return rewriter.create<tensor::ExtractSliceOp>(
      loc, newSliceType, tensor, offsets, noVals, noVals,
      ArrayRef<int64_t>{INT64_MIN, 1}, newShape, strides);
}

static void distributeLinalgCopyToThreads(RewriterBase &rewriter,
                                          linalg::CopyOp copy,
                                          ArrayRef<int64_t> workgroupSize,
                                          ArrayRef<int64_t> subgroupSize) {
  Location loc = copy.getLoc();
  MLIRContext *context = rewriter.getContext();

  OpBuilder::InsertionGuard guard(rewriter);

  SmallVector<OpFoldResult> tileSizes;

  // Get the copy size:
  auto copyTensorType = cast<TensorType>(copy.getResult(0).getType());
  auto rank = copyTensorType.getRank();
  SmallVector<OpFoldResult> tileSize(rank - 1, rewriter.getIndexAttr(1));
  int64_t elementBitWidth = copyTensorType.getElementTypeBitWidth();
  
  // TODO: determine tile sizes
  tileSizes.push_back(rewriter.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));
  tileSizes.push_back(rewriter.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));

  // Construct a ForallOp:
  SmallVector<OpFoldResult> lowerBounds(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> upperBounds =
      tensor::getMixedSizes(rewriter, loc, copy->getOperand(0));

  SmallVector<Attribute> mapping;
  int idx = 0;
  for (int64_t i = 0, e = rank; i < e; ++i) {
    unsigned mappingId =
        static_cast<unsigned>(gpu::MappingId::LinearDim0) + idx++;
    mapping.push_back(gpu::GPUThreadMappingAttr::get(
        context, static_cast<gpu::MappingId>(mappingId)));
  }
  mapping = llvm::to_vector(llvm::reverse(mapping));

  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      copy.getLoc(), lowerBounds, upperBounds, tileSizes,
      /*outputs=*/ValueRange{copy.getOperand(0)}, /*mapping=*/rewriter.getArrayAttr(mapping));

  rewriter.setInsertionPointToStart(newForallOp.getBody());

  auto inductionVars = newForallOp.getInductionVars();

  // TODO: properly handle workgroup sizes and subgroup sizes.
  auto numParts = workgroupSize[0] / subgroupSize[0];

  auto globalSlice = sliceTensor(rewriter, copy.getOperand(0), numParts, inductionVars);
  //auto localSlice = sliceTensor(rewriter, copy.getResult(0), numParts, inductionVars);
  // create an empty tensor with same shape of globalSlice:
  auto localSliceType = RankedTensorType::get(
      cast<RankedTensorType>(globalSlice->getType()).getShape(),
      cast<RankedTensorType>(globalSlice->getType()).getElementType());
  auto localSlice = rewriter.create<tensor::EmptyOp>(
      loc, TypeRange{localSliceType}, ValueRange{});

  if (!globalSlice || !localSlice) {
    copy.emitOpError("failed to slice tensor");
  }

  // TODO: make it multidimensional.
  //Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, gpu::Dimension::x);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, nullptr);

  // compute number of loads per thread
  auto sliceType = cast<RankedTensorType>(globalSlice->getType());
  auto sliceSize = std::accumulate(
      sliceType.getShape().begin(), sliceType.getShape().end(), 1,
      std::multiplies<int64_t>());
  auto numLoads = sliceSize / ((elementBitWidth / 8) * subgroupSize[0]);
  
  Value localSliceValue = localSlice;
  Value gatherOffset = laneId;
  Value gatherStride = rewriter.create<arith::ConstantIndexOp>(
      loc, (elementBitWidth / 8) * subgroupSize[0]);
  for (int i = 0; i < numLoads; ++i) {
    //compute the offset
    auto delinearizedIndex =
        rewriter
            .create<affine::AffineDelinearizeIndexOp>(
                loc, gatherOffset, sliceType.getShape())
            .getMultiIndex();
    localSliceValue = rewriter.create<IREE::GPU::GlobalLoadDMAOp>(
        loc, sliceType, *globalSlice, localSliceValue, delinearizedIndex);
    if (i < numLoads - 1) {
      gatherOffset =
          rewriter.create<arith::AddIOp>(loc, gatherOffset, gatherStride);
    }
  }

  // get scf.forall.in_parallel:
  /*
  auto inParallel = newForallOp.getTerminator();
  rewriter.setInsertionPoint(inParallel.getBody(), inParallel.getBody()->begin());
  // create a store:
  rewriter.create<tensor::InsertSliceOp>(
      loc, localSliceValue, *globalSlice, inductionVars,
      ValueRange{rewriter.create<arith::ConstantIndexOp>(
          loc, sliceType.getDimSize(sliceType.getRank() - 1))});
  */
  rewriter.replaceOp(copy, newForallOp);
}

} // namespace

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
        copies.push_back(copy);
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
      rewriter.setInsertionPoint(copy);
      distributeLinalgCopyToThreads(
          rewriter, copy, *workgroupSize, *subgroupSize);
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
