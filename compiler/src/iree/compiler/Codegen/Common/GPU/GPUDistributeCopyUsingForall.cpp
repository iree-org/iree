// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute-shared-memory-copy"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDISTRIBUTECOPYUSINGFORALLPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
//====---------------------------------------------------------------------===//
// Pass to lower workgroup memory copy to distibuted
// transfer_read/transfer_write ops.
//====---------------------------------------------------------------------===//

// For optimal performance we always want to copy 128 bits
static constexpr int kPreferredCopyNumBits = 128;

// Moves the copy into a single threaded forall.
static void distributeCopyToSingleThread(RewriterBase &rewriter,
                                         memref::CopyOp copy) {
  SmallVector<Attribute> mapping = {gpu::GPUThreadMappingAttr::get(
      rewriter.getContext(), gpu::MappingId::LinearDim0)};
  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      copy.getLoc(), ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
      ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
      /*outputs=*/ValueRange(), /*mapping=*/rewriter.getArrayAttr(mapping));
  rewriter.moveOpBefore(copy, newForallOp.getBody(),
                        newForallOp.getBody()->begin());
}

/// Distributes a copy with a thread mapping.
static void distributeCopyToThreads(RewriterBase &rewriter, memref::CopyOp copy,
                                    ArrayRef<OpFoldResult> tileSizes) {
  int64_t rank = tileSizes.size();
  assert(rank == copy.getTarget().getType().getRank() &&
         "tile size and copy rank mismatch");
  if (rank == 0) {
    distributeCopyToSingleThread(rewriter, copy);
    return;
  }

  Location loc = copy.getLoc();
  MLIRContext *context = rewriter.getContext();

  SmallVector<OpFoldResult> lowerBounds(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> upperBounds =
      memref::getMixedSizes(rewriter, loc, copy.getSource());

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
      /*outputs=*/ValueRange(), /*mapping=*/rewriter.getArrayAttr(mapping));

  rewriter.setInsertionPointToStart(newForallOp.getBody());

  AffineExpr d0, d1, d2;
  bindDims(context, d0, d1, d2);
  SmallVector<OpFoldResult> sizes;
  AffineMap minMap =
      AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {d0, d1 - d2}, context);
  for (auto [ub, tileSize, iterator] : llvm::zip_equal(
           upperBounds, tileSizes, newForallOp.getInductionVars())) {
    std::optional<int64_t> staticUb = getConstantIntValue(ub);
    std::optional<int64_t> staticTileSize = getConstantIntValue(tileSize);
    if ((staticUb && staticTileSize &&
         staticUb.value() % staticTileSize.value() == 0) ||
        (staticTileSize.value_or(0) == 1)) {
      sizes.push_back(tileSize);
    } else {
      sizes.push_back(
          rewriter
              .create<affine::AffineMinOp>(
                  loc, rewriter.getIndexType(), minMap,
                  ValueRange{
                      getValueOrCreateConstantIndexOp(rewriter, loc, tileSize),
                      getValueOrCreateConstantIndexOp(rewriter, loc, ub),
                      iterator})
              .getResult());
    }
  }

  SmallVector<OpFoldResult> offsets =
      getAsOpFoldResult(newForallOp.getInductionVars());
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  Value sourceTile = rewriter.create<memref::SubViewOp>(
      loc, copy.getSource(), offsets, sizes, strides);
  Value targetTile = rewriter.create<memref::SubViewOp>(
      loc, copy.getTarget(), offsets, sizes, strides);
  rewriter.replaceOpWithNewOp<memref::CopyOp>(copy, sourceTile, targetTile);
}

static SmallVector<OpFoldResult> getCopyTileSizes(Builder &b,
                                                  memref::CopyOp copy) {
  int64_t rank = copy.getTarget().getType().getRank();
  if (rank == 0) {
    return {};
  }

  SmallVector<OpFoldResult> tileSizes(rank - 1, b.getIndexAttr(1));
  int64_t elementBitWidth = llvm::cast<MemRefType>(copy.getTarget().getType())
                                .getElementTypeBitWidth();
  tileSizes.push_back(b.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));
  return tileSizes;
}

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
  auto newSliceType = RankedTensorType::get(newShape, tensorType.getElementType());
  // turn newShape into ValueRange:
  SmallVector<Value> newShapeOfSlice;
  for (auto dim : newShape) {
    newShapeOfSlice.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
  }
  
  // offset
  SmallVector<Value> offset(tensorType.getRank(),
                            rewriter.create<arith::ConstantIndexOp>(loc, 0));
  offset[outermostDim] = rewriter.create<arith::MulIOp>(loc, inductionVars[0],
      rewriter.create<arith::ConstantIndexOp>(loc, outermostDimSize / numParts));

  // strides of 1:
  SmallVector<Value> strides(tensorType.getRank(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 1));

  return rewriter.create<tensor::ExtractSliceOp>(
      loc, newSliceType, tensor, offset, newShapeOfSlice, strides);
}

static bool distributeLinalgCopyToThreads(RewriterBase &rewriter,
                                          linalg::CopyOp copy,
                                          ArrayRef<int64_t> workgroupSize,
                                          ArrayRef<int64_t> subgroupSize) {
  Location loc = copy.getLoc();
  MLIRContext *context = rewriter.getContext();

  SmallVector<OpFoldResult> tileSizes;

  // Get the copy size:
  auto copyTensorType = cast<TensorType>(copy.getResult(0).getType());
  auto rank = copyTensorType.getRank();
  SmallVector<OpFoldResult> tileSize(rank - 1, rewriter.getIndexAttr(1));
  int64_t elementBitWidth = copyTensorType.getElementTypeBitWidth();
  tileSizes.push_back(rewriter.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));

  // Construct a ForallOp:
  SmallVector<OpFoldResult> lowerBounds(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> upperBounds =
      memref::getMixedSizes(rewriter, loc, copy->getOperand(0));

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
      /*outputs=*/ValueRange(), /*mapping=*/rewriter.getArrayAttr(mapping));
  rewriter.setInsertionPointToStart(newForallOp.getBody());

  auto inductionVars = newForallOp.getInductionVars();

  // TODO: properly handle workgroup sizes and subgroup sizes.
  auto numParts = workgroupSize[0] * subgroupSize[0];

  auto globalSlice = sliceTensor(rewriter, copy.getOperand(0), numParts, inductionVars);
  auto localSlice = sliceTensor(rewriter, copy.getResult(0), numParts, inductionVars);
  if (!globalSlice || !localSlice) {
    copy.emitOpError("failed to slice tensor");
  }

  // TODO: make it multidimensional.
  //Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, gpu::Dimension::x);
  //Value laneId = rewriter.create<gpu::LaneIdOp>(loc, gpu::Dimension::x);

  // compute number of loads per thread
  auto sliceType = cast<RankedTensorType>(globalSlice->getType());
  auto sliceSize = std::accumulate(
      sliceType.getShape().begin(), sliceType.getShape().end(), 1,
      std::multiplies<int64_t>());
  auto numLoads = sliceSize / ((elementBitWidth / 8) * subgroupSize[0]);
  
  for (int i = 0; i < numLoads; ++i) {
  }

  rewriter.create<scf::YieldOp>(loc, *localSlice);
  return true;
}

} // namespace

namespace {
struct GPUDistributeCopyUsingForallPass final
    : impl::GPUDistributeCopyUsingForallPassBase<
          GPUDistributeCopyUsingForallPass> {
  void runOnOperation2() {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    
    // we will do this before/at the time of tiling, we need:
    // 1. tiling configurations, to compute how many to generate.
    //
    SmallVector<linalg::CopyOp> copies;
    funcOp.walk([&](linalg::CopyOp copy) { copies.push_back(copy); });

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
      bool success = distributeLinalgCopyToThreads(
          rewriter, copy, *workgroupSize, *subgroupSize);
      if (!success) {
        copy.emitOpError("failed to distribute copy");
        return signalPassFailure();
      }
    }

  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    SmallVector<memref::CopyOp> copies;

    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within thread/warp/lane foralls
    // to be skipped.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        // Skip ops contained within forall ops with thread/warp/lane mappings.
        if (forallOpHasMappingType<IREE::GPU::LaneIdAttr,
                                   gpu::GPUWarpMappingAttr,
                                   gpu::GPUThreadMappingAttr>(forallOp)) {
          return WalkResult::skip();
        }
      }
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        copies.push_back(copy);
      }
      return WalkResult::advance();
    });

    IRRewriter rewriter(context);
    for (auto copy : copies) {
      rewriter.setInsertionPoint(copy);
      SmallVector<OpFoldResult> tileSizes = getCopyTileSizes(rewriter, copy);
      distributeCopyToThreads(rewriter, copy, tileSizes);
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
