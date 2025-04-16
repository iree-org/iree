// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
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

  // strides of 1:
  SmallVector<Value> stridesValue;
  // Hack:
  SmallVector<int64_t> strides = {64, 1};
  for (auto dim : strides) {
    stridesValue.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
  }

  auto newSliceType =
      RankedTensorType::get(newShape, tensorType.getElementType());
  auto noVals = ValueRange{};
  return rewriter.create<tensor::ExtractSliceOp>(
      loc, newSliceType, tensor, offsets, noVals, noVals,
      /*static_offset =*/ArrayRef<int64_t>{INT64_MIN, 0},
      /*static_shape =*/newShape, /*static_strides =*/strides);
}

static void distributeLinalgCopyToThreads(RewriterBase &rewriter,
                                          linalg::CopyOp copy,
                                          ArrayRef<int64_t> workgroupSize,
                                          ArrayRef<int64_t> subgroupSize) {
  Location loc = copy.getLoc();
  MLIRContext *context = rewriter.getContext();

  OpBuilder::InsertionGuard guard(rewriter);

  auto getTotalSize = [](ArrayRef<int64_t> sizes) {
    return std::accumulate(sizes.begin(), sizes.end(), 1,
                           std::multiplies<int64_t>());
  };

  auto totalWorkgroupSize = getTotalSize(workgroupSize);

  // We should only have 1 D tile?
  SmallVector<OpFoldResult> tileSizes;

  // Get the copy size:
  auto copyTensorType = cast<TensorType>(copy.getResult(0).getType());
  auto rank = copyTensorType.getRank();
  SmallVector<OpFoldResult> tileSize(rank - 1, rewriter.getIndexAttr(1));
  int64_t elementBitWidth = copyTensorType.getElementTypeBitWidth();

  // auto totalCopySize = getTotalSize(copyTensorType.getShape());

  // divide the copy by subgroup:
  assert(subgroupSize.size() == 1); // only 1-D
  // auto subgroupCopySize = totalCopySize / subgroupSize[0];

  // TODO: determine tile sizes
  tileSizes.push_back(
      rewriter.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));
  tileSizes.push_back(
      rewriter.getIndexAttr(kPreferredCopyNumBits / elementBitWidth));

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
      /*outputs=*/ValueRange{copy.getOperand(0)},
      /*mapping=*/rewriter.getArrayAttr(mapping));

  rewriter.setInsertionPointToStart(newForallOp.getBody());

  auto inductionVars = newForallOp.getInductionVars();

  // TODO: make it multidimensional.
  Value subgroupId = rewriter.create<gpu::SubgroupIdOp>(loc, nullptr);
  Value laneId = rewriter.create<gpu::LaneIdOp>(loc, nullptr);

  // TODO: properly handle workgroup sizes and subgroup sizes.
  auto numParts = totalWorkgroupSize / subgroupSize[0];

  LLVM_DEBUG(llvm::dbgs() << "numParts: " << numParts << "\n");

  auto globalSlice =
      sliceTensor(rewriter, copy.getOperand(0), numParts, subgroupId);
  if (!globalSlice) {
    copy.emitOpError("failed to slice tensor");
  }

  // create an empty tensor with same shape of globalSlice:
  auto localSliceType = RankedTensorType::get(
      cast<RankedTensorType>(globalSlice->getType()).getShape(),
      cast<RankedTensorType>(globalSlice->getType()).getElementType());
  auto localSlice = rewriter.create<tensor::EmptyOp>(
      loc, TypeRange{localSliceType}, ValueRange{});

  // compute number of loads per thread
  auto sliceType = cast<RankedTensorType>(globalSlice->getType());
  auto sliceSize = getTotalSize(sliceType.getShape());

  auto elementByteWidth = elementBitWidth / 8;
  auto subgroupLoadSizeEachTime = elementByteWidth * subgroupSize[0];
  if (sliceSize % subgroupLoadSizeEachTime != 0) {
    copy.emitOpError("slice size is not divisible by load size");
    return;
  }
  auto numLoadsPerSlice = sliceSize / subgroupLoadSizeEachTime;

  Value localSliceValue = localSlice;
  Value gatherOffset = laneId;
  Value gatherStride =
      rewriter.create<arith::ConstantIndexOp>(loc, subgroupLoadSizeEachTime);
  for (int i = 0; i < numLoadsPerSlice; ++i) {
    // compute the offset
    auto delinearizedIndex = rewriter
                                 .create<affine::AffineDelinearizeIndexOp>(
                                     loc, gatherOffset, sliceType.getShape())
                                 .getMultiIndex();
    localSliceValue = rewriter.create<IREE::GPU::GlobalLoadDMAOp>(
        loc, sliceType, *globalSlice, localSliceValue, delinearizedIndex);
    if (i < numLoadsPerSlice - 1) {
      gatherOffset =
          rewriter.create<arith::AddIOp>(loc, gatherOffset, gatherStride);
    }
  }

  // get scf.forall.in_parallel:
  /*
  auto inParallel = newForallOp.getTerminator();
  rewriter.setInsertionPoint(inParallel.getBody(),
  inParallel.getBody()->begin());
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
      distributeLinalgCopyToThreads(rewriter, copy, *workgroupSize,
                                    *subgroupSize);
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
