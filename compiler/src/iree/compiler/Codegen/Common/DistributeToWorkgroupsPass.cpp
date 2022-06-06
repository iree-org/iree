// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== DistributeToWorkgroupsPass.cpp - Disribute tiles to workgroups -----===//
//
// This pass distributes the computation to workgroups. Unlike tile
// and distribute that tiles the computation to distribute the work, this pass
// splits `flow.dispatch.tensor.store` to have each workgroup compute a tile
// of the result.
//
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// When the dispatch is not yet distributed, distributes the
/// `flow.dispatch.tensor.store` operation that writes out the result
/// by writing just a slice of the store in each workgroup.
struct ExtractResultSlice
    : public OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  ExtractResultSlice(MLIRContext *context, ArrayRef<int64_t> tileSizes,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        tileSizes(tileSizes.begin(), tileSizes.end()) {}

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    Value source = storeOp.value();
    auto sourceType = source.getType().dyn_cast<ShapedType>();
    if (!sourceType) return failure();

    auto entryPointOp = getEntryPoint(storeOp->getParentOfType<func::FuncOp>());
    if (failed(entryPointOp)) return failure();

    /// Only split the store if the workgroup count region is empty.
    Region &workgroupCountRegion = entryPointOp.getValue().workgroup_count();
    if (!workgroupCountRegion.empty()) return failure();

    // Compute the slice to map to each workgroup.
    // 1) If tile size is not zero,
    //    - use that as the slice size
    //    - offset is tile size * thread_id
    // 2) If tile size is zero,
    //    - use the dim of the source as the slice size
    //    - offset is 0.
    int64_t sourceRank = sourceType.getRank();
    SmallVector<int64_t> filledTileSizes(tileSizes);
    filledTileSizes.resize(sourceRank, 0);
    Location loc = storeOp.getLoc();
    auto tileSizeVals = llvm::to_vector(
        llvm::map_range(filledTileSizes, [&](int64_t t) -> Value {
          return rewriter.create<arith::ConstantIndexOp>(loc, t);
        }));
    auto extentVals = llvm::to_vector(llvm::map_range(
        llvm::seq<int64_t>(0, sourceRank), [&](int64_t dim) -> Value {
          return rewriter.create<tensor::DimOp>(loc, source, dim);
        }));

    AffineExpr sym0, sym1, sym2;
    bindSymbols(rewriter.getContext(), sym0, sym1, sym2);
    // Offset map is always
    // affine_map<()[s0, s1] -> (s0 * s1)>()[%tileSize, %id]
    AffineMap offsetMap = AffineMap::get(0, 2, sym0 * sym1);
    // Size map is always
    // affine_map<()[s0, s1, s2, s3] -> (s0, s1 - s2 * s0)>()
    //     [%tileSize, %extent, %id]
    AffineMap sizeMap =
        AffineMap::get(0, 3, ArrayRef<AffineExpr>{sym0, sym1 - sym2 * sym0},
                       rewriter.getContext());

    Attribute zero = rewriter.getIndexAttr(0);
    Attribute one = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> offsets(sourceRank, zero),
        sizes(extentVals.begin(), extentVals.end()), strides(sourceRank, one);
    unsigned mappedDim = 0;
    SmallVector<int64_t> distributedTileSizes;
    // Map the innermost tiled dimension to `x`, next innermost tiled dimension
    // to `y` and so on.
    for (auto dim : llvm::reverse(llvm::seq<int64_t>(0, sourceRank))) {
      if (filledTileSizes[dim] && mappedDim < kNumMaxParallelDims) {
        Value id =
            rewriter.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, mappedDim);
        auto offsetOp = rewriter.create<AffineApplyOp>(
            loc, offsetMap, ValueRange{tileSizeVals[dim], id});
        offsets[dim] = offsetOp.getResult();
        auto sizeOp = rewriter.create<AffineMinOp>(
            loc, sizeMap, ValueRange{tileSizeVals[dim], extentVals[dim], id});
        sizes[dim] = sizeOp.getResult();
        distributedTileSizes.push_back(filledTileSizes[dim]);
        mappedDim++;
      }
    }

    // Extract the slice of the source each workgroups writes.
    auto sourceSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, source, offsets, sizes, strides);

    // Find the offset, size and stride into which the result is stored.
    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    if (failed(IREE::Flow::foldOffsetsSizesAndStrides(
            rewriter, loc, storeOp, sourceSlice, storeOp.getDroppedDims(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return failure();
    }

    // Update the entry point op to specify the number of threads to use
    // indicating that the code is distributed.
    WorkgroupCountRegionBuilder regionBuilder =
        [&distributedTileSizes](OpBuilder &builder, Location loc, Value device,
                                std::array<Value, 3> workload) {
          Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
          std::array<Value, 3> numWorkgroups = {one, one, one};
          for (auto ts : llvm::enumerate(distributedTileSizes)) {
            AffineExpr num;
            bindSymbols(builder.getContext(), num);
            AffineExpr denom =
                getAffineConstantExpr(ts.value(), builder.getContext());
            AffineMap ceilDiv = AffineMap::get(
                0, 1,
                getAffineBinaryOpExpr(AffineExprKind::CeilDiv, num, denom));
            numWorkgroups[ts.index()] = builder.create<AffineApplyOp>(
                loc, ceilDiv, workload[ts.index()]);
          }
          return numWorkgroups;
        };
    if (failed(defineWorkgroupCountRegion(rewriter, entryPointOp.getValue(),
                                          regionBuilder))) {
      return failure();
    }
    rewriter.eraseOp(entryPointOp.getValue());

    // Replace the full store with the store of the slice.
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, sourceSlice.getResult(), storeOp.target(),
        storeOp.target_dims(), combinedOffsets, combinedSizes, combinedStrides);
    return success();
  }

 private:
  // For now just use `int64_t` tile sizes. Eventually change these
  // to use `linalg::LinalgTilingOptions` (maybe).
  SmallVector<int64_t> tileSizes;
};

/// Folds the `flow.dispatch.tensor.load` -> `tensor.extract_slice` into
/// a `flow.dispatch.tensor.load` of just the slice.
struct FoldTensorLoadWithExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto loadOp =
        sliceOp.source().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp) {
      return failure();
    }

    SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;
    Location loc = sliceOp->getLoc();
    if (failed(IREE::Flow::foldOffsetsSizesAndStrides(
            rewriter, loc, loadOp, sliceOp, loadOp.getDroppedDims(),
            combinedOffsets, combinedSizes, combinedStrides))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        sliceOp, sliceOp.getType(), loadOp.source(), loadOp.source_dims(),
        combinedOffsets, combinedSizes, combinedStrides);
    return success();
  }
};

/// Pass to distribute the dispatch to workgroups.
struct DistributeToWorkgroupsPass
    : DistributeToWorkgroupsBase<DistributeToWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void DistributeToWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Insert the split for the `flow.dispatch.tensor.store`
  patterns.insert<ExtractResultSlice>(context, tileSizes);
  patterns.insert<FoldTensorLoadWithExtractSlice,
                  IREE::LinalgExt::SwapTilingInterfaceOp>(context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createDistributeToWorkgroupsPass() {
  return std::make_unique<DistributeToWorkgroupsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
