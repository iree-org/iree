//===- FoldMemRefAliasOps.cpp - Fold memref alias ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass folds loading/storing from/to subview ops into
// loading/storing from/to the original memref.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-fold-memref-alias-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Merges expand_shape operation with load/transferRead operation.
template <typename OpTy>
class LLVMCPULoadOpOfExpandShapeOpFolder final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};

/// Merges collapse_shape operation with load/transferRead operation.
template <typename OpTy>
class LLVMCPULoadOpOfCollapseShapeOpFolder final
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy loadOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

static SmallVector<Value>
calculateExpandedAccessIndices(AffineMap affineMap,
                               const SmallVector<Value> &indices, Location loc,
                               PatternRewriter &rewriter) {
  SmallVector<OpFoldResult> indicesOfr(llvm::to_vector(
      llvm::map_range(indices, [](Value v) -> OpFoldResult { return v; })));
  SmallVector<Value> expandedIndices;
  for (unsigned i = 0, e = affineMap.getNumResults(); i < e; i++) {
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc, affineMap.getSubMap({i}), indicesOfr);
    expandedIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
  return expandedIndices;
}

static LogicalResult
resolveSourceIndicesExpandShape(Location loc, PatternRewriter &rewriter,
                                memref::ExpandShapeOp expandShapeOp,
                                ValueRange indices,
                                SmallVectorImpl<Value> &sourceIndices) {
  // The below implementation uses computeSuffixProduct method, which only
  // allows int64_t values (i.e., static shape). Bail out if it has dynamic
  // shapes.
  if (!expandShapeOp.getResultType().hasStaticShape())
    return failure();

  MLIRContext *ctx = rewriter.getContext();
  for (ArrayRef<int64_t> groups : expandShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    int64_t groupSize = groups.size();

    // Construct the expression for the index value w.r.t to expand shape op
    // source corresponding the indices wrt to expand shape op result.
    SmallVector<int64_t> sizes(groupSize);
    for (int64_t i = 0; i < groupSize; ++i)
      sizes[i] = expandShapeOp.getResultType().getDimSize(groups[i]);
    SmallVector<int64_t> suffixProduct = computeSuffixProduct(sizes);
    SmallVector<AffineExpr> dims(groupSize);
    bindDimsList(ctx, MutableArrayRef{dims});
    AffineExpr srcIndexExpr = linearize(ctx, dims, suffixProduct);

    /// Apply permutation and create AffineApplyOp.
    SmallVector<OpFoldResult> dynamicIndices(groupSize);
    for (int64_t i = 0; i < groupSize; i++)
      dynamicIndices[i] = indices[groups[i]];

    // Creating maximally folded and composd affine.apply composes better with
    // other transformations without interleaving canonicalization passes.
    OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
        rewriter, loc,
        AffineMap::get(/*numDims=*/groupSize,
                       /*numSymbols=*/0, srcIndexExpr),
        dynamicIndices);
    sourceIndices.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }
  return success();
}

static LogicalResult
resolveSourceIndicesCollapseShape(Location loc, PatternRewriter &rewriter,
                                  memref::CollapseShapeOp collapseShapeOp,
                                  ValueRange indices,
                                  SmallVectorImpl<Value> &sourceIndices) {
  int64_t cnt = 0;
  SmallVector<Value> tmp(indices.size());
  SmallVector<OpFoldResult> dynamicIndices;
  for (ArrayRef<int64_t> groups : collapseShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    dynamicIndices.push_back(indices[cnt++]);
    int64_t groupSize = groups.size();

    // Calculate suffix product for all collapse op source dimension sizes.
    SmallVector<int64_t> sizes(groupSize);
    for (int64_t i = 0; i < groupSize; ++i)
      sizes[i] = collapseShapeOp.getSrcType().getDimSize(groups[i]);
    SmallVector<int64_t> suffixProduct = computeSuffixProduct(sizes);

    // Derive the index values along all dimensions of the source corresponding
    // to the index wrt to collapsed shape op output.
    auto d0 = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, suffixProduct);

    // Construct the AffineApplyOp for each delinearizingExpr.
    for (int64_t i = 0; i < groupSize; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc,
          AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
                         delinearizingExprs[i]),
          dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
    dynamicIndices.clear();
  }
  if (collapseShapeOp.getReassociationIndices().empty()) {
    auto zeroAffineMap = rewriter.getConstantAffineMap(0);
    int64_t srcRank =
        cast<MemRefType>(collapseShapeOp.getViewSource().getType()).getRank();
    for (int64_t i = 0; i < srcRank; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, zeroAffineMap, dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
  }
  return success();
}

/// Helpers to access the memref operand for each op.
template <typename LoadOrStoreOpTy>
static Value getMemRefOperand(LoadOrStoreOpTy op) {
  return op.getMemref();
}

static Value getMemRefOperand(vector::TransferReadOp op) {
  return op.getSource();
}

static Value getMemRefOperand(vector::LoadOp op) { return op.getBase(); }

template <typename OpTy>
LogicalResult LLVMCPULoadOpOfExpandShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto expandShapeOp =
      getMemRefOperand(loadOp).template getDefiningOp<memref::ExpandShapeOp>();

  if (!expandShapeOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp =
          dyn_cast<affine::AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesExpandShape(
          loadOp.getLoc(), rewriter, expandShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            loadOp, loadOp.getType(), expandShapeOp.getViewSource(),
            sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

template <typename OpTy>
LogicalResult LLVMCPULoadOpOfCollapseShapeOpFolder<OpTy>::matchAndRewrite(
    OpTy loadOp, PatternRewriter &rewriter) const {
  auto collapseShapeOp = getMemRefOperand(loadOp)
                             .template getDefiningOp<memref::CollapseShapeOp>();

  if (!collapseShapeOp)
    return failure();

  SmallVector<Value> indices(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());
  // For affine ops, we need to apply the map to get the operands to get the
  // "actual" indices.
  if (auto affineLoadOp =
          dyn_cast<affine::AffineLoadOp>(loadOp.getOperation())) {
    AffineMap affineMap = affineLoadOp.getAffineMap();
    auto expandedIndices = calculateExpandedAccessIndices(
        affineMap, indices, loadOp.getLoc(), rewriter);
    indices.assign(expandedIndices.begin(), expandedIndices.end());
  }
  SmallVector<Value> sourceIndices;
  if (failed(resolveSourceIndicesCollapseShape(
          loadOp.getLoc(), rewriter, collapseShapeOp, indices, sourceIndices)))
    return failure();
  llvm::TypeSwitch<Operation *, void>(loadOp)
      .Case([&](vector::LoadOp op) {
        rewriter.replaceOpWithNewOp<vector::LoadOp>(
            loadOp, loadOp.getType(), collapseShapeOp.getViewSource(),
            sourceIndices);
      })
      .Default([](Operation *) { llvm_unreachable("unexpected operation."); });
  return success();
}

void populateLLVMCPUFoldMemRefAliasOpPatterns(RewritePatternSet &patterns) {
  patterns.add<LLVMCPULoadOpOfExpandShapeOpFolder<vector::LoadOp>,
               LLVMCPULoadOpOfCollapseShapeOpFolder<vector::LoadOp>>(
      patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct LLVMCPUFoldMemRefAliasOpsPass final
    : public LLVMCPUFoldMemRefAliasOpsBase<LLVMCPUFoldMemRefAliasOpsPass> {
  void runOnOperation() override;
};

} // namespace

void LLVMCPUFoldMemRefAliasOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  memref::populateFoldMemRefAliasOpPatterns(patterns);
  populateLLVMCPUFoldMemRefAliasOpPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

std::unique_ptr<Pass> createLLVMCPUFoldMemRefAliasOpsPass() {
  return std::make_unique<LLVMCPUFoldMemRefAliasOpsPass>();
}

} // namespace iree_compiler
} // namespace mlir