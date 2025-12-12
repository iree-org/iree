// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldUnitExtentDims.cpp - Pass to fold unit extent dims of tensors -===//
//
// Light weight wrapper to call the patterns to fold unit extent dims with
// IREE control.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fold-unit-extent-dims"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSPASS
#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSFORFUNCPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Attention Pattern
//===----------------------------------------------------------------------===//

namespace {
/// Simplify collapse_shape(expand_shape) by removing unneeded unit dimensions
/// that get expanded and subsequently collapsed.
///
/// TODO: move this upstream with the other reshape folding patterns. This can
/// also be generalized to fold non-unit dimensions.
///
/// For example:
/// ```
/// %0 = expand_shape ... tensor<3x3x10xf32> into tensor<3x3x5x1x2xf32>
/// %1 = collapse_shape %1 ... tensor<3x3x5x1x2xf32> into tensor<9x5x2xf32>
/// ```
/// simplifies to
/// ```
/// %0 = expand_shape ... tensor<3x3x10xf32> into tensor<3x3x5x2xf32>
/// %1 = collapse_shape %1 ... tensor<3x3x5x2xf32> into tensor<9x5x2xf32>
/// ```
struct DropUnitDimsFromCollapseOfExpand
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = collapseOp.getSrc().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    const SmallVector<ReassociationIndices, 4> collapseReassoc =
        collapseOp.getReassociationIndices();
    ArrayRef<int64_t> interShape = expandOp.getType().getShape();
    ArrayRef<int64_t> outShape = collapseOp.getType().getShape();
    SmallVector<int64_t> interToOutMap(expandOp.getType().getRank());

    // Construct a set of dimensions (with respect to the intermediate
    // shape) that are unit length and get collapsed away by the final
    // `collapseOp` op.
    llvm::SmallDenseSet<int64_t> toDrop;
    for (const auto &[outDim, indices] : llvm::enumerate(collapseReassoc)) {
      for (auto [innerIdx, inDim] : llvm::enumerate(indices)) {
        // Can't drop this dim if it isnt statically 1 or if it isn't being
        // combined with any other dimensions.
        if (indices.size() == 1 || interShape[inDim] != 1) {
          continue;
        }

        // If outShape[outDim] == 1, we must preserve 1 unit dim,
        // so we drop the first. If the first is the only unit dim,
        // we can't drop it anyway.
        if (outShape[outDim] == 1 && innerIdx == 0) {
          continue;
        }
        toDrop.insert(inDim);
      }
    }

    // Remove dimensions from `toDrop` that weren't introduced by the
    // `expandOp` op.
    const auto expandReassoc = expandOp.getReassociationIndices();
    for (const auto &indices : expandReassoc) {
      // If all of indices are in `toDrop`, we must preserve at least one
      // to avoid an empty reassociation map during expansion.
      // This can happen when outShape does not have a unit dimension
      // corresponding to the unit dimensions being dropped here.
      if (llvm::all_of(indices,
                       [&](int64_t idx) { return toDrop.contains(idx); })) {
        toDrop.erase(indices[0]);
      }
    }
    if (toDrop.empty()) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "Didn't find any unit dims to drop");
    }

    // Construct a new intermediate shape without the foldable dims.
    SmallVector<int64_t> newInterShape;
    SmallVector<OpFoldResult> newInterSizes;
    newInterShape.reserve(interShape.size() - toDrop.size());
    newInterSizes.reserve(interShape.size() - toDrop.size());
    SmallVector<OpFoldResult> origInterSizes = getMixedValues(
        expandOp.getStaticOutputShape(), expandOp.getOutputShape(), rewriter);
    for (auto [idx, ofr] : llvm::enumerate(origInterSizes)) {
      if (!toDrop.contains(idx)) {
        std::optional<int64_t> staticDim = getConstantIntValue(ofr);
        newInterShape.push_back(staticDim.value_or(ShapedType::kDynamic));
        newInterSizes.push_back(ofr);
      }
    }

    /// Attempts to push back a new `ReassociationIndices` to `reassoc` that
    /// does not include the dimensions in `toDrop`.
    /// Returns true if new `ReassociationIndices` were appended to `reassoc`.
    auto pushBackReassociation =
        [&toDrop](SmallVectorImpl<ReassociationIndices> &reassoc, int64_t start,
                  int64_t origStart, int64_t count) -> bool {
      ReassociationIndices indices;
      indices.reserve(count);
      int64_t dim = start;
      for (auto idx : llvm::seq<int64_t>(origStart, origStart + count)) {
        if (!toDrop.contains(idx)) {
          indices.push_back(dim++);
        }
      }
      if (!indices.empty()) {
        reassoc.push_back(std::move(indices));
        return true;
      }
      return false;
    };

    // Construct new reassociations for the `collapse_shape` that does
    // not include the dropped dimensions.
    SmallVector<ReassociationIndices, 4> newCollapseReassoc;
    int64_t collapsedDim = 0;
    for (auto dim : llvm::seq<int64_t>(0, outShape.size())) {
      bool changed = pushBackReassociation(newCollapseReassoc, collapsedDim,
                                           collapseReassoc[dim][0],
                                           collapseReassoc[dim].size());
      if (changed) {
        collapsedDim += newCollapseReassoc.back().size();
      }
    }

    // Construct new reassociations for the `expand_shape` op that does
    // not include the dropped dimensions.
    SmallVector<ReassociationIndices, 4> newExpandReassoc;
    ArrayRef<int64_t> srcShape = expandOp.getSrcType().getShape();
    int64_t expandedDim = 0;
    for (auto dim : llvm::seq<int64_t>(0, srcShape.size())) {
      bool changed = pushBackReassociation(newExpandReassoc, expandedDim,
                                           expandReassoc[dim][0],
                                           expandReassoc[dim].size());
      if (changed) {
        expandedDim += newExpandReassoc.back().size();
      }
    }

    // Construct the new expand_shape and collapse_shape ops.
    // Note: we must handle the cases where the expand/collapse is no longer
    // needed. Both ops require a non-identity reassociation (i.e. they can't be
    // no-ops).
    Value newExpanded = expandOp.getSrc();
    if (!llvm::all_of(newExpandReassoc,
                      llvm::hasSingleElement<ReassociationIndicesRef>)) {
      newExpanded = tensor::ExpandShapeOp::create(
          rewriter, expandOp.getLoc(),
          RankedTensorType::get(newInterShape,
                                expandOp.getType().getElementType()),
          expandOp.getSrc(), newExpandReassoc, newInterSizes);
    }

    Value result = newExpanded;
    if (!llvm::all_of(newCollapseReassoc,
                      llvm::hasSingleElement<ReassociationIndicesRef>)) {
      result = tensor::CollapseShapeOp::create(rewriter, collapseOp.getLoc(),
                                               collapseOp.getType(),
                                               newExpanded, newCollapseReassoc);
    }
    rewriter.replaceOp(collapseOp, result);
    return success();
  }
};

// Fold unit dims from `tensor.extract` ops.
struct FoldUnitDimsFromExtractOp : OpRewritePattern<tensor::ExtractOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType srcType = extractOp.getTensor().getType();
    if (srcType.getShape().empty() ||
        llvm::none_of(srcType.getShape(),
                      [](int64_t size) { return size == 1; })) {
      return failure();
    }
    SmallVector<Value> oldIndices = extractOp.getIndices();

    SmallVector<int64_t> newShape;
    SmallVector<Value> newIndices;
    SmallVector<ReassociationIndices> reassoc;
    ReassociationIndices currReassoc;

    // Build reassociation groups where each non-unit dimension forms one output
    // dimension, and unit dimensions are grouped with adjacent non-unit dims.
    for (auto [idx, size] : llvm::enumerate(srcType.getShape())) {
      currReassoc.push_back(idx);

      if (size != 1) {
        // Non-unit dimension: this forms one output dimension
        // Finish current group and start a new one
        reassoc.push_back(std::move(currReassoc));
        currReassoc.clear();
        newShape.push_back(size);
        newIndices.push_back(oldIndices[idx]);
      }
    }

    // If we have trailing unit dims, merge them with the last group
    if (!currReassoc.empty() && !reassoc.empty()) {
      reassoc.back().append(currReassoc.begin(), currReassoc.end());
    }

    rewriter.setInsertionPointAfterValue(extractOp.getTensor());
    auto collapseOp = tensor::CollapseShapeOp::create(
        rewriter, extractOp.getLoc(), extractOp.getTensor(), reassoc);

    rewriter.setInsertionPointAfter(extractOp);
    auto newExtract = tensor::ExtractOp::create(
        rewriter, extractOp.getLoc(), extractOp.getResult().getType(),
        collapseOp.getResult(), newIndices);
    rewriter.replaceOp(extractOp, newExtract);
    return success();
  }
};

// Fold unit dims from `tensor.extract_slice` ops by inserting a collapse_shape
// on the input tensor.
//
// Example:
// ```
// %slice = tensor.extract_slice %src[%i, 0, %j, 0, %k]
//                                    [%si, 1, %sj, 1, %sk]
//                                    [1, 1, 1, 1, 1]
//   : tensor<10x1x20x1x30xf32> to tensor<?x1x?x1x?xf32>
// ```
// becomes:
// ```
// %collapsed = tensor.collapse_shape %src [[0], [1, 2], [3, 4]]
//   : tensor<10x1x20x1x30xf32> into tensor<10x20x30xf32>
// %slice = tensor.extract_slice %collapsed[%i, %j, %k]
//                                          [%si, %sj, %sk]
//                                          [1, 1, 1]
//   : tensor<10x20x30xf32> to tensor<?x?x?xf32>
// ```
struct FoldUnitDimsFromExtractSliceOp
    : OpRewritePattern<tensor::ExtractSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType srcType = sliceOp.getSourceType();
    if (srcType.getShape().empty() ||
        llvm::none_of(srcType.getShape(),
                      [](int64_t size) { return size == 1; })) {
      return failure();
    }

    // Get original offsets, sizes, and strides
    SmallVector<OpFoldResult> oldOffsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> oldSizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> oldStrides = sliceOp.getMixedStrides();

    // Build new shape and reassociation indices
    SmallVector<int64_t> newShape;
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    SmallVector<ReassociationIndices> reassoc;
    ReassociationIndices currReassoc;

    // Track which result dimensions are kept (for rank-reducing slices)
    llvm::SmallBitVector droppedDims = sliceOp.getDroppedDims();
    SmallVector<int64_t> newResultShape;
    SmallVector<OpFoldResult> resultSizes; // Track sizes for result dimensions
    int64_t resultDimIdx = 0;

    // Build reassociation groups where each non-unit dimension forms one output
    // dimension, and unit dimensions are grouped with adjacent non-unit dims.
    for (auto [idx, size] : llvm::enumerate(srcType.getShape())) {
      currReassoc.push_back(idx);

      if (size != 1) {
        // Non-unit dimension: this forms one output dimension
        reassoc.push_back(std::move(currReassoc));
        currReassoc.clear();
        newShape.push_back(size);
        newOffsets.push_back(oldOffsets[idx]);
        newSizes.push_back(oldSizes[idx]);
        newStrides.push_back(oldStrides[idx]);

        // Determine if this dimension is kept in the result
        if (!droppedDims.test(idx)) {
          // This dimension is in the result, get its size
          auto resultType = sliceOp.getType();
          if (resultDimIdx < resultType.getRank()) {
            newResultShape.push_back(resultType.getShape()[resultDimIdx]);
            resultSizes.push_back(oldSizes[idx]);
          }
          resultDimIdx++;
        }
      }
      // Unit dimensions are skipped in the collapsed tensor
    }

    // If we have trailing unit dims, merge them with the last group
    if (!currReassoc.empty() && !reassoc.empty()) {
      reassoc.back().append(currReassoc.begin(), currReassoc.end());
    }

    // If no unit dims were found, nothing to do
    if (newShape.size() == srcType.getRank()) {
      return failure();
    }

    // Create collapsed source tensor
    rewriter.setInsertionPointAfterValue(sliceOp.getSource());
    Value collapsedSrc = tensor::CollapseShapeOp::create(
        rewriter, sliceOp.getLoc(), sliceOp.getSource(), reassoc);

    // Create new extract_slice on the collapsed tensor (without unit dims)
    RankedTensorType newResultType = RankedTensorType::get(
        newResultShape, srcType.getElementType(), srcType.getEncoding());

    rewriter.setInsertionPoint(sliceOp);
    Value newSlice = tensor::ExtractSliceOp::create(
        rewriter, sliceOp.getLoc(), newResultType, collapsedSrc, newOffsets,
        newSizes, newStrides);

    // If the original result had unit dimensions, expand them back
    RankedTensorType originalResultType = sliceOp.getType();
    if (originalResultType.getRank() != newResultType.getRank()) {
      // Build reassociation for expanding back to original result shape
      // Map from collapsed result dims back to original result dims
      SmallVector<ReassociationIndices> resultReassoc;
      ReassociationIndices currResultReassoc;

      for (auto [idx, size] : llvm::enumerate(originalResultType.getShape())) {
        currResultReassoc.push_back(idx);

        if (size != 1) {
          // Non-unit dimension in result
          resultReassoc.push_back(std::move(currResultReassoc));
          currResultReassoc.clear();
        }
      }

      // Handle trailing unit dims
      if (!currResultReassoc.empty() && !resultReassoc.empty()) {
        resultReassoc.back().append(currResultReassoc.begin(),
                                    currResultReassoc.end());
      }

      // Build output shape for expand_shape
      SmallVector<OpFoldResult> outputShape;
      int64_t nonUnitIdx = 0;
      for (auto [resultIdx, size] :
           llvm::enumerate(originalResultType.getShape())) {
        if (size == 1) {
          // Unit dimension - use constant 1
          outputShape.push_back(rewriter.getIndexAttr(1));
        } else {
          // Non-unit dimension - use the size from resultSizes
          if (nonUnitIdx < resultSizes.size()) {
            outputShape.push_back(resultSizes[nonUnitIdx]);
            nonUnitIdx++;
          }
        }
      }

      Value expandedSlice = tensor::ExpandShapeOp::create(
          rewriter, sliceOp.getLoc(), originalResultType, newSlice,
          resultReassoc, outputShape);
      rewriter.replaceOp(sliceOp, expandedSlice);
    } else {
      rewriter.replaceOp(sliceOp, newSlice);
    }

    return success();
  }
};

} // namespace

static void
populatefoldUnitDimsPatterns(RewritePatternSet &foldUnitDimsPatterns) {
  linalg::ControlDropUnitDims options;
  auto defaultFn = options.controlFn;

  options.controlFn = [=](Operation *op) {
    // Ignore operations already in dispatches.
    if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
      return SmallVector<unsigned>{};
    }
    if (isa<IREE::LinalgExt::LinalgExtOp>(op)) {
      return IREE::LinalgExt::defaultControlDropUnitDims(op);
    }
    return defaultFn(op);
  };

  linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
  IREE::LinalgExt::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns,
                                                      options);
  linalg::populateMoveInitOperandsToInputPattern(foldUnitDimsPatterns);
  foldUnitDimsPatterns
      .insert<DropUnitDimsFromCollapseOfExpand, FoldUnitDimsFromExtractOp,
              FoldUnitDimsFromExtractSliceOp>(
          foldUnitDimsPatterns.getContext());
}

static LogicalResult
foldUnitDimsOnGlobal(IRRewriter &rewriter, IREE::Util::GlobalOpInterface global,
                     SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps,
                     SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps,
                     SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  SmallVector<int64_t> newShape;
  auto globalType = cast<RankedTensorType>(global.getGlobalType());
  for (auto size : globalType.getShape()) {
    if (size != 1) {
      newShape.push_back(size);
    }
  }
  auto newGlobalType = globalType.clone(newShape);
  auto initialValue = global.getGlobalInitialValue();
  if (!initialValue)
    return success();
  // TODO: Handle other cases
  auto newInitialValue =
      llvm::TypeSwitch<Attribute, Attribute>(initialValue)
          .Case<IREE::Util::UninitializedAttr>([&](Attribute) {
            return IREE::Util::UninitializedAttr::get(rewriter.getContext(),
                                                      newGlobalType);
          })
          .Case<IREE::Stream::NamedParameterAttr>(
              // TODO: Remove this case once frontends have caught up, we should
              // not have stream.parameter.named at this level.
              [&](IREE::Stream::NamedParameterAttr attr) {
                return IREE::Stream::NamedParameterAttr::get(
                    rewriter.getContext(), newGlobalType, attr.getScope(),
                    attr.getKey(), attr.getConfig());
              })
          .Case<IREE::Flow::NamedParameterAttr>(
              [&](IREE::Flow::NamedParameterAttr attr) {
                return IREE::Flow::NamedParameterAttr::get(
                    rewriter.getContext(), newGlobalType, attr.getScope(),
                    attr.getKey(), attr.getConfig());
              })
          .Default([&](Attribute) { return nullptr; });
  if (!newInitialValue) {
    return success();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(global);
  auto newGlobal =
      clone(rewriter, global, global->getResultTypes(), global->getOperands());
  newGlobal.setGlobalType(newGlobalType);
  newGlobal.setGlobalInitialValue(newInitialValue);

  // Rewrite loads and stores to use the new global.
  auto expandShapeReInds =
      getReassociationIndicesForReshape(globalType, newGlobalType);
  if (!expandShapeReInds) {
    return failure();
  }

  for (auto load : loadOps) {
    rewriter.setInsertionPoint(load);
    auto newLoad = clone(rewriter, load, {newGlobalType}, load->getOperands());
    newLoad.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        load, globalType, newLoad->getResult(0), expandShapeReInds.value());
  }
  for (auto store : storeOps) {
    rewriter.setInsertionPoint(store);
    Value collapse = tensor::CollapseShapeOp::create(
        rewriter, store.getLoc(), newGlobalType, store->getOperand(0),
        expandShapeReInds.value());
    auto newStore =
        clone(rewriter, store, store->getResultTypes(), store->getOperands());
    newStore.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
    newStore.setStoredGlobalValue(collapse);
    rewriter.eraseOp(store);
  }
  rewriter.eraseOp(global);
  return success();
}

namespace {
struct FoldUnitExtentDimsPass final
    : public impl::FoldUnitExtentDimsPassBase<FoldUnitExtentDimsPass> {
  void runOnOperation() override;
};

struct FoldUnitExtentDimsForFuncPass final
    : public impl::FoldUnitExtentDimsForFuncPassBase<
          FoldUnitExtentDimsForFuncPass> {
  void runOnOperation() override;
};

} // namespace

void FoldUnitExtentDimsPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  SymbolTable moduleSymbols(moduleOp);
  Explorer explorer(moduleOp, TraversalAction::RECURSE);
  explorer.initialize();
  IRRewriter rewriter(context);

  // Fold unit dims of GlobalOpInterface ops.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    IREE::Util::GlobalOpInterface global = globalInfo->op;
    auto tensorType = dyn_cast<RankedTensorType>(global.getGlobalType());
    if (!tensorType || !global.isGlobalPrivate()) {
      return;
    }
    if (llvm::none_of(tensorType.getShape(),
                      [](int64_t size) { return size == 1; })) {
      return;
    }
    SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps =
        llvm::to_vector(globalInfo->getLoads());
    SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps =
        llvm::to_vector(globalInfo->getStores());
    if (failed(foldUnitDimsOnGlobal(rewriter, global, loadOps, storeOps,
                                    moduleSymbols))) {
      return signalPassFailure();
    }
  });

  RewritePatternSet foldUnitDimsPatterns(context);
  populatefoldUnitDimsPatterns(foldUnitDimsPatterns);
  GreedyRewriteConfig rewriterConfig;
  rewriterConfig.setMaxIterations(GreedyRewriteConfig::kNoLimit);
  if (failed(applyPatternsGreedily(moduleOp, std::move(foldUnitDimsPatterns),
                                   rewriterConfig))) {
    return signalPassFailure();
  }
}

void FoldUnitExtentDimsForFuncPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet foldUnitDimsPatterns(context);
  populatefoldUnitDimsPatterns(foldUnitDimsPatterns);
  GreedyRewriteConfig rewriterConfig;
  rewriterConfig.setMaxIterations(GreedyRewriteConfig::kNoLimit);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(foldUnitDimsPatterns), rewriterConfig))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
