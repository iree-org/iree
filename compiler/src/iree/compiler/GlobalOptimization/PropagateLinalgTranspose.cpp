// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PropagateLinalgTranspose.cpp - Pass to propagate transposes ---------==//
//
// The pass is to propagate linalg.transpose operations through a restricted
// set of operations based on a set of local propagation decisions.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-propagate-linalg-transpose"

namespace mlir::iree_compiler::GlobalOptimization {

//===----------------------------------------------------------------------===//
// Transpose permutation helpers
//===----------------------------------------------------------------------===//

static bool isIdentityPermutation(ArrayRef<int64_t> perm) {
  for (auto [index, dim] : llvm::enumerate(perm)) {
    if (index != dim) {
      return false;
    }
  }
  return true;
}

// Constructs a transpose of the given tensor and permutation.
static Value createTransposeInit(OpBuilder &builder, Value source,
                                 ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
  return empty;
}

// Constructs a transpose of the given tensor and permutation.
static Value createTranspose(OpBuilder &builder, Value source,
                             ArrayRef<int64_t> perm) {
  Value empty = createTransposeInit(builder, source, perm);
  return builder
      .create<linalg::TransposeOp>(source.getLoc(), source, empty, perm)
      ->getResult(0);
}

static RankedTensorType getPermutedTensorType(RankedTensorType type,
                                              SmallVector<int64_t> perm) {
  SmallVector<int64_t> permutedShape = applyPermutation(type.getShape(), perm);
  return RankedTensorType::get(permutedShape, type.getElementType());
}

//===----------------------------------------------------------------------===//
// Transpose specialization
//===----------------------------------------------------------------------===//

// Indicates whether the given linalg op represents a transpose. In particular,
// it requires a single input where the indexing maps are full permutations and
// non-equal.
static bool isaTransposeOpInterface(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumParallelLoops() != linalgOp.getNumLoops())
    return false;

  if (linalgOp.getNumDpsInputs() != 1 || linalgOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = linalgOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isPermutation() ||
      !mapRange.back().isPermutation() || mapRange.front() == mapRange.back()) {
    return false;
  }
  return llvm::hasSingleElement(linalgOp.getBlock()->getOperations());
}

// Specializes linalg.generic op to linalg.transpose if it is transposing a
// single input.
static void specializeGenericTransposeOp(RewriterBase &rewriter,
                                         linalg::GenericOp genericOp) {
  if (!isaTransposeOpInterface(genericOp)) {
    return;
  }

  auto mapRange = genericOp.getIndexingMapsArray();
  AffineMap outMap = mapRange.back();
  AffineMap inMap = mapRange.front();
  SmallVector<int64_t> perm;
  // To get the permutation, look at each output index and find which
  // dimension in the input we're reading from for that index.
  for (AffineExpr expr : outMap.getResults()) {
    perm.push_back(*inMap.getResultPosition(expr));
  }
  rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
      genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0], perm);
}

//===----------------------------------------------------------------------===//
// Other pattern helpers
//===----------------------------------------------------------------------===//

/// If the `op` is a ContractionOpInterface, return the generalized op if
/// generalizing is allowed. Otherwise if the `op` is a linalg::GenericOp,
/// then just return the generic op.
static FailureOr<linalg::GenericOp>
getGenericOpOrGeneralizeContraction(RewriterBase &rewriter, Operation *op,
                                    bool allowGeneralizing) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return failure();
  }
  // TODO: Right now this is restricted to contractions due to fragility around
  // handling of convolutions.
  if (!isa<linalg::GenericOp>(linalgOp) &&
      !(allowGeneralizing && linalg::isaContractionOpInterface(linalgOp))) {
    return failure();
  }
  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (genericOp) {
    return genericOp;
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);
  return linalg::generalizeNamedOp(rewriter, linalgOp);
}

//===----------------------------------------------------------------------===//
// Transpose Bubbling Patterns
//===----------------------------------------------------------------------===//

namespace {

// Fuses a transpose with the init of a linalg.generic op or contraction op.
// Contraction ops are generalized and then treated as a generic. For example,
//
// linalg.generic {
//   indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
//                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   ins(%0 : tensor<2x7x5>) outs(%1 : tensor<7x2x5>)
//
// %2 = linalg.transpose ... permutation = [0, 2, 1] :
//                           tensor<7x2x5> -> tensor<7x5x2>
// Becomes
//
// linalg.generic {
//   indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d0, d1)>,
//                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   ins(%0 : tensor<2x7x5>) outs(%3 : tensor<7x5x2>)
class FuseTransposeWithProducerLinalgOp
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;
  FuseTransposeWithProducerLinalgOp(MLIRContext *ctx, bool aggressiveProp,
                                    PatternBenefit b = 1)
      : OpRewritePattern<linalg::TransposeOp>(ctx, b),
        allowGeneralizing(aggressiveProp) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(transposeOp)) {
      return failure();
    }
    OpResult result = dyn_cast<OpResult>(transposeOp.getInput());
    if (!result) {
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose input defined by block argument");
    }
    if (!result.hasOneUse()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "multi use transpose input");
    }
    auto linalgOp = dyn_cast<linalg::LinalgOp>(result.getOwner());
    if (!linalgOp) {
      return rewriter.notifyMatchFailure(
          transposeOp, "non-linalg op producer for transpose input");
    }

    int64_t resultIndex = result.getResultNumber();
    auto maybeGenericOp = getGenericOpOrGeneralizeContraction(
        rewriter, result.getOwner(), allowGeneralizing);
    if (failed(maybeGenericOp)) {
      return rewriter.notifyMatchFailure(
          transposeOp, "linalg op producer is not generic or contraction");
    }

    auto genericOp = maybeGenericOp.value();
    result = genericOp->getOpResult(resultIndex);

    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto invPerm = invertPermutationVector(perm);

    // 1. Get the transposed init of the generic.
    Value init = genericOp.getDpsInits()[resultIndex];
    SmallVector<Value> inits = genericOp.getDpsInits();
    Value newInit = createTranspose(rewriter, init, perm);
    inits[resultIndex] = newInit;

    SmallVector<Type> resultTypes(genericOp->getResultTypes());
    resultTypes[resultIndex] = newInit.getType();

    // 2. Update the indexing map of the transposed init operand by permuting
    // the results of the map.
    SmallVector<AffineMap> newIndexingMaps = genericOp.getIndexingMapsArray();
    AffineMap resultMap =
        newIndexingMaps[genericOp.getNumDpsInputs() + resultIndex];
    SmallVector<AffineExpr> newExprs =
        applyPermutation(resultMap.getResults(), perm);
    AffineMap transposedMap =
        AffineMap::get(resultMap.getNumDims(), resultMap.getNumSymbols(),
                       newExprs, rewriter.getContext());
    newIndexingMaps[genericOp.getNumDpsInputs() + resultIndex] = transposedMap;

    // 3. Create the new generic with the same iteration order.
    auto newGenericOp = rewriter.create<linalg::GenericOp>(
        genericOp.getLoc(), resultTypes, genericOp.getDpsInputs(), newInit,
        newIndexingMaps, genericOp.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
    rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                               newGenericOp.getRegion().begin());

    // 4. Remap iteration space of the generic to match the dimension order of
    // the output.
    if (newGenericOp.getNumResults() == 1) {
      SmallVector<unsigned int> interchange;
      int64_t permIdx = 0;
      for (int i = 0, e = transposedMap.getNumDims(); i < e; ++i) {
        if (transposedMap.isFunctionOfDim(i)) {
          interchange.push_back(
              llvm::cast<AffineDimExpr>(transposedMap.getResult(permIdx))
                  .getPosition());
          permIdx++;
          continue;
        }
        interchange.push_back(i);
      }
      auto interchangedGenericOp =
          linalg::interchangeGenericOp(rewriter, newGenericOp, interchange);
      // Interchange only fails if interchangeGenericOpPrecondition fails, which
      // only fails if the interchange vector is not invertible or doesn't match
      // the number of loops in the generic, both of which are guaranteed by
      // the fact that the output map must be a projection in the above
      // construction.
      assert(succeeded(interchangedGenericOp) &&
             "failed to interchange transposed generic");
      newGenericOp = *interchangedGenericOp;
    }

    // 5. Replace the result of the transpose with the transposed init.
    rewriter.replaceOp(transposeOp, newGenericOp->getResult(resultIndex));
    for (auto [oldRes, newRes] :
         llvm::zip_equal(genericOp.getResults(), newGenericOp->getResults())) {
      if (oldRes.getResultNumber() == resultIndex)
        continue;
      rewriter.replaceAllUsesWith(oldRes, newRes);
    }
    return success();
  }

private:
  bool allowGeneralizing = false;
};

// Bubbles a transpose through a tensor.collapse_shape.
class BubbleTransposeThroughCollapseShape
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(transposeOp)) {
      return failure();
    }
    Value source = transposeOp.getDpsInputOperand(0)->get();
    auto collapseOp = source.getDefiningOp<tensor::CollapseShapeOp>();
    // Do not propagate through reshapes if the transpose has multiple users, as
    // this could end up duplicating the transposes. We should only propagate
    // through reshape when it is free to do so.
    if (!collapseOp || !collapseOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose input is not a single-use collapse shape");
    }

    SmallVector<ReassociationIndices> reassociations =
        collapseOp.getReassociationIndices();

    // Because we are doing transpose(collapse_shape), all expanded groups are
    // transposed together. As a result, to get the permutation of the new
    // transpose, we can just flatten the transposed reassociation indices.
    // For example,
    //
    // reassociation_map = [[0, 1, 2], [3], [4, 5]]
    // permutation = [1, 2, 0]
    //
    // Becomes
    //
    // permutation = [3, 4, 5, 0, 1, 2]
    // reassociation_map = [[0], [1, 2], [3, 4, 5]]
    applyPermutationToVector(reassociations, transposeOp.getPermutation());

    SmallVector<int64_t> newPerm;
    SmallVector<ReassociationIndices> newReassociations;
    int64_t expandedDim = 0;
    for (auto reassoc : reassociations) {
      ReassociationIndices newReassoc;
      for (auto dim : reassoc) {
        newPerm.push_back(dim);
        newReassoc.push_back(expandedDim++);
      }
      newReassociations.push_back(newReassoc);
    }

    Value newTranspose =
        createTranspose(rewriter, collapseOp.getSrc(), newPerm);
    Value newReshape = rewriter.create<tensor::CollapseShapeOp>(
        collapseOp.getLoc(), transposeOp.getResultTypes()[0], newTranspose,
        newReassociations);
    rewriter.replaceOp(transposeOp, newReshape);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Transpose Sinking Patterns
//===----------------------------------------------------------------------===//

namespace {

// Combines two transposes into one. This shouldn't be strictly necessary as
// fusion should cancel inverse transposes, but doing this here can open up
// new propagation opportunities and eases the analysis in fusion/later passes.
class ComposeTransposes : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp consumer,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(consumer)) {
      return failure();
    }
    Value input = consumer.getInput();
    auto producer = input.getDefiningOp<linalg::TransposeOp>();
    if (!producer) {
      return failure();
    }

    ArrayRef<int64_t> producerPerm = producer.getPermutation();
    ArrayRef<int64_t> consumerPerm = consumer.getPermutation();
    SmallVector<int64_t> composedPerm =
        applyPermutation(producerPerm, consumerPerm);

    Value transposedSource = producer.getInput();
    if (!isIdentityPermutation(composedPerm)) {
      transposedSource =
          createTranspose(rewriter, transposedSource, composedPerm);
    }
    rewriter.replaceOp(consumer, transposedSource);
    return success();
  }
};

// Sinks a transpose through a tensor.extract_slice iff the transpose turns
// the extracted slice into a contiguous slice.
class SinkTransposeThroughExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(extractOp)) {
      return failure();
    }
    Value source = extractOp.getSource();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    // Applying `perm` takes a list from the pre-transpose ordering to the
    // post-transpose ordering.
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    // Applying `invPerm` takes a list from the post-transpose ordering to the
    // pre-transpose ordering. Sinking a transpose through an op is largely a
    // matter of rewriting it in pre-transpose space, and thus just applying
    // the inverse permutation.
    auto invPerm = invertPermutationVector(perm);

    SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = extractOp.getMixedStrides();
    ArrayRef<int64_t> srcShape = extractOp.getSourceType().getShape();

    // Permute the offsets, sizes, and strides to pre-transpose ordering.
    applyPermutationToVector(offsets, invPerm);
    applyPermutationToVector(sizes, invPerm);
    applyPermutationToVector(strides, invPerm);
    SmallVector<int64_t> baseShape = applyPermutation(srcShape, invPerm);

    // Check if the resulting offsets, sizes, and strides correspond to a
    // contiguous slice and can thus be mappable to a `flow.tensor.update` op.
    // This should always be worth doing because this can remove a dispatch for
    // the slice, and the transpose is on the slice rather than the full tensor.
    if (!IREE::Flow::isOffsetSizeAndStrideMappableToFlow(offsets, sizes,
                                                         strides, baseShape)) {
      return rewriter.notifyMatchFailure(
          extractOp, "transposed slice not mappable to flow ops");
    }

    ArrayRef<int64_t> staticSizes = extractOp.getStaticSizes();
    ArrayRef<int64_t> sliceShape = extractOp.getResultType().getShape();
    std::optional<llvm::SmallDenseSet<unsigned>> maybeRankReducingMask =
        mlir::computeRankReductionMask(staticSizes, sliceShape);
    if (!maybeRankReducingMask) {
      return rewriter.notifyMatchFailure(
          extractOp, "failed to compute rank reducing mask");
    }
    llvm::SmallDenseSet<unsigned> rankReducingMask = *maybeRankReducingMask;

    // Find rank reducing map in the pre-transposed domain.
    int64_t dim = 0;
    llvm::SmallDenseMap<int64_t, int64_t> rankReducedMap;
    // Since `dim` is in the pre-transposed domain, and is incrementing each
    // iteration, `idx` must also be in the pre-transposed domain.
    for (int64_t idx = 0, e = perm.size(); idx < e; ++idx) {
      // Get index in the transposed domain, since `rankReducingMask` is in
      // the transposed domain.
      if (!rankReducingMask.contains(perm[idx])) {
        // Domain of `rankReducedMap` is in pre-transposed domain.
        rankReducedMap[idx] = dim++;
      }
    }

    // Compute the new permutation by dropping all rank-reduced dimensions.
    SmallVector<int64_t> rankReducedPerm;
    for (int64_t i : perm) {
      if (!rankReducingMask.contains(i)) {
        rankReducedPerm.push_back(rankReducedMap[i]);
      }
    }

    auto rankReducedInvPerm = invertPermutationVector(rankReducedPerm);

    RankedTensorType sliceType = getPermutedTensorType(
        cast<RankedTensorType>(extractOp.getType()), rankReducedInvPerm);
    Value slice = rewriter.create<tensor::ExtractSliceOp>(
        extractOp.getLoc(), sliceType, transposeOp.getInput(), offsets, sizes,
        strides);
    // Transpose back to the original slice.
    if (!isIdentityPermutation(rankReducedPerm)) {
      slice = createTranspose(rewriter, slice, rankReducedPerm);
    }
    rewriter.replaceOp(extractOp, slice);
    return success();
  }
};

// Sinks a transpose through a tensor.expand_shape.
class SinkTransposeThroughExpandShape
    : public OpRewritePattern<tensor::ExpandShapeOp> {
public:
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(expandOp)) {
      return failure();
    }
    Value source = expandOp.getSrc();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    // Do not propagate through reshapes if the transpose has multiple users, as
    // this could end up duplicating the transposes. We should only propagate
    // through reshape when it is free to do so.
    if (!transposeOp || !transposeOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          expandOp, "expand shape input is not a single-use transpose");
    }

    auto invPerm = invertPermutationVector(transposeOp.getPermutation());
    SmallVector<ReassociationIndices> reassociations =
        expandOp.getReassociationIndices();

    // Because we are doing expand_shape(transpose), all expanded groups are
    // transposed together. As a result, to get the permutation of the new
    // transpose, we can just flatten the transposed reassociation indices.
    // For example,
    //
    // permutation = [0, 2, 1]
    // reassociation_map = [[0, 1, 2], [3], [4, 5]]
    //
    // Becomes
    //
    // reassociation_map = [[0, 1, 2], [3, 4], [5]]
    // permutation = [0, 1, 2, 4, 5, 3]
    applyPermutationToVector(reassociations, invPerm);

    SmallVector<int64_t> newInvPerm;
    SmallVector<ReassociationIndices> newReassociations;
    int64_t expandedDim = 0;
    for (auto reassoc : reassociations) {
      ReassociationIndices newReassoc;
      for (auto dim : reassoc) {
        newInvPerm.push_back(dim);
        newReassoc.push_back(expandedDim++);
      }
      newReassociations.push_back(newReassoc);
    }

    auto newPerm = invertPermutationVector(newInvPerm);

    RankedTensorType expandedType = getPermutedTensorType(
        cast<RankedTensorType>(expandOp.getType()), newInvPerm);
    Value transposedReshape = rewriter.create<tensor::ExpandShapeOp>(
        expandOp.getLoc(), expandedType, transposeOp.getInput(),
        newReassociations);
    Value originalReshape =
        createTranspose(rewriter, transposedReshape, newPerm);
    rewriter.replaceOp(expandOp, originalReshape);
    return success();
  }
};

// Fuses a transpose with the input of a linalg.generic op or contraction op.
// Contraction ops are generalized and then treated as a generic. For example,
//
// %0 = linalg.transpose ... permutation = [0, 2, 1] :
//                           tensor<2x5x7> -> tensor<2x7x5>
// linalg.generic {
//   indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
//                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   ins(%0 : tensor<2x7x5>)
//
// Becomes
//
// linalg.generic {
//   indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
//                    affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   ins(%0 : tensor<2x5x7>)
//
// This is considered just one way to model transpose propagation to generics,
// another option would be to interpret the transpose on the iterators of the
// generic, thus producing a transpose on the output and any other inputs to
// the generic. This has the potential introduce more transposes/data movement
// and isn't the way this pass is modeled. Global data layout transformations
// like that are better suited for pack/unpack propagation rooted on specific
// operations.
//
// TODO: Rewrite this to use elementwise op fusion patterns.
class FuseTransposeWithLinalgOpConsumer
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;
  FuseTransposeWithLinalgOpConsumer(MLIRContext *ctx, bool aggressiveProp,
                                    PatternBenefit b = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx, b),
        allowGeneralizing(aggressiveProp) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
      return failure();
    }
    OpOperand *transposeOperand = nullptr;
    linalg::TransposeOp transposeOp;
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      auto maybeTransposeOp = input->get().getDefiningOp<linalg::TransposeOp>();
      if (maybeTransposeOp && maybeTransposeOp->hasOneUse()) {
        transposeOp = maybeTransposeOp;
        transposeOperand = input;
        break;
      }
    }
    if (!transposeOperand) {
      return rewriter.notifyMatchFailure(linalgOp, "no transpose operand");
    }

    int64_t inputIndex = transposeOperand->getOperandNumber();
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto invPerm = invertPermutationVector(perm);

    // To do the fusion, we can simply apply the permutation of the transpose
    // to the results of the associated input's indexing map, and then forward
    // the input to the transpose to the consumer generic.
    auto maybeGenericOp = getGenericOpOrGeneralizeContraction(
        rewriter, linalgOp, allowGeneralizing);
    if (failed(maybeGenericOp)) {
      return failure();
    }
    auto genericOp = maybeGenericOp.value();
    transposeOperand = genericOp.getDpsInputOperand(inputIndex);
    rewriter.startOpModification(genericOp);

    SmallVector<AffineMap> newIndexingMaps = genericOp.getIndexingMapsArray();
    AffineMap inputMap = genericOp.getMatchingIndexingMap(transposeOperand);
    SmallVector<AffineExpr> newExprs =
        applyPermutation(inputMap.getResults(), invPerm);
    AffineMap transposedMap =
        AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                       newExprs, rewriter.getContext());
    newIndexingMaps[inputIndex] = transposedMap;
    genericOp.setIndexingMapsAttr(
        rewriter.getAffineMapArrayAttr(newIndexingMaps));

    genericOp.setOperand(inputIndex, transposeOp.getInput());
    rewriter.finalizeOpModification(genericOp);
    return success();
  }

private:
  bool allowGeneralizing = false;
};

bool isUnaryElementwiseGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInputs() != 1 ||
      !linalg::isElementwise(genericOp)) {
    return false;
  }

  // Skip transposes and broadcasts. Transposes make more sense to fuse
  // rather than propagate through, and broadcasts are cheaper to transpose
  // before broadcasting.
  if (genericOp.getMatchingIndexingMap(genericOp.getDpsInputOperand(0)) !=
      genericOp.getMatchingIndexingMap(genericOp.getDpsInitOperand(0))) {
    return false;
  }
  return true;
}

// Sinks a transpose through the input of a unary elementwise operation.
class SinkTransposeThroughUnaryElementwiseInput
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
      return failure();
    }

    if (!isUnaryElementwiseGeneric(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "not unary elementwise");
    }

    auto transposeOp =
        genericOp.getDpsInputs()[0].getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return rewriter.notifyMatchFailure(genericOp, "no transpose operand");
    }

    if (!transposeOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          genericOp, "do not propagate multi-use transpose");
    }

    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto invPerm = invertPermutationVector(perm);

    // Create a new empty init for the transposed generic.
    Value newInit =
        createTransposeInit(rewriter, genericOp.getDpsInits()[0], invPerm);

    // We do not need to update indexing maps because this is a unary
    // elementwise op where the input and output maps are the same. Just
    // replace the operands with transposed variants.
    auto newGenericOp = mlir::clone(rewriter, genericOp, newInit.getType(),
                                    {transposeOp.getInput(), newInit});
    rewriter.replaceOp(
        genericOp, createTranspose(rewriter, newGenericOp->getResult(0), perm));
    return success();
  }
};

// Bubbles a transpose through the init of a unary elementwise operation.
class BubbleTransposeThroughUnaryElementwiseDpsInit
    : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp = transposeOp.getInput().getDefiningOp<linalg::GenericOp>();
    if (!genericOp) {
      return failure();
    }
    if (!IREE::Flow::isNonNullAndOutsideDispatch({genericOp, transposeOp})) {
      return failure();
    }

    if (!isUnaryElementwiseGeneric(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp, "not unary elementwise");
    }

    if (!genericOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(genericOp, "not single user");
    }

    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    Value newTranspose =
        createTranspose(rewriter, genericOp.getOperand(0), perm);

    // Create a new empty init for the transposed generic.
    Value newInit =
        createTransposeInit(rewriter, genericOp.getDpsInits()[0], perm);

    // We do not need to update indexing maps because this is a unary
    // elementwise op where the input and output maps are the same. Just
    // replace the operands with transposed variants.
    auto newGenericOp = mlir::clone(rewriter, genericOp, newInit.getType(),
                                    {newTranspose, newInit});
    rewriter.replaceOp(transposeOp, newGenericOp);
    return success();
  }
};

// Folds `linalg.transpose(tensor.empty(), dest)` to `dest`.
class FoldTransposeOfEmpty : public OpRewritePattern<linalg::TransposeOp> {
public:
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!transposeOp.getOperand(0).getDefiningOp<tensor::EmptyOp>()) {
      return rewriter.notifyMatchFailure(
          transposeOp, "transpose source is not tensor empty");
    }
    rewriter.replaceOp(transposeOp, transposeOp.getDpsInits()[0]);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Linalg Named Op -> Named Op Conversions
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy, typename ReplTy, int64_t inputIdx>
class NamedOpConversion : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  NamedOpConversion(MLIRContext *ctx, SmallVector<int64_t> perm,
                    PatternBenefit b = 1)
      : OpRewritePattern<OpTy>(ctx, b), permutation(perm) {}

  LogicalResult matchAndRewrite(OpTy namedOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(namedOp)) {
      return failure();
    }

    Value input = namedOp.getInputs()[inputIdx];
    auto transpose = input.getDefiningOp<linalg::TransposeOp>();
    if (!transpose) {
      return failure();
    }

    SmallVector<int64_t> transPerm(transpose.getPermutation());
    if (transPerm != permutation) {
      return rewriter.notifyMatchFailure(
          namedOp, "transpose permutation does not match target permutation");
    }
    SmallVector<NamedAttribute> attrs = getPrunedAttributeList(namedOp);
    SmallVector<Value> newInputs = namedOp.getInputs();
    newInputs[inputIdx] = transpose.getInput();
    rewriter.replaceOpWithNewOp<ReplTy>(namedOp, newInputs,
                                        namedOp.getDpsInits(), attrs);
    return success();
  }

private:
  // Non-type literal array template parameters are a C++20 feature, so instead
  // all the named op patterns pass their permutation explicitly as a
  // SmallVector.
  SmallVector<int64_t> permutation;
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct PropagateLinalgTransposePass
    : public PropagateLinalgTransposeBase<PropagateLinalgTransposePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  PropagateLinalgTransposePass(bool enableAggressivePropagation) {
    this->enableAggressivePropagation = enableAggressivePropagation;
  }
  PropagateLinalgTransposePass(const PropagateLinalgTransposePass &pass)
      : PropagateLinalgTransposePass(pass.enableAggressivePropagation) {}

  void runOnOperation() override;

private:
  Option<bool> testSinkingOnly{
      *this, "test-sinking-only",
      llvm::cl::desc("Flag used for lit-testing sinking patterns only. "
                     "Not for general usage"),
      llvm::cl::init(false)};
  Option<bool> testBubblingOnly{
      *this, "test-bubbling-only",
      llvm::cl::desc("Flag used for lit-testing bubbling patterns only. "
                     "Not for general usage"),
      llvm::cl::init(false)};
};
} // namespace

static void populateNamedOpSinkingPatterns(MLIRContext *context,
                                           RewritePatternSet &sinkingPatterns) {
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::MatmulOp,
                                /*ReplacementType=*/linalg::MatmulTransposeBOp,
                                /*inputIdx=*/1>>(context,
                                                 SmallVector<int64_t>{1, 0});
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::MatmulOp,
                                /*ReplacementType=*/linalg::MatmulTransposeAOp,
                                /*inputIdx=*/0>>(context,
                                                 SmallVector<int64_t>{1, 0});
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::MatmulTransposeBOp,
                                /*ReplacementType=*/linalg::MatmulOp,
                                /*inputIdx=*/1>>(context,
                                                 SmallVector<int64_t>{1, 0});
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::MatmulTransposeAOp,
                                /*ReplacementType=*/linalg::MatmulOp,
                                /*inputIdx=*/0>>(context,
                                                 SmallVector<int64_t>{1, 0});
  sinkingPatterns.insert<
      NamedOpConversion</*OpType=*/linalg::BatchMatmulOp,
                        /*ReplacementType=*/linalg::BatchMatmulTransposeBOp,
                        /*inputIdx=*/1>>(context,
                                         SmallVector<int64_t>{0, 2, 1});
  sinkingPatterns.insert<
      NamedOpConversion</*OpType=*/linalg::BatchMatmulOp,
                        /*ReplacementType=*/linalg::BatchMatmulTransposeAOp,
                        /*inputIdx=*/0>>(context,
                                         SmallVector<int64_t>{0, 2, 1});
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::BatchMatmulTransposeBOp,
                                /*ReplacementType=*/linalg::BatchMatmulOp,
                                /*inputIdx=*/1>>(context,
                                                 SmallVector<int64_t>{0, 2, 1});
  sinkingPatterns
      .insert<NamedOpConversion</*OpType=*/linalg::BatchMatmulTransposeAOp,
                                /*ReplacementType=*/linalg::BatchMatmulOp,
                                /*inputIdx=*/0>>(context,
                                                 SmallVector<int64_t>{0, 2, 1});
}

static void
populateCommonCanonicalizationPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns) {
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::populateFoldTensorEmptyPatterns(patterns,
                                          /*foldSingleUseOnly=*/false);
  patterns.add<FoldTransposeOfEmpty>(context);
}

void PropagateLinalgTransposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  // First, specialize all transposes to `linalg.transpose`. This dramatically
  // simplifies all subsequent propagation patterns, both in matching and
  // rewriting.
  {
    SmallVector<linalg::GenericOp> genericCandidates;
    funcOp.walk([&](linalg::GenericOp genericOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
        genericCandidates.push_back(genericOp);
      }
    });
    IRRewriter rewriter(&getContext());
    for (auto genericOp : genericCandidates) {
      rewriter.setInsertionPoint(genericOp);
      specializeGenericTransposeOp(rewriter, genericOp);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After specializing transpose ops ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // First try to fuse transposes with some consumer linalg named ops before
  // any reshape propagation. Some transposes may be adjacent to named ops,
  // and it is more canonical if we can fuse the ops into a new named op.
  if (!testBubblingOnly) {
    RewritePatternSet sinkingPatterns(context);
    sinkingPatterns.insert<SinkTransposeThroughExtractSlice>(context);
    sinkingPatterns.insert<SinkTransposeThroughExpandShape>(context);
    populateNamedOpSinkingPatterns(context, sinkingPatterns);
    populateCommonCanonicalizationPatterns(context, sinkingPatterns);
    sinkingPatterns.add<SinkTransposeThroughUnaryElementwiseInput>(
        context, /*benefit=*/2);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(sinkingPatterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After canonicalizing transpose in place ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Propagate transposes upwards, and fuse with any producer generic ops. Also
  // propagate reshapes upwards to open up more transpose fusion opportunities.
  if (!testSinkingOnly) {
    linalg::ControlFusionFn reshapePropagationFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          Operation *consumer = fusedOperand->getOwner();
          if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
            return false;
          }

          // Do not reshape producer linalg op if it has more than one user.
          auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
          if (!producerLinalgOp || !producerLinalgOp->hasOneUse()) {
            return false;
          }
          // Only reshape generic ops, or any op if aggressive propagation is
          // enabled.
          if (!enableAggressivePropagation &&
              !isa<linalg::GenericOp>(producerLinalgOp)) {
            return false;
          }
          // Only propagate expand_shape ops up through producers because it
          // is always possible to bubble a transpose through an collapse_shape
          // and thus is handled separately.
          if (!isa<tensor::ExpandShapeOp>(consumer)) {
            return false;
          }
          // Only propagate if the immediate consumer of the reshape is a
          // transpose.
          return consumer->hasOneUse() &&
                 llvm::isa<linalg::TransposeOp>(*(consumer->user_begin()));
        };
    RewritePatternSet bubblingPatterns(context);
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubblingPatterns,
                                                      reshapePropagationFn);
    linalg::FillOp::getCanonicalizationPatterns(bubblingPatterns, context);
    bubblingPatterns.insert<FuseTransposeWithProducerLinalgOp>(
        context, enableAggressivePropagation);
    bubblingPatterns.insert<BubbleTransposeThroughCollapseShape>(context);
    bubblingPatterns.add<BubbleTransposeThroughUnaryElementwiseDpsInit>(
        context, /*benefit=*/2);
    bubblingPatterns.insert<ComposeTransposes>(context);
    populateCommonCanonicalizationPatterns(context, bubblingPatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(bubblingPatterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After bubbling transpose ops up ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Propagate transposes downwards, and fuse with any non-unary generic ops
  // or linalg named ops. Also propagate reshapes downwards to open up more
  // transpose fusion opportunities.
  if (!testBubblingOnly) {
    RewritePatternSet sinkingPatterns(context);
    linalg::ControlFusionFn reshapePropagationFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          Operation *consumer = fusedOperand->getOwner();
          if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
            return false;
          }
          auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer);
          if (!consumerLinalgOp) {
            return false;
          }
          // Only reshape generic ops.
          if (!enableAggressivePropagation &&
              !isa<linalg::GenericOp>(consumerLinalgOp)) {
            return false;
          }
          // Only propagate collapse_shape ops down through consumers because it
          // is always possible to sink a transpose through an expand_shape and
          // thus is handled separately.
          if (!isa<tensor::CollapseShapeOp>(producer)) {
            return false;
          }
          // Require that the immediate producer of the reshape is a transpose.
          return isa_and_nonnull<linalg::TransposeOp>(
              producer->getOperand(0).getDefiningOp());
        };
    linalg::populateFoldReshapeOpsByExpansionPatterns(sinkingPatterns,
                                                      reshapePropagationFn);
    sinkingPatterns.insert<SinkTransposeThroughExtractSlice>(context);
    sinkingPatterns.insert<SinkTransposeThroughExpandShape>(context);
    sinkingPatterns.insert<FuseTransposeWithLinalgOpConsumer>(
        context, enableAggressivePropagation);
    sinkingPatterns.insert<ComposeTransposes>(context);
    populateNamedOpSinkingPatterns(context, sinkingPatterns);
    populateCommonCanonicalizationPatterns(context, sinkingPatterns);
    sinkingPatterns.add<SinkTransposeThroughUnaryElementwiseInput>(
        context, /*benefit=*/2);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(sinkingPatterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After sinking transpose ops down ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass(bool enableAggressivePropagation) {
  return std::make_unique<PropagateLinalgTransposePass>(
      enableAggressivePropagation);
}

} // namespace mlir::iree_compiler::GlobalOptimization
