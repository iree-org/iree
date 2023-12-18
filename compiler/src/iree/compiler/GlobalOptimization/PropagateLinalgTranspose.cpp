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
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
static Value createTranspose(OpBuilder &builder, Value source,
                             ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
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
// Transpose propagation
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
      return failure();
    }

    ArrayRef<int64_t> staticSizes = extractOp.getStaticSizes();
    ArrayRef<int64_t> sliceShape = extractOp.getResultType().getShape();
    llvm::SmallDenseSet<unsigned> rankReducingMask =
        *mlir::computeRankReductionMask(staticSizes, sliceShape);

    int64_t dim = 0;
    llvm::SmallDenseMap<int64_t, int64_t> rankReducedMap;
    for (int64_t i = 0, e = perm.size(); i < e; ++i) {
      if (!rankReducingMask.contains(i)) {
        rankReducedMap[i] = dim++;
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
      return failure();
    }

    ArrayRef<int64_t> perm = transposeOp.getPermutation();

    auto invPerm = invertPermutationVector(perm);
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
    // reassociation_map = [[0, 1, 2], [4, 5], [3]]
    // permutation = [0, 1, 2, 3, 4, 5]
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

// Fuses a transpose with the input of a linalg.generic op. For example,
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
class FuseTransposeWithGenericConsumer
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
      return failure();
    }
    OpOperand *transposeOperand = nullptr;
    linalg::TransposeOp transposeOp;
    for (int64_t i = 0, e = genericOp.getInputs().size(); i < e; ++i) {
      OpOperand *inputOperand = &genericOp->getOpOperand(i);
      Value input = inputOperand->get();
      transposeOp = input.getDefiningOp<linalg::TransposeOp>();
      if (transposeOp) {
        transposeOperand = inputOperand;
        break;
      }
    }
    if (!transposeOperand) {
      return failure();
    }

    int64_t inputIndex = transposeOperand->getOperandNumber();
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto invPerm = invertPermutationVector(perm);

    // To do the fusion, we can simply apply the permutation of the transpose
    // to the results of the associated input's indexing map, and then forward
    // the input to the transpose to the consumer generic.
    rewriter.startRootUpdate(genericOp);

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
    rewriter.finalizeRootUpdate(genericOp);
    return success();
  }
};

// Sinks a transpose to the input of a linalg named op. The conditions for the
// rewrite are
//   1) One of the input producers to the named op is a linalg.transpose
//   2) The named op is generalizable (and is not a transpose)
// The easiest way to get the rewrite we want then is to just try to generalize
// all transposed named ops and let the generic pattern handle the actual
// rewrite.
class GeneralizeInputTransposedNamedOp
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
      return failure();
    }
    // Don't generalize transposes.
    if (isa<linalg::TransposeOp>(linalgOp)) {
      return failure();
    }
    bool hasTranspose = false;
    for (Value input : linalgOp.getDpsInputs()) {
      if (input.getDefiningOp<linalg::TransposeOp>()) {
        hasTranspose = true;
        break;
      }
    }
    if (!hasTranspose) {
      return failure();
    }
    if (failed(linalg::generalizeNamedOp(rewriter, linalgOp))) {
      return failure();
    }
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Linalg Named Op -> Named Op Conversions
//===----------------------------------------------------------------------===//

namespace {

template <typename OpTy, typename ReplTy, int64_t inputIdx, int64_t N>
class NamedOpConversion : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;
  NamedOpConversion(MLIRContext *ctx, std::array<int64_t, N> perm,
                    PatternBenefit b = 1)
      : OpRewritePattern<OpTy>(ctx, b), permutation(perm) {}

  LogicalResult matchAndRewrite(OpTy namedOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(namedOp)) {
      return failure();
    }
    Value input = namedOp.getInputs()[inputIdx];
    SmallVector<NamedAttribute> attrs = getPrunedAttributeList(namedOp);
    if (auto transpose = input.getDefiningOp<linalg::TransposeOp>()) {
      if (llvm::all_of(llvm::zip_equal(transpose.getPermutation(), permutation),
                       [](std::tuple<int64_t, int64_t> it) {
                         return std::get<0>(it) == std::get<1>(it);
                       })) {
        SmallVector<Value> newInputs = namedOp.getInputs();
        newInputs[inputIdx] = transpose.getInput();
        rewriter.replaceOpWithNewOp<ReplTy>(namedOp, newInputs,
                                            namedOp.getDpsInits(), attrs);
        return success();
      }
    }
    return failure();
  }

private:
  // non-type literal array template parameters are a C++20 feature.
  std::array<int64_t, N> permutation;
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
};
} // namespace

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

  {
    RewritePatternSet sinkingPatterns(context);
    sinkingPatterns.insert<
        NamedOpConversion</*OpType=*/linalg::MatmulOp,
                          /*ReplacementType=*/linalg::MatmulTransposeBOp,
                          /*inputIdx=*/1,
                          /*inputRank=*/2>>(context,
                                            std::array<int64_t, 2>{1, 0});
    sinkingPatterns.insert<
        NamedOpConversion</*OpType=*/linalg::MatmulOp,
                          /*ReplacementType=*/linalg::MatmulTransposeAOp,
                          /*inputIdx=*/0,
                          /*inputRank=*/2>>(context,
                                            std::array<int64_t, 2>{1, 0});
    sinkingPatterns
        .insert<NamedOpConversion</*OpType=*/linalg::MatmulTransposeBOp,
                                  /*ReplacementType=*/linalg::MatmulOp,
                                  /*inputIdx=*/1,
                                  /*inputRank=*/2>>(
            context, std::array<int64_t, 2>{1, 0});
    sinkingPatterns
        .insert<NamedOpConversion</*OpType=*/linalg::MatmulTransposeAOp,
                                  /*ReplacementType=*/linalg::MatmulOp,
                                  /*inputIdx=*/0,
                                  /*inputRank=*/2>>(
            context, std::array<int64_t, 2>{1, 0});
    sinkingPatterns.insert<
        NamedOpConversion</*OpType=*/linalg::BatchMatmulOp,
                          /*ReplacementType=*/linalg::BatchMatmulTransposeBOp,
                          /*inputIdx=*/1,
                          /*inputRank=*/3>>(context,
                                            std::array<int64_t, 3>{0, 2, 1});
    sinkingPatterns.insert<
        NamedOpConversion</*OpType=*/linalg::BatchMatmulOp,
                          /*ReplacementType=*/linalg::BatchMatmulTransposeAOp,
                          /*inputIdx=*/0,
                          /*inputRank=*/3>>(context,
                                            std::array<int64_t, 3>{0, 2, 1});
    sinkingPatterns
        .insert<NamedOpConversion</*OpType=*/linalg::BatchMatmulTransposeBOp,
                                  /*ReplacementType=*/linalg::BatchMatmulOp,
                                  /*inputIdx=*/1,
                                  /*inputRank=*/3>>(
            context, std::array<int64_t, 3>{0, 2, 1});
    sinkingPatterns
        .insert<NamedOpConversion</*OpType=*/linalg::BatchMatmulTransposeAOp,
                                  /*ReplacementType=*/linalg::BatchMatmulOp,
                                  /*inputIdx=*/0,
                                  /*inputRank=*/3>>(
            context, std::array<int64_t, 3>{0, 2, 1});
    sinkingPatterns.insert<SinkTransposeThroughExtractSlice>(context);
    sinkingPatterns.insert<SinkTransposeThroughExpandShape>(context);
    sinkingPatterns.insert<FuseTransposeWithGenericConsumer>(context);
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

  // Pre-empting additional patterns for bubbling transposes.
  // if (!testSinkingOnly) {
  //   ...
  // }

  // Currently this only runs after all propagation has finished. There are
  // cases where combining transposes can allow further propagation, but
  // similarly there are cases where combining adjacent transposes limits later
  // propagation patterns. For now this keeps it simple as once propagation has
  // finished, it should in all cases be better to fuse.
  // TODO: Run this to some kind of fixed point with propagation. This is tricky
  // because propagation can make trivial modifications to the IR (e.g. through
  // reshapes).
  {
    RewritePatternSet patterns(context);
    patterns.insert<ComposeTransposes>(context);
    if (enableAggressivePropagation) {
      patterns.insert<GeneralizeInputTransposedNamedOp>(context);
    }
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After propagating transpose ops ---\n";
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
