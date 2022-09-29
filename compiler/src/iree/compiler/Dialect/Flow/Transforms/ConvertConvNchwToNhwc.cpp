// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-conv-nchw-to-nhwc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

using TransposeIndices = SmallVector<int64_t, 4>;

static const StringLiteral transposeEmptyMarker = "__nchw_to_nhwc_init__";
static const StringLiteral transposePropagateUpMarker = "__nchw_to_nhwc_up__";
static const StringLiteral transposePropagateDownMarker =
    "__nchw_to_nhwc_down__";

static TransposeIndices invertIndices(TransposeIndices targetIndices) {
  auto rank = targetIndices.size();
  TransposeIndices inverted(rank);
  for (auto i : llvm::enumerate(targetIndices)) {
    inverted[i.value()] = i.index();
  }
  return inverted;
}

static TransposeIndices getTransposeIndices(linalg::TransposeOp op) {
  return llvm::to_vector(op.getPermutation());
}

static bool isStaticallyShaped(Value input) {
  if (auto inputType = input.getType().dyn_cast<ShapedType>())
    return inputType.hasStaticShape();
  return false;
}

// Get the transpose indices if the given input comes from a transpose and is
// marked to propagate down.
static Optional<TransposeIndices> getIndicesFromInput(Value input) {
  if (!isStaticallyShaped(input)) return llvm::None;
  auto parent = input.getDefiningOp<linalg::TransposeOp>();
  if (parent && parent->hasAttr(transposePropagateDownMarker))
    return getTransposeIndices(parent);
  return llvm::None;
}

// Get the transpose indices if the given output is used by at least one
// transpose and that transpose is marked to propagate up. Additionally don't
// propagate if there are conflicting transposes.
static Optional<TransposeIndices> getIndicesFromOutput(Value output) {
  if (!isStaticallyShaped(output)) return llvm::None;
  Optional<linalg::TransposeOp> transposedOut;
  if (llvm::all_of(output.getUses(), [&transposedOut](const OpOperand &use) {
        auto owner = dyn_cast<linalg::TransposeOp>(use.getOwner());
        if (owner && owner->hasAttr(transposePropagateUpMarker)) {
          if (transposedOut.has_value()) {
            if (getTransposeIndices(transposedOut.value()) ==
                getTransposeIndices(owner))
              return true;
            return false;
          }
          transposedOut = owner;
          return true;
        }
        return false;
      })) {
    if (transposedOut.has_value())
      return getTransposeIndices(transposedOut.value());
  }
  return llvm::None;
}

// Helper to shuffle vectors according to the transpose indices.
template <typename T>
static SmallVector<T> shuffleFromIndices(SmallVector<T> unshuffled,
                                         TransposeIndices targetIndices) {
  auto rank = unshuffled.size();
  assert(targetIndices.size() == rank &&
         "Mismatch between number of elements in input and number of indices");
  SmallVector<T> shuffled(rank);

  for (auto i : llvm::enumerate(targetIndices)) {
    shuffled[i.index()] = unshuffled[i.value()];
  }
  return shuffled;
}

// Transpose the given tensor based on the given transpose indices. Marks the
// created transpose based on the propagation direction.
static Value createTranspose(PatternRewriter &rewriter, Location loc,
                             Value input, TransposeIndices targetIndices,
                             bool propagateUp) {
  RankedTensorType inType = input.getType().cast<RankedTensorType>();
  auto elementType = inType.getElementType();
  auto inputShape(inType.getShape());

  auto outputShape =
      shuffleFromIndices<int64_t>(llvm::to_vector(inputShape), targetIndices);

  Value output =
      rewriter.create<tensor::EmptyOp>(loc, outputShape, elementType);
  output.getDefiningOp()->setAttr(transposeEmptyMarker, rewriter.getUnitAttr());

  auto transpose =
      rewriter.create<linalg::TransposeOp>(loc, input, output, targetIndices);
  transpose->setAttr(
      propagateUp ? transposePropagateUpMarker : transposePropagateDownMarker,
      rewriter.getUnitAttr());
  return transpose.getResults()[0];
}

// Supports conv and pooling ops, where pooling ops don't transpose the filter.
template <typename ConvOpTy, typename ConvTargetOpTy>
static LogicalResult convertConvLikeNchwToNhwc(PatternRewriter &rewriter,
                                               ConvOpTy convOp,
                                               bool transposeFilter) {
  LLVM_DEBUG(llvm::dbgs() << "inspecting " << convOp << "\n");

  Location loc = convOp.getLoc();

  Value input = convOp.image();
  Value filter = convOp.filter();
  Value output = convOp.getOutputs()[0];

  if (!isStaticallyShaped(input) || !isStaticallyShaped(output) ||
      (transposeFilter && !isStaticallyShaped(filter))) {
    return failure();
  }

  TransposeIndices NCHWIndices = {0, 2, 3, 1};

  auto transposedInput =
      createTranspose(rewriter, loc, input, NCHWIndices, true);
  auto transposedFilter = filter;
  if (transposeFilter) {
    TransposeIndices FCHWIndices = {2, 3, 1, 0};
    transposedFilter =
        createTranspose(rewriter, loc, filter, FCHWIndices, true);
  }
  auto transposedOutput =
      createTranspose(rewriter, loc, output, NCHWIndices, true);

  auto conv =
      rewriter
          .create<ConvTargetOpTy>(loc, transposedOutput.getType(),
                                  ValueRange{transposedInput, transposedFilter},
                                  transposedOutput, convOp.getStrides(),
                                  convOp.getDilations())
          .getResult(0);

  auto returnToNCHW =
      createTranspose(rewriter, loc, conv, invertIndices(NCHWIndices), false);

  rewriter.replaceOp(convOp, returnToNCHW);
  return success();
}

namespace {

/*
 *  Convolution conversion patterns
 */

struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::Conv2DNchwFchwOp,
                                     linalg::Conv2DNhwcHwcfOp>(rewriter, convOp,
                                                               true);
  }
};

struct ConvertLinalgPoolingNchwMax
    : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwMaxOp,
                                     linalg::PoolingNhwcMaxOp>(rewriter, poolOp,
                                                               false);
  }
};

struct ConvertLinalgPoolingNchwSum
    : OpRewritePattern<linalg::PoolingNchwSumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwSumOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwSumOp,
                                     linalg::PoolingNhwcSumOp>(rewriter, poolOp,
                                                               false);
  }
};

/*
 *  Transpose propagation patterns
 */

struct PropagateThroughTensorPadPattern : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  PropagateThroughTensorPadPattern(MLIRContext *context, bool propagateUp)
      : OpRewritePattern<tensor::PadOp>(context), propagateUp(propagateUp) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    TransposeIndices transposeIndices;

    if (propagateUp) {
      auto indices = getIndicesFromOutput(padOp.getResult());
      if (!indices.has_value()) return failure();
      transposeIndices = indices.value();
    } else {
      auto indices = getIndicesFromInput(padOp.getSource());
      if (!indices.has_value()) return failure();
      transposeIndices = invertIndices(indices.value());
    }

    LLVM_DEBUG(llvm::dbgs() << "propagating " << padOp << "\n");

    Location loc = padOp.getLoc();

    auto input = padOp.getSource();
    SmallVector<OpFoldResult> mixedLow = shuffleFromIndices<OpFoldResult>(
        padOp.getMixedLowPad(), transposeIndices);
    SmallVector<OpFoldResult> mixedHigh = shuffleFromIndices<OpFoldResult>(
        padOp.getMixedHighPad(), transposeIndices);

    auto transposedInput =
        createTranspose(rewriter, loc, input, transposeIndices, true);

    SmallVector<int64_t> outputShape(padOp.getResultType().getShape());
    SmallVector<int64_t> transposedOutputShape =
        shuffleFromIndices<int64_t>(outputShape, transposeIndices);
    RankedTensorType transposedOutputType = RankedTensorType::get(
        transposedOutputShape, padOp.getResultType().getElementType());

    auto newPad = rewriter.create<tensor::PadOp>(loc, transposedOutputType,
                                                 transposedInput, mixedLow,
                                                 mixedHigh, padOp.getNofold());
    BlockAndValueMapping mapper;
    padOp.getRegion().cloneInto(&newPad.getRegion(), mapper);

    auto returnToNCHW = createTranspose(rewriter, loc, newPad.getResult(),
                                        invertIndices(transposeIndices), false);

    rewriter.replaceOp(padOp, returnToNCHW);
    return success();
  }

 private:
  bool propagateUp;
};

struct PropagateThroughLinalgFillPattern : OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  PropagateThroughLinalgFillPattern(MLIRContext *context, bool propagateUp)
      : OpRewritePattern<linalg::FillOp>(context), propagateUp(propagateUp) {}

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    TransposeIndices transposeIndices;

    if (propagateUp) {
      auto indices = getIndicesFromOutput(fillOp.getResult(0));
      if (!indices.has_value()) return failure();
      transposeIndices = indices.value();
    } else {
      auto indices = getIndicesFromInput(fillOp.value());
      if (!indices.has_value()) return failure();
      transposeIndices = invertIndices(indices.value());
    }

    LLVM_DEBUG(llvm::dbgs() << "propagating " << fillOp << "\n");
    Location loc = fillOp.getLoc();

    auto transposedOutput =
        createTranspose(rewriter, loc, fillOp.output(), transposeIndices, true);

    auto newTensor =
        rewriter.create<linalg::FillOp>(loc, fillOp.value(), transposedOutput)
            .getResult(0);

    auto returnToNCHW = createTranspose(rewriter, loc, newTensor,
                                        invertIndices(transposeIndices), false);

    rewriter.replaceOp(fillOp, returnToNCHW);
    return success();
  }

 private:
  bool propagateUp;
};

struct PropagateThroughLinalgGenericPattern
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  PropagateThroughLinalgGenericPattern(MLIRContext *context, bool propagateUp)
      : OpRewritePattern<linalg::GenericOp>(context),
        propagateUp(propagateUp) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    TransposeIndices transposeIndices;

    // For now restrict to single results.
    if (genericOp.getNumResults() != 1) return failure();

    if (propagateUp) {
      auto indices = getIndicesFromOutput(genericOp.getOutputs()[0]);
      if (!indices.has_value()) return failure();
      transposeIndices = indices.value();
    } else {
      // TODO: Enable directly fusing the transpose with the inputs.
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "propagating " << genericOp << "\n");

    Location loc = genericOp.getLoc();

    auto transposedOutput = genericOp.getOutputs()[0];
    auto indexingMaps = genericOp.getIndexingMapsArray();

    if (propagateUp) {
      transposedOutput = createTranspose(rewriter, loc, transposedOutput,
                                         transposeIndices, true);

      AffineMap outMap = indexingMaps.back();
      SmallVector<AffineExpr> outExprs(outMap.getResults());
      SmallVector<AffineExpr> exprs =
          shuffleFromIndices<AffineExpr>(outExprs, transposeIndices);
      indexingMaps[indexingMaps.size() - 1] =
          AffineMap::get(outMap.getNumDims(), outMap.getNumSymbols(), exprs,
                         genericOp->getContext());
    }

    SmallVector<Value> newInputs;
    for (auto input : llvm::enumerate(genericOp.getInputs())) {
      newInputs.push_back(input.value());
    }

    SmallVector<utils::IteratorType> iteratorTypes =
        genericOp.getIteratorTypesArray();

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc, transposedOutput.getType().cast<RankedTensorType>(), newInputs,
        transposedOutput, indexingMaps, iteratorTypes);
    BlockAndValueMapping mapper;
    genericOp.getRegion().cloneInto(&newGeneric.getRegion(), mapper);

    Value returnToNCHW = newGeneric.getResult(0);
    if (propagateUp) {
      returnToNCHW = createTranspose(rewriter, loc, returnToNCHW,
                                     invertIndices(transposeIndices), false);
    }

    rewriter.replaceOp(genericOp, returnToNCHW);
    return success();
  }

 private:
  bool propagateUp;
};

struct PropagateThroughTensorEmptyPattern : OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp emptyOp,
                                PatternRewriter &rewriter) const override {
    if (emptyOp->hasAttr(transposeEmptyMarker)) return failure();
    TransposeIndices transposeIndices;

    auto indices = getIndicesFromOutput(emptyOp.getResult());
    if (!indices.has_value()) return failure();
    transposeIndices = indices.value();

    LLVM_DEBUG(llvm::dbgs() << "propagating " << emptyOp << "\n");

    Location loc = emptyOp.getLoc();

    SmallVector<OpFoldResult> mixedSizes = shuffleFromIndices<OpFoldResult>(
        emptyOp.getMixedSizes(), transposeIndices);

    auto newTensor = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, emptyOp.getType().getElementType());
    auto returnToNCHW = createTranspose(rewriter, loc, newTensor.getResult(),
                                        invertIndices(transposeIndices), false);

    rewriter.replaceOp(emptyOp, returnToNCHW);
    return success();
  }
};

/*
 *  Folding away cancelling transposes and generalizing
 */

// Cancel if this transpose is tagged with a propagating tag and the defining op
// for the input is the inverse of this transpose
struct CancelNCHWToNHWCTransposePattern
    : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto transposeIndices = invertIndices(getTransposeIndices(transposeOp));

    auto parentOp =
        transposeOp->getOperand(0).getDefiningOp<linalg::TransposeOp>();
    if (parentOp) {
      if (getTransposeIndices(parentOp) == transposeIndices) {
        rewriter.replaceOp(transposeOp, parentOp->getOperand(0));
        return success();
      }
    }

    return failure();
  }
};

struct GeneralizeTransposeOpPattern : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (transposeOp->hasAttr(transposePropagateUpMarker) ||
        transposeOp->hasAttr(transposePropagateDownMarker)) {
      auto context = rewriter.getContext();
      auto rank =
          transposeOp.getResultTypes()[0].cast<RankedTensorType>().getRank();

      auto transposeIndices = getTransposeIndices(transposeOp);

      SmallVector<AffineExpr> idExprs;
      for (auto i = 0; i < rank; i++)
        idExprs.push_back(getAffineDimExpr(i, context));

      SmallVector<AffineExpr> swapExprs =
          shuffleFromIndices<AffineExpr>(idExprs, transposeIndices);

      SmallVector<AffineMap> indexingMaps = {
          AffineMap::get(rank, 0, idExprs, context),
          AffineMap::get(rank, 0, swapExprs, context)};
      SmallVector<utils::IteratorType> iteratorTypes(
          rank, utils::IteratorType::parallel);

      rewriter.replaceOpWithNewOp<linalg::GenericOp>(
          transposeOp, transposeOp.getResultTypes()[0],
          transposeOp.getOperand(0), transposeOp.getOperand(1), indexingMaps,
          iteratorTypes, [](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
          });
      return success();
    }
    return failure();
  }
};

// The high level strategy for this pass is as follows:
//     1. Do the conversions for all conv_nchw_fchw ops (and pooling ops) and
//     wrap the converted convolutions in transposes. Each transpose is tagged
//     to indicate which direction the transpose should propagate through the
//     graph.
//     2. Traverse the ops in the function in reverse to propagate transposes
//     marked for upwards propagation to their parents. Ideally just before ops
//     such as arith.constant or function arguments.
//     3. Propagate the transposes marked for downward propagation to its users,
//     ideally to just before return.
//     4. Canonicalize out all adjacent cancelling transposes and generalize the
//     remaining transposes to allow for fusing them with nearby ops.
struct ConvertConvNchwToNhwcPass
    : public ConvertConvNchwToNhwcBase<ConvertConvNchwToNhwcPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    Operation *funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.insert<ConvertLinalgConvNchwFchw>(context);
      patterns.insert<ConvertLinalgPoolingNchwMax>(context);
      patterns.insert<ConvertLinalgPoolingNchwSum>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Propagate transposes up the graph.
    {
      SmallVector<Operation *> ops;
      funcOp->walk([&](Operation *op) { ops.push_back(op); });

      RewritePatternSet patterns(context);
      patterns.insert<PropagateThroughTensorPadPattern>(context, true);
      patterns.insert<PropagateThroughTensorEmptyPattern>(context);
      patterns.insert<PropagateThroughLinalgFillPattern>(context, true);
      patterns.insert<PropagateThroughLinalgGenericPattern>(context, true);
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));

      SmallVector<Operation *> reverseOps(llvm::reverse(ops));
      (void)applyOpPatternsAndFold(reverseOps, frozenPatterns, false);
    }

    // Propagate transposes down the graph.
    {
      RewritePatternSet patterns(context);
      patterns.insert<PropagateThroughTensorPadPattern>(context, false);
      patterns.insert<PropagateThroughLinalgFillPattern>(context, false);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // Cancel out transposes.
    {
      RewritePatternSet patterns(context);
      patterns.insert<CancelNCHWToNHWCTransposePattern>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    // Generalize remaining transposes to allow fusion with other ops.
    {
      RewritePatternSet patterns(context);
      patterns.insert<GeneralizeTransposeOpPattern>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertConvNchwToNhwcPass() {
  return std::make_unique<ConvertConvNchwToNhwcPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
