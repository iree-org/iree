// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SplitReduction.cpp ----------------------------===//
//
// Split reduction dimension to increase parallelism of a linalg operation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SPLITREDUCTIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitMatmulReductionRatio("iree-dispatch-creation-split-matmul-reduction",
                              llvm::cl::desc("split ratio"), llvm::cl::init(1));

static llvm::cl::opt<int64_t> splitArgmaxReductionRatio(
    "iree-dispatch-creation-split-argmax-reduction",
    llvm::cl::desc("Ratio to split argmax. Set to 0 or 1 to disable"),
    llvm::cl::init(1));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-dispatch-creation-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

template <typename LinalgOpTy>
static FailureOr<linalg::SplitReductionResult>
splitReductionImpl(RewriterBase &rewriter, LinalgOpTy op,
                   linalg::ControlSplitReductionFn controlSplitReductionFn) {
  return linalg::splitReduction(rewriter, op, controlSplitReductionFn);
}

// Matches the combiner pattern in a linalg.generic argmax-style reduction:
// Example MLIR:
// %4:2 = linalg.generic {
//     indexing_maps = [...],
//     iterator_types = ["parallel", "reduction"]
//   } ins(%arg0 : tensor<?x128xbf16>) outs(%1, %3 : tensor<?xbf16>,
//   tensor<?xi64>) {
// ^bb0(%in: bf16, %out: bf16, %out_0: i64):
//   %5 = linalg.index 1 : index
//   %6 = arith.index_cast %5 : index to i64
//   %7 = arith.maximumf %in, %out : bf16
//   %8 = arith.cmpf ogt, %in, %out : bf16
//   %9 = arith.select %8, %6, %out_0 : i64
//   linalg.yield %7, %9 : bf16, i64
// } -> (tensor<?xbf16>, tensor<?xi64>)
//
// This function extracts the `arith.maximumf`, `arith.cmpf`, and `arith.select`
// operations from the body to facilitate transformations such as split
// reduction.
static LogicalResult
collectArgmaxCombinerOps(linalg::GenericOp genericOp,
                         SmallVector<Operation *, 4> &combinerOps) {
  assert(IREE::LinalgExt::isArgmaxOp(genericOp) &&
         "expected operation to be an argmax op");

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

  // Extract max value producer: arith.maximumf.
  Value maxResult = yieldOp.getOperand(0);
  auto maxOp = dyn_cast<arith::MaximumFOp>(maxResult.getDefiningOp());
  if (!maxOp)
    return failure();

  // Extract index result producer: arith.select.
  Value indexResult = yieldOp.getOperand(1);
  auto selectOp = dyn_cast<arith::SelectOp>(indexResult.getDefiningOp());
  if (!selectOp)
    return failure();

  // Extract the condition of the select, expected to be arith.cmpf with
  // predicate OGT.
  auto cmpOp = dyn_cast<arith::CmpFOp>(selectOp.getCondition().getDefiningOp());
  if (!cmpOp || cmpOp.getPredicate() != arith::CmpFPredicate::OGT)
    return failure();

  combinerOps.push_back(maxOp);
  combinerOps.push_back(selectOp);
  combinerOps.push_back(cmpOp);

  return success();
}

template <>
FailureOr<linalg::SplitReductionResult> splitReductionImpl<linalg::GenericOp>(
    RewriterBase &rewriter, linalg::GenericOp genericOp,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  assert(IREE::LinalgExt::isArgmaxOp(genericOp) &&
         "expected operation to be an argmax op");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);
  Location loc = genericOp->getLoc();

  linalg::SplitReductionOptions control = controlSplitReductionFn(genericOp);
  int64_t ratio = control.ratio;
  unsigned insertSplitIndex = control.index;
  unsigned insertSplitDimension = control.index;
  if (ratio <= 1) {
    return rewriter.notifyMatchFailure(
        genericOp, "split ratio needs to be greater than 1");
  }

  SmallVector<unsigned> dims;
  genericOp.getReductionDims(dims);

  if (dims.size() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "needs a single reduction dimension");
  }
  unsigned reductionDim = dims[0];
  if (control.innerParallel) {
    insertSplitDimension = reductionDim + 1;
  }

  SmallVector<int64_t, 4> loopRanges = genericOp.getStaticLoopRanges();
  int64_t reductionDimSize = loopRanges[reductionDim];
  if (reductionDimSize == ShapedType::kDynamic ||
      reductionDimSize % ratio != 0) {
    return rewriter.notifyMatchFailure(
        genericOp, "Reduction dimension not divisible by split ratio");
  }

  if (insertSplitIndex >
      genericOp.getShape(genericOp.getDpsInitOperand(0)).size()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Insert dimension position too large "
                                       "compared to intermediate tensor size");
  }

  SmallVector<Operation *, 4> combinerOps;
  if (failed(collectArgmaxCombinerOps(genericOp, combinerOps)))
    return rewriter.notifyMatchFailure(genericOp, "invalid combiner");

  Operation *reductionOp = combinerOps[0];
  std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
  if (!identity.has_value())
    return rewriter.notifyMatchFailure(
        genericOp, "Unknown identity value for the reduction");

  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;
  // Calculate the new shapes and indexing maps of the input operands.
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    AffineMap map = genericOp.getMatchingIndexingMap(operand);
    SmallVector<int64_t> newShape;
    SmallVector<AffineExpr> exprs;
    SmallVector<ReassociationIndices> reassociation;
    unsigned index = 0;
    for (unsigned idx : llvm::seq<unsigned>(0, map.getNumResults())) {
      unsigned dim = map.getDimPosition(idx);
      if (reductionDim == dim) {
        if (control.innerParallel) {
          newShape.push_back(genericOp.getShape(operand)[idx] /
                             ratio); // reduce
          newShape.push_back(ratio); // parallel (insert)
          exprs.push_back(rewriter.getAffineDimExpr(
              dim < insertSplitDimension ? dim : dim + 1));
          exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
        } else {
          newShape.push_back(ratio); // parallel (insert)
          newShape.push_back(genericOp.getShape(operand)[idx] /
                             ratio); // reduce
          exprs.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
          exprs.push_back(rewriter.getAffineDimExpr(
              dim < insertSplitDimension ? dim : dim + 1));
        }
        reassociation.push_back({index++, index++});
        continue;
      }
      newShape.push_back(genericOp.getShape(operand)[idx]);
      exprs.push_back(rewriter.getAffineDimExpr(
          dim < insertSplitDimension ? dim : dim + 1));
      reassociation.push_back({index++});
    }
    newMaps.push_back(
        AffineMap::get(map.getNumDims() + 1, 0, exprs, genericOp.getContext()));
    // If the shape is unchanged the input doesn't change.
    if (newShape == genericOp.getShape(operand)) {
      newInputs.push_back(operand->get());
      continue;
    }
    Type newType = RankedTensorType::get(
        newShape,
        cast<RankedTensorType>(operand->get().getType()).getElementType());

    Value newInput = rewriter.create<tensor::ExpandShapeOp>(
        loc, newType, operand->get(), reassociation);
    newInputs.push_back(newInput);
  }

  SmallVector<SmallVector<int64_t>> newOutputShapes;
  SmallVector<AffineMap> outputMaps;
  for (int i = 0; i < genericOp.getNumDpsInits(); ++i) {
    OpOperand *output = genericOp.getDpsInitOperand(i);
    AffineMap oldOutputMap = genericOp.getMatchingIndexingMap(output);
    ArrayRef<int64_t> oldShape = genericOp.getShape(output);
    SmallVector<int64_t> thisOutputShape;

    SmallVector<AffineExpr> outputExpr;
    for (unsigned idx = 0; idx <= oldShape.size(); ++idx) {
      if (idx == insertSplitIndex) {
        thisOutputShape.push_back(ratio);
        outputExpr.push_back(rewriter.getAffineDimExpr(insertSplitDimension));
      }
      if (idx < oldShape.size()) {
        thisOutputShape.push_back(oldShape[idx]);
        unsigned dim = oldOutputMap.getDimPosition(idx);
        outputExpr.push_back(rewriter.getAffineDimExpr(
            dim < insertSplitDimension ? dim : dim + 1));
      }
    }

    AffineMap newOutputMap = AffineMap::get(oldOutputMap.getNumDims() + 1, 0,
                                            outputExpr, rewriter.getContext());
    newMaps.push_back(newOutputMap);
    newOutputShapes.push_back(thisOutputShape);
  }

  // Handle dynamic dimensions for identity value tensor.
  SmallVector<Value> dynValDims;
  SmallVector<int64_t> newOutputShape = newOutputShapes[0];
  for (size_t i = 0; i < newOutputShape.size(); ++i) {
    if (ShapedType::isDynamic(newOutputShape[i])) {
      dynValDims.push_back(rewriter.create<tensor::DimOp>(
          loc, genericOp.getDpsInputOperand(0)->get(), i));
    }
  }

  Type valueElemType = genericOp.getRegionOutputArgs()[0].getType();
  Value emptyValTensor = rewriter.create<tensor::EmptyOp>(
      loc, newOutputShape, valueElemType, dynValDims);
  Value constantOp = rewriter.create<arith::ConstantOp>(loc, *identity);
  Value identityVal =
      rewriter.create<linalg::FillOp>(loc, constantOp, emptyValTensor)
          .getResult(0);

  // Handle dynamic dimensions for identity index tensor.
  SmallVector<Value> dynIdxDims;
  newOutputShape = newOutputShapes[1];
  for (size_t i = 0; i < newOutputShape.size(); ++i) {
    if (ShapedType::isDynamic(newOutputShape[i])) {
      dynIdxDims.push_back(rewriter.create<tensor::DimOp>(
          loc, genericOp.getDpsInputOperand(0)->get(), i));
    }
  }
  Type idxElemType = genericOp.getRegionOutputArgs()[1].getType();
  Value zeroIdx = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(idxElemType));
  Value idxInitTensor = rewriter.create<tensor::EmptyOp>(
      loc, newOutputShape, idxElemType, dynIdxDims);
  Value identityIndex =
      rewriter.create<linalg::FillOp>(loc, zeroIdx, idxInitTensor).getResult(0);

  SmallVector<utils::IteratorType> newIteratorTypes;
  for (auto [index, iteratorType] :
       llvm::enumerate(genericOp.getIteratorTypesArray())) {
    if (insertSplitDimension == index)
      newIteratorTypes.push_back(utils::IteratorType::parallel);
    newIteratorTypes.push_back(iteratorType);
  }
  if (insertSplitDimension == genericOp.getIteratorTypesArray().size()) {
    newIteratorTypes.push_back(utils::IteratorType::parallel);
  }

  // Create partial linalg.generic op with global index computation.
  Value tileSize =
      rewriter.create<arith::ConstantIndexOp>(loc, reductionDimSize / ratio);
  auto partialOp = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{identityVal.getType(), identityIndex.getType()}, newInputs,
      ValueRange{identityVal, identityIndex}, newMaps, newIteratorTypes);

  rewriter.inlineRegionBefore(genericOp.getRegion(), partialOp.getRegion(),
                              partialOp.getRegion().begin());

  Block &body = partialOp.getRegion().front();
  rewriter.setInsertionPointToStart(&body);

  unsigned innerIdxDim = reductionDim + 1;
  unsigned outerIdxDim = insertSplitDimension;

  // Compute global index (gidx) for reduction when the original reduction
  // dimension is split into [outerIdx, innerIdx] using `ratio`. This is used to
  // correctly compute the global index for comparisons and index selection.
  Value outerIdx = rewriter.create<linalg::IndexOp>(loc, outerIdxDim);
  Value innerIdx = rewriter.create<linalg::IndexOp>(loc, innerIdxDim);
  Value offset = rewriter.create<arith::MulIOp>(loc, outerIdx, tileSize);
  Value gidx = rewriter.create<arith::AddIOp>(loc, offset, innerIdx);

  auto selectOp = dyn_cast<arith::SelectOp>(combinerOps[1]);
  Value oldIdx = selectOp.getTrueValue();
  Value newIdx = gidx;
  if (oldIdx.getType() != gidx.getType()) {
    newIdx = rewriter.create<arith::IndexCastOp>(loc, oldIdx.getType(), gidx);
  }
  selectOp.setOperand(1, newIdx);
  rewriter.setInsertionPointAfter(partialOp);

  unsigned intermRank = newOutputShape.size();
  AffineMap valueMap = rewriter.getMultiDimIdentityMap(intermRank);
  AffineMap indexMap = valueMap;
  SmallVector<utils::IteratorType> reductionIteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  for (unsigned i : llvm::seq<unsigned>(0, intermRank)) {
    if (insertSplitIndex == i) {
      reductionIteratorTypes.push_back(utils::IteratorType::reduction);
    } else {
      resultExprs.push_back(rewriter.getAffineDimExpr(i));
      reductionIteratorTypes.push_back(utils::IteratorType::parallel);
    }
  }

  AffineMap outputMap =
      AffineMap::get(intermRank, 0, resultExprs, rewriter.getContext());
  SmallVector<AffineMap> finalReductionMaps = {valueMap, indexMap, outputMap,
                                               outputMap};

  assert(combinerOps.size() == 3 &&
         "Expected exactly 3 combiner ops for argmax (max, select, cmp)");
  assert(isa<arith::MaximumFOp>(combinerOps[0]) &&
         "First combiner op must be arith.maximumf");
  assert(isa<arith::SelectOp>(combinerOps[1]) &&
         "Second combiner op must be arith.select");
  assert(isa<arith::CmpFOp>(combinerOps[2]) &&
         "Third combiner op must be arith.cmpf");

  // Create block for final reduction region.
  auto finalReduction = rewriter.create<linalg::GenericOp>(
      loc, genericOp.getResultTypes(),
      ValueRange{partialOp.getResult(0), partialOp.getResult(1)},
      genericOp.getDpsInits(), finalReductionMaps, reductionIteratorTypes,
      [combinerOps](OpBuilder &b, Location loc, ValueRange inputs) {
        Operation *clonedMax = b.clone(*combinerOps[0]);
        clonedMax->setOperands({inputs[0], inputs[2]});
        Operation *clonedCmp = b.clone(*combinerOps[2]);
        clonedCmp->setOperands({inputs[0], inputs[2]});
        Operation *clonedSel = b.clone(*combinerOps[1]);
        clonedSel->setOperands({clonedCmp->getResult(0), inputs[1], inputs[3]});
        b.create<linalg::YieldOp>(
            loc, ValueRange{clonedMax->getResult(0), clonedSel->getResult(0)});
      });

  rewriter.replaceOp(genericOp, finalReduction.getResults());
  // Init or alloc and fillOp are not applicable for argmax op; set to nullptr.
  return linalg::SplitReductionResult{
      /*initOrAlloc*/ nullptr, /*fillOp*/ nullptr,
      cast<linalg::LinalgOp>(partialOp.getOperation()), finalReduction};
}

static LogicalResult splitReductionOnMatmul(
    RewriterBase &rewriter, linalg::MatmulOp op,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList =
      linalg::getPrunedAttributeList(op);

  // Do not transform the matmul ops that have encoded operands.
  auto hasEncoding = [](Type type) -> bool {
    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    return rankedTensorType && rankedTensorType.getEncoding();
  };
  if (llvm::any_of(op.getOperandTypes(), hasEncoding)) {
    return failure();
  }

  FailureOr<linalg::SplitReductionResult> result =
      linalg::splitReduction(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

template <typename LinalgOpTy>
static LogicalResult
splitReductionWrapper(RewriterBase &rewriter, LinalgOpTy op,
                      linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList =
      linalg::getPrunedAttributeList(op);

  // Do not transform the matmul ops that have encoded operands.
  auto hasEncoding = [](Type type) -> bool {
    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    return rankedTensorType && rankedTensorType.getEncoding();
  };
  if (llvm::any_of(op.getOperandTypes(), hasEncoding)) {
    return failure();
  }

  FailureOr<linalg::SplitReductionResult> result =
      splitReductionImpl(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

namespace {
struct SplitReductionPass final
    : public impl::SplitReductionPassBase<SplitReductionPass> {
  void runOnOperation() override {
    if (splitMatmulReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty() &&
        splitArgmaxReductionRatio.getValue() <= 1) {
      return;
    }

    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    SmallVector<linalg::MatmulOp> matmulCandidates;
    SmallVector<IREE::LinalgExt::TopkOp> topkCandidates;
    SmallVector<linalg::GenericOp> argmaxCandidates;

    IRRewriter rewriter(context);
    funcOp->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<linalg::MatmulOp>([&](auto matmulOp) {
            if (splitMatmulReductionRatio > 1) {
              matmulCandidates.push_back(matmulOp);
            }
          })
          .Case<IREE::LinalgExt::TopkOp>([&](auto topkOp) {
            if (!topkSplitReductionRatio.empty()) {
              topkCandidates.push_back(topkOp);
            }
          })
          .Case<linalg::GenericOp>([&](auto genericOp) {
            if (splitArgmaxReductionRatio > 1 &&
                IREE::LinalgExt::isArgmaxOp(genericOp)) {
              argmaxCandidates.push_back(genericOp);
            }
          });
    });

    // Split matmul ops.
    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      // For matmul make the new parallel dimension first so that it looks
      // like a batch_matmul and can follow the same codegen.
      return {int64_t(splitMatmulReductionRatio), 0, /*innerParallel=*/false};
    };
    for (auto op : matmulCandidates) {
      (void)splitReductionWrapper(rewriter, op, matmulSplitReductionControlFn);
    }

    // Split argmax ops.
    auto argmaxSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      return {splitArgmaxReductionRatio, op.getNumLoops() - 1,
              /*innerParallel=*/false};
    };
    for (auto op : argmaxCandidates) {
      if (failed(splitReductionWrapper(rewriter, op,
                                       argmaxSplitReductionControlFn))) {
        op.emitOpError("failed to split argmax operation");
        return signalPassFailure();
      }
    }

    // Split topk ops.
    IREE::LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, topkSplitReductionControlFn);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
