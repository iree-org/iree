// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_SWAPSTRIDEDINSERTSLICEWITHCONTRACTIONPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

namespace {

/// Swap a strided `tensor.insert_slice` (tensor.insert_slice into zeros) with
/// its consumer contraction (matmul/conv) when the contraction reads the
/// scattered tensor with a projected permutation on the strided dims. This is
/// valid for 1x1 backward convolutions where the insert_slice commutes with the
/// matmul.
///
/// Before:
///   %scattered = insert_slice %src into %zeros [offs][sizes][strides]
///   %result = contraction(%scattered, %filter)
///   %trunced = truncf(%result)
///
/// After:
///   %small_result = contraction(%src, %filter)
///   %small_trunced = truncf(%small_result)
///   %result = insert_slice %small_trunced into %zeros [offs][sizes][strides]
class SwapStridedInsertSliceWithContraction
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // Destination must be a zero splat constant.
    Value dest = insertOp.getDest();
    Attribute destAttr;
    if (!matchPattern(dest, m_Constant(&destAttr))) {
      return failure();
    }
    auto splatAttr = dyn_cast<SplatElementsAttr>(destAttr);
    if (!splatAttr) {
      return failure();
    }
    Attribute splatVal = splatAttr.getSplatValue<Attribute>();
    bool isZero =
        TypeSwitch<Attribute, bool>(splatVal)
            .Case<FloatAttr>([](auto a) { return a.getValue().isZero(); })
            .Case<IntegerAttr>([](auto a) { return a.getValue().isZero(); })
            .Default([](auto) { return false; });
    if (!isZero) {
      return failure();
    }

    // Must have static metadata with at least one non-unit stride.
    SmallVector<int64_t> offsets(insertOp.getStaticOffsets());
    SmallVector<int64_t> strides(insertOp.getStaticStrides());
    SmallVector<int64_t> sizes(insertOp.getStaticSizes());
    if (ShapedType::isDynamicShape(offsets) ||
        ShapedType::isDynamicShape(strides) ||
        ShapedType::isDynamicShape(sizes)) {
      return failure();
    }
    if (llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
      return failure();
    }

    // The insert_slice result must feed a contraction with single use.
    if (!insertOp->hasOneUse()) {
      return failure();
    }
    Value insertResult = insertOp.getResult();
    // The consumer must be a contraction: 2 inputs, 1 output, reductions,
    // and a mul+add body. We check the body explicitly because
    // isaContractionOpInterface also requires projected permutation maps,
    // which 1x1 backward convs don't have before unit-dim folding.
    auto genericOp = dyn_cast<linalg::GenericOp>(*insertOp->user_begin());
    if (!genericOp || genericOp.getNumDpsInputs() != 2 ||
        genericOp.getNumDpsInits() != 1 ||
        genericOp.getNumReductionLoops() == 0) {
      return failure();
    }
    if (!mlir::linalg::detail::isContractionBody(
            *genericOp.getBlock(), [](Operation *first, Operation *second) {
              return (isa<arith::MulFOp>(first) &&
                      isa<arith::AddFOp>(second)) ||
                     (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second));
            })) {
      return failure();
    }

    // Identify which operand is the insert_slice result.
    OpOperand *scatterOperand = nullptr;
    OpOperand *otherOperand = nullptr;
    for (OpOperand *input : genericOp.getDpsInputOperands()) {
      if (input->get() == insertResult) {
        scatterOperand = input;
      } else {
        otherOperand = input;
      }
    }
    if (!scatterOperand || !otherOperand) {
      return failure();
    }

    Value source = insertOp.getSource();
    auto sourceTy = cast<RankedTensorType>(source.getType());
    auto scatteredTy = cast<RankedTensorType>(insertOp.getResult().getType());
    auto resultTy = cast<RankedTensorType>(genericOp.getResultTypes()[0]);
    AffineMap scatterMap = genericOp.getMatchingIndexingMap(scatterOperand);
    AffineMap resultMap = genericOp.getIndexingMapsArray().back();
    SmallVector<int64_t> loopRanges =
        cast<linalg::LinalgOp>(genericOp.getOperation()).getStaticLoopRanges();
    auto iterTypes = genericOp.getIteratorTypesArray();
    unsigned rank = scatteredTy.getRank();
    Location loc = insertOp.getLoc();

    // For strided dims, the indexing map expression must be equivalent to a
    // single parallel loop dim (after zeroing out unit-range reduction dims).
    // This accepts 1x1 convs (d_spatial + d_kernel with kernel=1, which
    // simplifies to d_spatial) while rejecting 3x3, scaled dims (3*d0), or
    // sums of parallel dims (d0 + d1).
    MLIRContext *ctx = rewriter.getContext();
    DenseMap<AffineExpr, AffineExpr> unitReductionMap;
    for (unsigned i = 0, e = iterTypes.size(); i < e; ++i) {
      if (iterTypes[i] == utils::IteratorType::reduction &&
          i < loopRanges.size() && loopRanges[i] == 1) {
        unitReductionMap[getAffineDimExpr(i, ctx)] =
            getAffineConstantExpr(0, ctx);
      }
    }

    SmallVector<int64_t> newResultShape(resultTy.getShape());
    for (unsigned d = 0; d < rank; d++) {
      if (strides[d] == 1) {
        continue;
      }
      AffineExpr expr = scatterMap.getResult(d).replace(unitReductionMap);
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        return failure();
      }
      unsigned parallelDim = dimExpr.getPosition();
      if (parallelDim >= iterTypes.size() ||
          iterTypes[parallelDim] != utils::IteratorType::parallel) {
        return failure();
      }
      // Compute new result shape by mapping parallel dim through the result
      // map.
      for (auto [resIdx, resExpr] : llvm::enumerate(resultMap.getResults())) {
        auto resDimExpr = dyn_cast<AffineDimExpr>(resExpr);
        if (resDimExpr && resDimExpr.getPosition() == parallelDim) {
          newResultShape[resIdx] = sourceTy.getDimSize(d);
        }
      }
    }
    auto newResultTy =
        RankedTensorType::get(newResultShape, resultTy.getElementType());

    // The contraction init must come from a fill (needed to create the
    // smaller fill).
    rewriter.setInsertionPoint(genericOp);
    auto fillOp =
        genericOp.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
    if (!fillOp) {
      return failure();
    }

    // Helper: clone a linalg.generic with a new result type, replacing its
    // inputs/outputs but preserving the body, maps, and iterator types.
    auto cloneGenericWithShape =
        [&](linalg::GenericOp op, ValueRange inputs, ValueRange outputs,
            RankedTensorType newTy) -> linalg::GenericOp {
      return linalg::GenericOp::create(
          rewriter, loc, newTy, inputs, outputs, op.getIndexingMapsArray(),
          op.getIteratorTypesArray(),
          [&](OpBuilder &b, Location l, ValueRange args) {
            IRMapping mapping;
            for (auto [oldArg, newArg] :
                 llvm::zip(op.getBlock()->getArguments(), args)) {
              mapping.map(oldArg, newArg);
            }
            for (auto &bodyOp : op.getBlock()->without_terminator()) {
              b.clone(bodyOp, mapping);
            }
            auto yield = cast<linalg::YieldOp>(op.getBlock()->getTerminator());
            SmallVector<Value> yieldVals;
            for (Value v : yield.getOperands()) {
              yieldVals.push_back(mapping.lookupOrDefault(v));
            }
            linalg::YieldOp::create(b, l, yieldVals);
          });
    };

    // Create the smaller contraction on the un-scattered source.
    Value newEmpty = tensor::EmptyOp::create(rewriter, loc, newResultShape,
                                             resultTy.getElementType());
    Value newFill = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(),
                                           ValueRange{newEmpty})
                        .getResult(0);
    SmallVector<Value> newInputs;
    for (OpOperand *input : genericOp.getDpsInputOperands()) {
      newInputs.push_back(input == scatterOperand ? source : input->get());
    }
    Value currentResult =
        cloneGenericWithShape(genericOp, newInputs, ValueRange{newFill},
                              newResultTy)
            .getResult(0);

    // Clone the chain of single-input elementwise consumers (e.g., truncf)
    // for the smaller result shape.
    SmallVector<Operation *> consumerChain;
    Operation *lastOp = genericOp;
    while (lastOp->hasOneUse()) {
      auto genericUser = dyn_cast<linalg::GenericOp>(*lastOp->user_begin());
      if (!genericUser || genericUser.getNumDpsInputs() != 1 ||
          genericUser.getNumDpsInits() != 1 ||
          genericUser.getNumReductionLoops() != 0) {
        break;
      }
      // Only follow pure elementwise ops with identity maps. A transpose
      // would invalidate the shape replacement, and linalg.index ops in the
      // body would produce wrong results on the smaller shape.
      if (!llvm::all_of(genericUser.getIndexingMapsArray(),
                        [](AffineMap m) { return m.isIdentity(); })) {
        break;
      }
      if (!genericUser.getBlock()->getOps<linalg::IndexOp>().empty()) {
        break;
      }
      consumerChain.push_back(genericUser);
      auto consumerElemTy =
          cast<RankedTensorType>(genericUser.getResultTypes()[0])
              .getElementType();
      auto newTy = RankedTensorType::get(newResultShape, consumerElemTy);
      Value empty = tensor::EmptyOp::create(rewriter, loc, newResultShape,
                                            consumerElemTy);
      currentResult =
          cloneGenericWithShape(genericUser, ValueRange{currentResult},
                                ValueRange{empty}, newTy)
              .getResult(0);
      lastOp = genericUser;
    }

    // Create the output scatter into zeros.
    auto currentTy = cast<RankedTensorType>(currentResult.getType());
    SmallVector<int64_t> outputShape(rank);
    for (unsigned d = 0; d < rank; d++) {
      outputShape[d] =
          strides[d] != 1 ? scatteredTy.getDimSize(d) : currentTy.getDimSize(d);
    }
    auto outputTy =
        RankedTensorType::get(outputShape, currentTy.getElementType());
    Value outputZeros = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(
            outputTy, rewriter.getZeroAttr(currentTy.getElementType())));
    auto newInsertSlice = tensor::InsertSliceOp::create(
        rewriter, loc, currentResult, outputZeros,
        getAsOpFoldResult(rewriter.getI64ArrayAttr(offsets)),
        getAsOpFoldResult(rewriter.getI64ArrayAttr(newResultShape)),
        getAsOpFoldResult(rewriter.getI64ArrayAttr(strides)));

    // Replace the last op in the chain (the one with external users) and
    // erase everything else in reverse order.
    Operation *replacedOp =
        consumerChain.empty() ? genericOp.getOperation() : consumerChain.back();
    rewriter.replaceOp(replacedOp, newInsertSlice.getResult());
    for (auto it = consumerChain.rbegin(); it != consumerChain.rend(); ++it) {
      if (*it != replacedOp) {
        rewriter.eraseOp(*it);
      }
    }
    if (replacedOp != genericOp.getOperation()) {
      rewriter.eraseOp(genericOp);
    }
    rewriter.eraseOp(insertOp);

    return success();
  }
};

struct SwapStridedInsertSliceWithContractionPass
    : impl::SwapStridedInsertSliceWithContractionPassBase<
          SwapStridedInsertSliceWithContractionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<SwapStridedInsertSliceWithContraction>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
