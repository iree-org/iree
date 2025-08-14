// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-set-encoding"

namespace mlir::iree_compiler::DispatchCreation {
#define GEN_PASS_DEF_SETENCODINGPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

using IREE::Encoding::EncodingAttr;
using IREE::Encoding::MatmulKAttr;

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

static Value setEncoding(OpBuilder &builder, Location loc, Value source,
                         Attribute encodingAttr) {
  auto resultType =
      cast<RankedTensorType>(source.getType()).cloneWithEncoding(encodingAttr);
  return builder.create<IREE::Encoding::SetEncodingOp>(loc, resultType, source);
};

static Value unsetEncoding(OpBuilder &builder, Location loc, Value source,
                           SmallVector<OpFoldResult> sizes) {
  SmallVector<Value> dynamicSizesVec;
  SmallVector<int64_t> staticSizesVec;
  dispatchIndexOpFoldResults(sizes, dynamicSizesVec, staticSizesVec);

  auto sourceType = cast<RankedTensorType>(source.getType());
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  return builder.create<IREE::Encoding::UnsetEncodingOp>(
      loc, unsetEncodingReturnType, source, dynamicSizesVec);
}

/// Given a LinalgOp and one of its OpOperands, return the element type,
/// inferring unsignedness from the body of the LinalgOp
static Type getContractionInputTypeWithSignedness(OpBuilder &builder,
                                                  linalg::LinalgOp linalgOp,
                                                  OpOperand *operand) {
  assert(linalg::isaContractionOpInterface(linalgOp));
  assert(operand->getOwner() == linalgOp.getOperation());
  auto elemType = getElementTypeOrSelf(operand->get().getType());
  // Infer if unsigned from body ops
  Value blockArg = linalgOp.getMatchingBlockArgument(operand);
  for (auto bodyCastOp : blockArg.getParentBlock()->getOps<arith::ExtUIOp>()) {
    if (bodyCastOp->getOperand(0) == blockArg) {
      return builder.getIntegerType(elemType.getIntOrFloatBitWidth(),
                                    /*isSigned=*/false);
    }
  }
  return elemType;
}

static SmallVector<linalg::LinalgOp>
getDataTilingCandidates(FunctionOpInterface funcOp) {
  SmallVector<linalg::LinalgOp> result;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (!IREE::Encoding::hasDataTilingHint(op)) {
      return;
    }
    result.push_back(op);
  });
  return result;
}

static LogicalResult setDataTilingEncodings(RewriterBase &rewriter,
                                            linalg::LinalgOp linalgOp,
                                            EncodingOptions encodingOption) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value out = linalgOp.getDpsInitOperand(0)->get();
  Type lhsElemType = getContractionInputTypeWithSignedness(
      rewriter, linalgOp, linalgOp.getDpsInputOperand(0));
  Type rhsElemType = getContractionInputTypeWithSignedness(
      rewriter, linalgOp, linalgOp.getDpsInputOperand(1));
  Type outElemType = getContractionInputTypeWithSignedness(
      rewriter, linalgOp, linalgOp.getDpsInitOperand(0));
  if (!lhsElemType || !rhsElemType || !outElemType) {
    return failure();
  }
  SmallVector<Type> elemTypes = {lhsElemType, rhsElemType, outElemType};

  // The `iteration_sizes` are the linalg op's static loop ranges. From the
  // combination of `iteration_sizes` and `user_indexing_maps`, we can later
  // derive information such as the iteration size of the M/N dimensions of a
  // matmul-like operation for example.
  FailureOr<SmallVector<int64_t>> maybeIterationSizes =
      linalgOp.getStaticLoopRanges();
  if (failed(maybeIterationSizes)) {
    return failure();
  }
  SmallVector<int64_t> iterationSizes = std::move(maybeIterationSizes.value());

  Location loc = linalgOp.getLoc();
  SmallVector<AffineMap> maps = linalgOp.getIndexingMapsArray();

  auto opType = IREE::Encoding::EncodingOpType::matmul;
  auto setEncodingWrapper = [&](Value src, int64_t operandIndex) -> Value {
    MLIRContext *ctx = linalgOp.getContext();
    Attribute encoding;
    switch (encodingOption) {
    case EncodingOptions::Generic: {
      encoding = EncodingAttr::get(ctx, operandIndex, opType, elemTypes, maps,
                                   iterationSizes);
      break;
    }
    case EncodingOptions::MatmulK: {
      SmallVector<int32_t> kDims;
      AffineMap indexingMap = maps[operandIndex];
      auto cDims = linalg::inferContractionDims(linalgOp);
      for (auto k : cDims->k) {
        std::optional<unsigned> dimIdx =
            indexingMap.getResultPosition(rewriter.getAffineDimExpr(k));
        if (!dimIdx) {
          continue;
        }
        kDims.push_back(dimIdx.value());
      }
      encoding = MatmulKAttr::get(ctx, kDims);
      break;
    }
    default: {
      assert(false && "Unsupported encoding option");
      return Value();
    }
    }
    return setEncoding(rewriter, loc, src, encoding);
  };
  auto encodedLhs = setEncodingWrapper(lhs, IREE::Encoding::MATMUL_LHS);
  auto encodedRhs = setEncodingWrapper(rhs, IREE::Encoding::MATMUL_RHS);
  auto encodedOut = setEncodingWrapper(out, IREE::Encoding::MATMUL_RESULT);
  Value opTiled = clone(rewriter, linalgOp, encodedOut.getType(),
                        ValueRange{encodedLhs, encodedRhs, encodedOut})
                      ->getResult(0);

  // Sizes are computed by original output size.
  SmallVector<OpFoldResult> outSizes =
      tensor::getMixedSizes(rewriter, loc, out);
  Value result = unsetEncoding(rewriter, loc, opTiled, outSizes);

  rewriter.replaceOp(linalgOp, result);
  return success();
}

namespace {
/// Pattern to fold a `linalg.fill` -> `iree_encoding.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding final
    : OpRewritePattern<IREE::Encoding::SetEncodingOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Encoding::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::getMixedSizes(rewriter, loc, fillOp.getOutputs()[0]);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(encodingOp, fillOp.getInputs(),
                                                ValueRange{newEmptyOp});
    return success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Set padding encodings
//===---------------------------------------------------------------------===//

struct PaddedValue {
  Value paddedValue;
  SmallVector<Value> dynamicDims;
};

// For a given `operand`, if its producer is a `flow.dispatch.region`,
// generate a new value for `operand` that has the padding encoding.
// The producer `flow.dispatch.region` need not be the immediate defining op
// of `operand`. This method tracks through operations like
// `tensor.expand_shape/tensor.collapse_shape` to get to the producer dispatch.
// Once the producer dispatch is found, its result is modified to be of the same
// type as `operand` but with the padding encodings. To keep things consistent,
// the operations that are encountered before getting to the original producing
// `flow.dispatch.region` are replicated into the producer dispatch.
static std::optional<PaddedValue> padProducerOfValue(RewriterBase &rewriter,
                                                     Value operand) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  std::optional<std::pair<OpResult, SmallVector<Operation *>>>
      maybeProducerDispatchAndOpChain =
          getProducerDispatchValueAndOpChain(operand);
  if (!maybeProducerDispatchAndOpChain) {
    return std::nullopt;
  }
  OpResult producerValue;
  SmallVector<Operation *> opChain;
  std::tie(producerValue, opChain) =
      std::move(maybeProducerDispatchAndOpChain.value());
  auto producerDispatch =
      cast<IREE::Flow::DispatchRegionOp>(producerValue.getOwner());

  Location loc = producerDispatch.getLoc();
  unsigned resultNumber = producerValue.getResultNumber();
  // Compute the padding encoding.  Set to dynamic for backend to pick the right
  // value.
  SmallVector<int64_t> paddingValue(operandType.getRank(), 0);
  paddingValue.back() = ShapedType::kDynamic;
  auto encoding =
      IREE::Encoding::PaddingAttr::get(rewriter.getContext(), paddingValue);

  // Compute the result types of the new dispatch.
  auto newResultType = operandType.cloneWithEncoding(encoding);
  auto newResultTypes = llvm::to_vector(producerDispatch->getResultTypes());
  newResultTypes[resultNumber] = newResultType;

  // Compute the result dynamic dims.
  SmallVector<OpFoldResult> operandDims =
      tensor::getMixedSizes(rewriter, loc, operand);
  SmallVector<Value> operandDynamicDims;
  std::tie(std::ignore, operandDynamicDims) = decomposeMixedValues(operandDims);

  SmallVector<Value> newResultDynamicDims;
  for (OpResult producerDispatchResult : producerDispatch.getResults()) {
    if (producerDispatchResult != producerValue) {
      llvm::append_range(newResultDynamicDims,
                         producerDispatch.getResultDynamicDims(
                             producerDispatchResult.getResultNumber()));
      continue;
    }
    llvm::append_range(newResultDynamicDims, operandDynamicDims);
  }

  auto newDispatchOp = rewriter.create<IREE::Flow::DispatchRegionOp>(
      producerDispatch->getLoc(), newResultTypes, newResultDynamicDims,
      producerDispatch.getWorkload());

  // Move over the body of the old dispatch.
  Region &newBody = newDispatchOp.getBody();
  Region &producerDispatchBody = producerDispatch.getBody();
  rewriter.cloneRegionBefore(producerDispatchBody, newBody, newBody.begin());

  // Move over the slice operations if needed.
  Region &producerWorkgroupCountBody = producerDispatch.getWorkgroupCount();
  if (!producerWorkgroupCountBody.empty()) {
    Region &newWorkgroupCountBody = newDispatchOp.getWorkgroupCount();
    rewriter.cloneRegionBefore(producerWorkgroupCountBody,
                               newWorkgroupCountBody,
                               newWorkgroupCountBody.begin());
  }

  // Clone the operation chain.
  IRMapping map;
  auto returnOp = cast<IREE::Flow::ReturnOp>(newBody.front().getTerminator());
  Value yieldedVal = returnOp.getOperand(resultNumber);

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(returnOp);
  map.map(producerValue, yieldedVal);

  for (Operation *op : llvm::reverse(opChain)) {
    rewriter.clone(*op, map);
  }
  // Find the new value to yield.
  Value newYieldedVal = map.lookup(operand);
  auto encodingOp = rewriter.create<IREE::Encoding::SetEncodingOp>(
      returnOp->getLoc(), newResultType, newYieldedVal);
  rewriter.modifyOpInPlace(
      returnOp, [&]() { returnOp.setOperand(resultNumber, encodingOp); });

  return PaddedValue{newDispatchOp->getResult(resultNumber),
                     operandDynamicDims};
}

// For a given operation, pad the operands corresponding to `operandNums`.
static SmallVector<unsigned> padOperandsOfOp(RewriterBase &rewriter,
                                             Operation *op,
                                             ArrayRef<unsigned> operandNums) {
  // Do not pad the operands of operations not within dispatches.
  auto dispatchOp = op->getParentOfType<IREE::Flow::DispatchRegionOp>();
  if (!dispatchOp) {
    return {};
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dispatchOp);

  SmallVector<unsigned> paddedOperands;
  for (auto operandNum : operandNums) {
    OpOperand &operand = op->getOpOperand(operandNum);
    std::optional<PaddedValue> paddedVal =
        padProducerOfValue(rewriter, operand.get());
    if (!paddedVal) {
      continue;
    }
    rewriter.modifyOpInPlace(op, [&]() {
      OpBuilder::InsertionGuard g2(rewriter);
      rewriter.setInsertionPoint(op);
      Type operandType = operand.get().getType();
      auto unsetEncodignOp = rewriter.create<IREE::Encoding::UnsetEncodingOp>(
          op->getLoc(), operandType, paddedVal->paddedValue,
          paddedVal->dynamicDims);
      op->setOperand(operandNum, unsetEncodignOp.getResult());
    });
  }
  return paddedOperands;
}

// Return a list of operands to be padded for each `op`.
static SmallVector<unsigned> getOperandsToPad(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp)) {
    return {};
  }

  // Bail out on matvec / vecmat and skinny matmul problems.
  // TODO(MaheshRavishankar): There is a possibility to use a more specialized
  // encoding for making this decision late and using `SpecializeEncoding`
  // as the place where this gets resolved to the pad encoding. That was
  // initially tried, but rolled back cause the design of the encoding needs to
  // be flushed out a bit more. So currently moving this logic during set
  // encoding itself. Also this logic might require clean up (see
  // https://github.com/iree-org/iree/pull/20732/files#r2087241983)
  SmallVector<unsigned> reductionDims;
  linalgOp.getReductionDims(reductionDims);
  int64_t parallelDimSize = 1;
  llvm::SmallSetVector<int32_t, 4> reductionDimsSet;
  reductionDimsSet.insert_range(reductionDims);
  SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
  for (auto [idx, dimSize] : llvm::enumerate(loopRanges)) {
    if (reductionDimsSet.contains(idx)) {
      // Bail if the reduction dimension is dynamic.
      if (ShapedType::isDynamic(dimSize)) {
        return {};
      }
      continue;
    }
    if (ShapedType::isStatic(parallelDimSize)) {
      if (ShapedType::isDynamic(dimSize)) {
        parallelDimSize = ShapedType::kDynamic;
        continue;
      }
      parallelDimSize *= dimSize;
    }
  }

  // TODO(MaheshRavishankar): Make this command line controllable.
  static constexpr int64_t kSkinnyMatmulThreshold = 64;
  if (ShapedType::isStatic(parallelDimSize) &&
      parallelDimSize < kSkinnyMatmulThreshold) {
    // This matmul is skinny, do not pad.
    return {};
  }

  // Do not pad if the producer dispatch region contains an attention operation
  // as that results in complicated multi-dimensional load/store access patterns
  // which aren't well supported with padding.
  // TODO(#21149): Try to generalize this by either providing better support in
  // codegen or catching these load/store access patterns cases in a more
  // general way.
  SmallVector<unsigned> operandsNum = {0, 1};
  for (unsigned operandNum : operandsNum) {
    OpOperand &operand = op->getOpOperand(operandNum);
    std::optional<std::pair<OpResult, SmallVector<Operation *>>>
        dispatchAndOpChain = getProducerDispatchValueAndOpChain(operand.get());
    if (dispatchAndOpChain.has_value()) {
      auto producerDispatch = cast<IREE::Flow::DispatchRegionOp>(
          dispatchAndOpChain->first.getOwner());
      WalkResult res =
          producerDispatch->walk([&](IREE::LinalgExt::AttentionOp op) {
            return WalkResult::interrupt();
          });
      if (res.wasInterrupted()) {
        return {};
      }
    }
  }

  return operandsNum;
}

// Main driver method to add encodings to pad. Typically these are
// intermediate values produced by `flow.dispatch.region`.
static LogicalResult setPaddingEncodings(RewriterBase &rewriter,
                                         FunctionOpInterface funcOp) {
  // Collect all operations whose operands can be padded.
  using OpListType =
      SmallVector<std::tuple<Operation *, SmallVector<unsigned>>>;
  OpListType paddedOps;
  funcOp.walk([&](Operation *op) {
    SmallVector<unsigned> paddedOperands = getOperandsToPad(op);
    if (paddedOperands.empty()) {
      return;
    }
    paddedOps.emplace_back(std::tuple{op, std::move(paddedOperands)});
  });
  for (auto [op, operandsNums] : paddedOps) {
    // Only pad LHS or RHS of matmul ops.
    padOperandsOfOp(rewriter, op, operandsNums);
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Pass definition
//===---------------------------------------------------------------------===//

namespace {
struct SetEncodingPass final : impl::SetEncodingPassBase<SetEncodingPass> {
  using Base::Base;
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    RewritePatternSet postPatterns(context);

    switch (encodingOption) {
    case EncodingOptions::Generic:
    case EncodingOptions::MatmulK: {
      SmallVector<linalg::LinalgOp> candidates =
          getDataTilingCandidates(funcOp);
      for (linalg::LinalgOp linalgOp : candidates) {
        if (failed(
                setDataTilingEncodings(rewriter, linalgOp, encodingOption))) {
          return signalPassFailure();
        }
      }
      linalg::FillOp::getCanonicalizationPatterns(postPatterns, context);
      postPatterns.add<FoldFillWithSetEncoding>(context);
      break;
    }
    case EncodingOptions::Padding: {
      if (failed(setPaddingEncodings(rewriter, funcOp))) {
        return signalPassFailure();
      }
      break;
    }
    }

    memref::populateResolveRankedShapedTypeResultDimsPatterns(postPatterns);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(postPatterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
