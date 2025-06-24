// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
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

/// Returns true iff the linalgOp has a body like a regular matmul, i.e.
/// yield(add(out, mul(cast(in0), cast(in1))))
static bool hasMatmulLikeBody(linalg::LinalgOp linalgOp) {
  auto outBlockArg =
      linalgOp.getMatchingBlockArgument(linalgOp.getDpsInitOperand(0));
  auto yieldOp =
      dyn_cast<linalg::YieldOp>(outBlockArg.getParentBlock()->getTerminator());
  if (!yieldOp) {
    return false;
  }
  Operation *addOp = yieldOp->getOperand(0).getDefiningOp();
  if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Value addLhs = addOp->getOperand(0);
  Value addRhs = addOp->getOperand(1);
  Operation *addLhsOp = addLhs.getDefiningOp();
  Operation *addRhsOp = addRhs.getDefiningOp();
  if (!(addLhsOp && addRhs == outBlockArg) &&
      !(addRhsOp && addLhs == outBlockArg)) {
    return false;
  }
  Operation *mulOp = addLhsOp ? addLhsOp : addRhsOp;
  if (!isa<arith::MulFOp, arith::MulIOp>(mulOp)) {
    return false;
  }
  Value mulLhs = mulOp->getOperand(0);
  Value mulRhs = mulOp->getOperand(1);
  auto mulLhsOp = mulLhs.getDefiningOp<CastOpInterface>();
  auto mulRhsOp = mulRhs.getDefiningOp<CastOpInterface>();
  if (!isa<BlockArgument>(mulLhs) && !mulLhsOp && !isa<BlockArgument>(mulRhs) &&
      !mulRhsOp) {
    return false;
  }
  if ((mulLhsOp && !isa<BlockArgument>(mulLhsOp->getOperand(0))) ||
      (mulRhsOp && !isa<BlockArgument>(mulRhsOp->getOperand(0)))) {
    return false;
  }
  return true;
}

/// Not all contractions are supported by data tiling, so return true if:
///   1) linalgOp has contraction indexingMaps.
///   2) There are not more than one of each contraction dimension
///   3) There is and M or N dimension, and there is a K dimension
///   4) linalgOp has the same body as an ordinary int or float matmul
///
/// These restrictions are required because data tiling currently creates
/// an Mmt4DOp or BatchMmt4DOp on the packed inputs.
///
/// TODO(#16176): Loosen restrictions on contraction ops once data tiling
/// can support more cases.
static LogicalResult isSupportedContractionOp(PatternRewriter &rewriter,
                                              linalg::LinalgOp linalgOp) {
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expected isaContractionOpInterface");
  }
  auto cDims = linalg::inferContractionDims(linalgOp);
  if (failed(cDims) || cDims->batch.size() > 1 || cDims->m.size() > 1 ||
      cDims->n.size() > 1 || cDims->k.size() > 1) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected {|Batch|, |M|, |N|, |K|} <= 1");
  }
  if ((cDims->n.empty() && cDims->m.empty()) || cDims->k.empty()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected M or N dims and K dim to not be empty");
  }
  if (!hasMatmulLikeBody(linalgOp)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expected op to have a matmul body, i.e. yield(add(out, "
                  "mul(cast(in0), cast(in1))))");
  }
  return success();
}

static bool hasWorkgroupCounts(Operation *op) {
  auto parentDispatchOp = op->getParentOfType<IREE::Flow::DispatchRegionOp>();
  return parentDispatchOp && !parentDispatchOp.getWorkgroupCount().empty();
}

namespace {

class SetContractionOpEncoding final
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  explicit SetContractionOpEncoding(MLIRContext *ctx, EncodingOptions &option)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx),
        encodingOption(option) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {

    if (!linalgOp.hasPureTensorSemantics()) {
      return failure();
    }
    if (getCompilationInfo(linalgOp)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the op has preset compilation strategy, skip SetEncoding");
    }
    if (hasWorkgroupCounts(linalgOp.getOperation())) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the op is in a region with workgroup counts, skip "
                    "SetEncoding");
    }
    if (failed(isSupportedContractionOp(rewriter, linalgOp))) {
      return failure();
    }

    auto inputs = linalgOp.getDpsInputs();
    auto outputs = linalgOp.getDpsInits();

    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value out = outputs[0];

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
    SmallVector<int64_t> iterationSizes =
        std::move(maybeIterationSizes.value());

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

private:
  EncodingOptions encodingOption;
};

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
  if (!operandType || operandType.getRank() == 0) {
    return std::nullopt;
  }

  SmallVector<Operation *> opChain;
  auto producerValue = dyn_cast<OpResult>(operand);
  while (producerValue &&
         !isa<IREE::Flow::DispatchRegionOp>(producerValue.getOwner())) {
    if (!llvm::hasSingleElement(producerValue.getUses())) {
      return std::nullopt;
    }

    // If it is an operation that we want to look past, add it to the chain
    // and update the `producerValue`.
    Operation *currOperation = producerValue.getOwner();
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(currOperation)) {
      opChain.push_back(currOperation);
      producerValue = dyn_cast<OpResult>(currOperation->getOperand(0));
      continue;
    }

    // Conservative, bail out.
    return std::nullopt;
  }

  if (!producerValue) {
    return std::nullopt;
  }

  auto producerDispatch =
      dyn_cast<IREE::Flow::DispatchRegionOp>(producerValue.getOwner());
  // TODO(MaheshRavishankar): Multi-result producer dispatches can be supported.
  // Will require to move the consumer dispatch immediately after the producer
  // instead of what is done below and move other operands of the consumer
  // dispatch before the producer dispatch.
  if (!producerDispatch ||
      !llvm::hasSingleElement(producerDispatch.getBody()) ||
      producerDispatch->getNumResults() != 1) {
    return std::nullopt;
  }
  if (!llvm::hasSingleElement(producerValue.getUses())) {
    return std::nullopt;
  }

  Location loc = producerDispatch.getLoc();
  unsigned resultNumber = producerValue.getResultNumber();
  // Compute the padding encoding.  Set to dynamic for backend to pick the right
  // value.
  SmallVector<int64_t> paddingValue(operandType.getRank(), 0);
  paddingValue.back() = ShapedType::kDynamic;
  auto encoding = IREE::Encoding::PadEncodingLayoutAttr::get(
      rewriter.getContext(), paddingValue);

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
SmallVector<unsigned> getOperandsToPad(Operation *op) {
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
    if (!ShapedType::isDynamic(parallelDimSize)) {
      if (ShapedType::isDynamic(dimSize)) {
        parallelDimSize = ShapedType::kDynamic;
        continue;
      }
      parallelDimSize *= dimSize;
    }
  }

  // TODO(MaheshRavishankar): Make this command line controllable.
  static constexpr int64_t kSkinnyMatmulThreshold = 64;
  if (!ShapedType::isDynamic(parallelDimSize) &&
      parallelDimSize < kSkinnyMatmulThreshold) {
    // This matmul is skinny, do not pad.
    return {};
  }

  return {0, 1};
}

// Main driver method to add encodings to pad. Typically these are
// intermediate values produced by `flow.dispatch.region`.
static LogicalResult setPaddingEncodings(MLIRContext *context,
                                         FunctionOpInterface funcOp) {
  IRRewriter rewriter(context);

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

  // Apply the dim resolution patterns.
  RewritePatternSet dimResolutionPatterns(context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(
      dimResolutionPatterns);
  GreedyRewriteConfig config;
  config.enableFolding(true).setMaxIterations(GreedyRewriteConfig::kNoLimit);
  if (failed(applyPatternsGreedily(funcOp, std::move(dimResolutionPatterns),
                                   config))) {
    return funcOp.emitOpError("failed to resolve tensor.dim operations");
  }
  return success();
}

//===---------------------------------------------------------------------===//
// Pass definition
//===---------------------------------------------------------------------===//

struct SetEncodingPass final : impl::SetEncodingPassBase<SetEncodingPass> {
  using Base::Base;
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Implement the padding encoding.
    if (encodingOption == DispatchCreation::EncodingOptions::Padding) {
      if (failed(setPaddingEncodings(context, funcOp))) {
        return signalPassFailure();
      }
      return;
    }

    RewritePatternSet patterns(context);
    patterns.add<SetContractionOpEncoding>(context, encodingOption.getValue());
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.add<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
