// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHUNSTRUCTUREDTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

struct FftRfftOpConversion : OpRewritePattern<torch::Torch::AtenFftRfftOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(torch::Torch::AtenFftRfftOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value self = op.getSelf();

    int64_t dim;
    Value dimVal = op.getDim();
    if (isa<torch::Torch::NoneType>(dimVal.getType())) {
      dim = -1;
    } else if (!matchPattern(dimVal, torch::Torch::m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: requires dim to be constant");
    }

    if (!isa<torch::Torch::NoneType>(op.getN().getType())) {
      return rewriter.notifyMatchFailure(op, "unimplemented: parameter n");
    }

    if (!isa<torch::Torch::NoneType>(op.getNorm().getType())) {
      return rewriter.notifyMatchFailure(op, "unimplemented: parameter norm");
    }

    auto inputTensorType = cast<torch::Torch::ValueTensorType>(self.getType());
    if (!inputTensorType || !inputTensorType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected input type having sizes");
    }
    ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
    dim += dim < 0 ? inputShape.size() : 0;

    int64_t fftLength = inputShape[dim];
    if (fftLength == torch::Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(op,
                                         "expected known FFT dimension size");
    }
    if (!llvm::isPowerOf2_64(fftLength)) {
      return rewriter.notifyMatchFailure(
          op, "expected FFT length to be a power of two");
    }

    // Transpose if FFT dimension is not the last one
    SmallVector<int64_t> preFftShape(inputShape);
    const int64_t lastDim = inputShape.size() - 1;
    const bool needTranspose = dim != lastDim;
    if (needTranspose) {
      Value cstLastDim = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(lastDim));
      Value cstFftDim = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(dim));
      std::swap(preFftShape[dim], preFftShape[lastDim]);

      self = torch::Torch::AtenTransposeIntOp::create(
          rewriter, loc,
          inputTensorType.getWithSizesAndDtype(preFftShape,
                                               inputTensorType.getDtype()),
          self, cstFftDim, cstLastDim);
    }

    // Cast to the builtin tensor type.
    Value builtinCast = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc,
        cast<torch::Torch::ValueTensorType>(self.getType()).toBuiltinTensor(),
        self);

    auto rewriteRes =
        IREE::LinalgExt::rewriteRfft(op, builtinCast, fftLength, rewriter);
    if (failed(rewriteRes)) {
      return failure();
    }

    auto [real, imag] = rewriteRes.value();

    // Cast back
    SmallVector<int64_t> postFftShape(preFftShape);
    postFftShape.back() = fftLength / 2 + 1;
    Type postFftType = inputTensorType.getWithSizesAndDtype(
        postFftShape, inputTensorType.getDtype());
    Value torchReal = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, postFftType, real);
    Value torchImag = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, postFftType, imag);

    // Unsqueeze a 1 dimension at the end
    SmallVector<int64_t> unsqueezedTensorSizes(postFftShape);
    unsqueezedTensorSizes.push_back(1);
    Type unsqueezedTensorType = inputTensorType.getWithSizesAndDtype(
        unsqueezedTensorSizes, inputTensorType.getDtype());
    Value axisUnsqueeze = torch::Torch::ConstantIntOp::create(
        rewriter, loc, rewriter.getI64IntegerAttr(-1));
    Value unsqueezedReal = torch::Torch::AtenUnsqueezeOp::create(
        rewriter, loc, unsqueezedTensorType, torchReal, axisUnsqueeze);
    Value unsqueezedImag = torch::Torch::AtenUnsqueezeOp::create(
        rewriter, loc, unsqueezedTensorType, torchImag, axisUnsqueeze);

    // Concatenate real and imag
    Type listType = torch::Torch::ListType::get(unsqueezedTensorType);
    Value slices = torch::Torch::PrimListConstructOp::create(
        rewriter, loc, listType,
        llvm::ArrayRef<Value>{unsqueezedReal, unsqueezedImag});
    SmallVector<int64_t> concatenatedTensorSizes(unsqueezedTensorSizes);
    concatenatedTensorSizes.back() = 2;
    Type concatenatedTensorType = inputTensorType.getWithSizesAndDtype(
        concatenatedTensorSizes, inputTensorType.getDtype());
    Value concatenated = torch::Torch::AtenCatOp::create(
        rewriter, loc, concatenatedTensorType, slices, axisUnsqueeze);

    // View as complex (and transpose back)
    SmallVector<int64_t> complexResultSizes(concatenatedTensorSizes);
    complexResultSizes.pop_back();
    torch::Torch::ValueTensorType complexResultType =
        cast<torch::Torch::ValueTensorType>(
            inputTensorType.getWithSizesAndDtype(
                complexResultSizes,
                mlir::ComplexType::get(inputTensorType.getDtype())));
    if (needTranspose) {
      Value complex = torch::Torch::AtenViewAsComplexOp::create(
          rewriter, loc, complexResultType, concatenated);

      Value cstLastDim = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(lastDim));
      Value cstFftDim = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(dim));
      std::swap(complexResultSizes[dim], complexResultSizes[lastDim]);

      rewriter.replaceOpWithNewOp<torch::Torch::AtenTransposeIntOp>(
          op,
          complexResultType.getWithSizesAndDtype(complexResultSizes,
                                                 complexResultType.getDtype()),
          complex, cstFftDim, cstLastDim);
    } else {
      rewriter.replaceOpWithNewOp<torch::Torch::AtenViewAsComplexOp>(
          op, complexResultType, concatenated);
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// FlexAttention -> OnlineAttention conversion
//===----------------------------------------------------------------------===//

static Value convertToBuiltinTensor(PatternRewriter &rewriter, Location loc,
                                    Value torchTensor) {
  auto tensorType = cast<torch::Torch::ValueTensorType>(torchTensor.getType());
  return torch::TorchConversion::ToBuiltinTensorOp::create(
      rewriter, loc, tensorType.toBuiltinTensor(), torchTensor);
}

/// Inline a single-block torch function's body at the current insertion point.
/// Returns the result value. Falls back to func.call for multi-block functions.
static Value inlineTorchFunction(PatternRewriter &rewriter, Location loc,
                                 FlatSymbolRefAttr funcSymbol, ValueRange args,
                                 Operation *contextOp) {
  auto module = contextOp->getParentOfType<ModuleOp>();
  auto funcOp = module.lookupSymbol<func::FuncOp>(funcSymbol);
  if (!funcOp || funcOp.isExternal() || !funcOp.getBody().hasOneBlock()) {
    auto callOp = func::CallOp::create(rewriter, loc, funcSymbol,
                                       funcOp.getResultTypes(), args);
    return callOp.getResult(0);
  }

  Block &entryBlock = funcOp.getBody().front();
  IRMapping mapper;
  for (auto [blockArg, callArg] : llvm::zip(entryBlock.getArguments(), args)) {
    mapper.map(blockArg, callArg);
  }

  for (Operation &op : entryBlock.without_terminator()) {
    rewriter.clone(op, mapper);
  }

  auto returnOp = cast<func::ReturnOp>(entryBlock.getTerminator());
  return mapper.lookupOrDefault(returnOp.getOperand(0));
}

/// Build the score modification region inside the OnlineAttention op.
/// Both mask_mod and score_mod are inlined into the region to avoid separate
/// mask materialization and to enable fusion during attention decomposition.
///
/// The region computes:
///   1. mask = mask_mod_fn(b, h, q_idx, kv_idx)  [if mask_mod present]
///   2. score = select(mask, score, -inf)         [if mask_mod present]
///   3. score = score_mod_fn(score, b, h, q, kv)  [if score_mod present]
///   4. yield score
static void
createScoreModificationRegion(PatternRewriter &rewriter, Location loc,
                              IREE::LinalgExt::OnlineAttentionOp onlineAttnOp,
                              FlatSymbolRefAttr scoreModSymbol,
                              FlatSymbolRefAttr maskModSymbol,
                              FloatType floatType, Operation *contextOp) {
  Region &region = onlineAttnOp.getRegion();
  OpBuilder::InsertionGuard guard(rewriter);
  Block *block =
      rewriter.createBlock(&region, region.end(), {floatType}, {loc});

  Value score = block->getArgument(0);
  bool needIndices = scoreModSymbol || maskModSymbol;

  // Build index torch tensors: b, h, q_idx, kv_idx (dims 0-3).
  SmallVector<Value> torchIndices;
  if (needIndices) {
    auto signlessI32 = rewriter.getIntegerType(32);
    auto signedI32 =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
    auto torchI32Scalar = torch::Torch::ValueTensorType::get(
        rewriter.getContext(), ArrayRef<int64_t>{}, signedI32);
    for (int dim : {0, 1, 2, 3}) {
      Value idx = IREE::LinalgExt::IndexOp::create(rewriter, loc, dim);
      Value idxI32 =
          arith::IndexCastOp::create(rewriter, loc, signlessI32, idx);
      Value idxTensor = tensor::FromElementsOp::create(
          rewriter, loc, RankedTensorType::get({}, signlessI32),
          ValueRange{idxI32});
      Value torchIdx = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, torchI32Scalar, idxTensor);
      torchIndices.push_back(torchIdx);
    }
  }

  // Inline mask_mod: compute mask and apply select(mask, score, -inf).
  if (maskModSymbol) {
    Value maskResult = inlineTorchFunction(rewriter, loc, maskModSymbol,
                                           torchIndices, contextOp);

    auto boolType = rewriter.getIntegerType(1);
    Value builtinBool = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, RankedTensorType::get({}, boolType), maskResult);
    Value boolScalar =
        tensor::ExtractOp::create(rewriter, loc, builtinBool, ValueRange{});

    Value negInf = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(
            floatType,
            APFloat::getInf(floatType.getFloatSemantics(), /*Negative=*/true)));
    score = arith::SelectOp::create(rewriter, loc, boolScalar, score, negInf);
  }

  // Inline score_mod: transform the (possibly masked) score.
  if (scoreModSymbol) {
    auto f32ScalarTensor = RankedTensorType::get({}, floatType);
    auto torchF32Scalar = torch::Torch::ValueTensorType::get(
        rewriter.getContext(), ArrayRef<int64_t>{}, floatType);
    Value scoreTensor = tensor::FromElementsOp::create(
        rewriter, loc, f32ScalarTensor, ValueRange{score});
    Value torchScore = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, torchF32Scalar, scoreTensor);

    SmallVector<Value> scoreArgs = {torchScore};
    scoreArgs.append(torchIndices.begin(), torchIndices.end());
    Value torchResult = inlineTorchFunction(rewriter, loc, scoreModSymbol,
                                            scoreArgs, contextOp);

    Value builtinResult = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, f32ScalarTensor, torchResult);
    score =
        tensor::ExtractOp::create(rewriter, loc, builtinResult, ValueRange{});
  }

  IREE::LinalgExt::YieldOp::create(rewriter, loc, score);
}

struct FlexAttentionOpConversion
    : OpRewritePattern<torch::Torch::HigherOrderFlexAttentionOp> {
  using Base::Base;

  static constexpr int64_t kAttentionRank = 4;

  LogicalResult matchAndRewrite(torch::Torch::HigherOrderFlexAttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    Value scaleVal = op.getScale();

    auto scoreModSymbol = op.getScoreModFnAttr();
    auto maskModSymbol = op.getMaskModFnAttr();

    // Extract return_lse and return_max_scores.
    bool returnLse, returnMaxScores;
    if (!matchPattern(op.getReturnLse(),
                      torch::Torch::m_TorchConstantBool(&returnLse))) {
      return rewriter.notifyMatchFailure(
          op, "expected return_lse to be a constant bool");
    }
    if (!matchPattern(op.getReturnMaxScores(),
                      torch::Torch::m_TorchConstantBool(&returnMaxScores))) {
      return rewriter.notifyMatchFailure(
          op, "expected return_max_scores to be a constant bool");
    }

    // Extract shapes from Q, K, V.
    auto queryType = cast<torch::Torch::ValueTensorType>(query.getType());
    auto valueType = cast<torch::Torch::ValueTensorType>(value.getType());

    ArrayRef<int64_t> queryShape = queryType.getSizes();
    ArrayRef<int64_t> valueShape = valueType.getSizes();

    // Q: [B, H, M, K1], K: [B, H, N, K1], V: [B, H, N, K2]
    int64_t batch = queryShape[0];
    int64_t numHeads = queryShape[1];
    int64_t seqLenQ = queryShape[2];
    int64_t headDim = queryShape[3];
    int64_t valueDim = valueShape[3];

    if (headDim == torch::Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(
          op, "dynamic head dimension not supported");
    }

    auto floatType = Float32Type::get(rewriter.getContext());

    // Resolve scale: try constant float, else default to 1/sqrt(headDim).
    double scaleDouble;
    if (!matchPattern(scaleVal,
                      torch::Torch::m_TorchConstantFloat(&scaleDouble))) {
      scaleDouble = 1.0 / std::sqrt(static_cast<double>(headDim));
    }
    Value scale = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, scaleDouble));

    // Convert Q, K, V to builtin tensors.
    Value builtinQ = convertToBuiltinTensor(rewriter, loc, query);
    Value builtinK = convertToBuiltinTensor(rewriter, loc, key);
    Value builtinV = convertToBuiltinTensor(rewriter, loc, value);

    // 6D iteration space: (b, h, m, n, k1, k2)
    AffineExpr b, h, m, n, k1, k2;
    bindDims(rewriter.getContext(), b, h, m, n, k1, k2);
    int64_t numDims = 6;

    auto getMap = [&](ArrayRef<AffineExpr> results) {
      return AffineMap::get(numDims, 0, results, rewriter.getContext());
    };

    AffineMap qMap = getMap({b, h, m, k1});
    AffineMap kMap = getMap({b, h, n, k1});
    AffineMap vMap = getMap({b, h, n, k2});
    AffineMap scaleMap = AffineMap::get(numDims, 0, rewriter.getContext());
    AffineMap outputMap = getMap({b, h, m, k2});
    AffineMap maxMap = getMap({b, h, m});
    AffineMap sumMap = getMap({b, h, m});

    // No mask operand: mask_mod is inlined into the score region.
    SmallVector<AffineMap> indexingMaps = {qMap, kMap, vMap, scaleMap};
    indexingMaps.push_back(outputMap);
    indexingMaps.push_back(maxMap);
    indexingMaps.push_back(sumMap);

    // Create output tensor.
    auto outputShape = SmallVector<int64_t>{batch, numHeads, seqLenQ, valueDim};
    Value outputEmpty =
        tensor::EmptyOp::create(rewriter, loc, outputShape, floatType);

    // Create and fill max/sum tensors.
    auto rowRedShape = SmallVector<int64_t>{batch, numHeads, seqLenQ};
    Value rowRedEmpty =
        tensor::EmptyOp::create(rewriter, loc, rowRedShape, floatType);

    Value accInit = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, 0.0));
    Value maxInit = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(floatType,
                              APFloat::getLargest(floatType.getFloatSemantics(),
                                                  /*Negative=*/true)));
    Value sumInit = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(floatType, 0.0));

    Value accFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{accInit}, outputEmpty)
            .getResult(0);
    Value maxFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{maxInit}, rowRedEmpty)
            .getResult(0);
    Value sumFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{sumInit}, rowRedEmpty)
            .getResult(0);

    // Create OnlineAttentionOp without a mask operand.
    auto onlineAttnOp = IREE::LinalgExt::OnlineAttentionOp::create(
        rewriter, loc,
        TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
        builtinQ, builtinK, builtinV, scale, /*mask=*/Value(), accFill, maxFill,
        sumFill, rewriter.getAffineMapArrayAttr(indexingMaps),
        /*decomposition_config=*/DictionaryAttr::get(rewriter.getContext()));

    // Build score modification region with inlined mask_mod and score_mod.
    createScoreModificationRegion(rewriter, loc, onlineAttnOp, scoreModSymbol,
                                  maskModSymbol, floatType, op);

    Value attnResult = onlineAttnOp.getResult(0);
    Value maxResult = onlineAttnOp.getResult(1);
    Value sumResult = onlineAttnOp.getResult(2);

    // Post-process: output = (1/sum) * attnResult
    SmallVector<AffineMap> postMaps = compressUnusedDims(
        SmallVector<AffineMap>{sumMap, outputMap, outputMap});
    SmallVector<utils::IteratorType> postIterTypes(
        postMaps[0].getNumDims(), utils::IteratorType::parallel);

    // Determine the output element type from the torch result type.
    auto torchOutputType =
        cast<torch::Torch::ValueTensorType>(op.getOutput().getType());
    auto builtinOutputType = torchOutputType.toBuiltinTensor();
    Type outputElemType =
        cast<RankedTensorType>(builtinOutputType).getElementType();
    Value outputInit =
        tensor::EmptyOp::create(rewriter, loc, outputShape, outputElemType);

    auto normalizeOp = linalg::GenericOp::create(
        rewriter, loc, outputInit.getType(), ValueRange{sumResult, attnResult},
        ValueRange{outputInit}, postMaps, postIterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value one = arith::ConstantOp::create(
              b, loc, b.getFloatAttr(args[0].getType(), 1.0));
          Value reciprocal = arith::DivFOp::create(b, loc, one, args[0]);
          Value result = arith::MulFOp::create(b, loc, reciprocal, args[1]);
          result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                        /*isUnsignedCast=*/false);
          linalg::YieldOp::create(b, loc, result);
        });
    Value normalizedOutput = normalizeOp.getResult(0);

    // Convert output back to torch tensor.
    Value torchOutput = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, torchOutputType, normalizedOutput);

    // Helper to extract ValueTensorType from a possibly-optional result type.
    auto getTensorType = [](Type type) -> torch::Torch::ValueTensorType {
      if (auto vtt = dyn_cast<torch::Torch::ValueTensorType>(type)) {
        return vtt;
      }
      if (auto opt = dyn_cast<torch::Torch::OptionalType>(type)) {
        return dyn_cast<torch::Torch::ValueTensorType>(opt.getContainedType());
      }
      return nullptr;
    };

    // Handle logsumexp: log(sum) + max, shape [B, H, M]
    Value logsumexpResult;
    auto lseType = getTensorType(op.getLogsumexp().getType());
    if (returnLse && lseType) {
      auto builtinLseType = lseType.toBuiltinTensor();
      Type lseElemType =
          cast<RankedTensorType>(builtinLseType).getElementType();
      Value lseInit =
          tensor::EmptyOp::create(rewriter, loc, rowRedShape, lseElemType);

      auto identityMap3D =
          AffineMap::getMultiDimIdentityMap(3, rewriter.getContext());
      SmallVector<AffineMap> lseMaps(3, identityMap3D);
      SmallVector<utils::IteratorType> lseIterTypes(
          3, utils::IteratorType::parallel);

      auto lseOp = linalg::GenericOp::create(
          rewriter, loc, lseInit.getType(), ValueRange{sumResult, maxResult},
          ValueRange{lseInit}, lseMaps, lseIterTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value logSum = math::LogOp::create(b, loc, args[0]);
            Value lse = arith::AddFOp::create(b, loc, logSum, args[1]);
            lse = convertScalarToDtype(b, loc, lse, args[2].getType(),
                                       /*isUnsignedCast=*/false);
            linalg::YieldOp::create(b, loc, lse);
          });
      Value torchLse = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, lseType, lseOp.getResult(0));
      logsumexpResult = torchLse;
    } else {
      logsumexpResult = torch::Torch::ConstantNoneOp::create(rewriter, loc);
    }

    // Handle max_scores: directly from max result.
    Value maxScoresResult;
    auto maxType = getTensorType(op.getMaxScores().getType());
    if (returnMaxScores && maxType) {
      auto builtinMaxType = maxType.toBuiltinTensor();
      Type maxElemType =
          cast<RankedTensorType>(builtinMaxType).getElementType();

      Value maxVal = maxResult;
      if (maxElemType != floatType) {
        Value castInit =
            tensor::EmptyOp::create(rewriter, loc, rowRedShape, maxElemType);
        auto identityMap3D =
            AffineMap::getMultiDimIdentityMap(3, rewriter.getContext());
        auto castOp = linalg::GenericOp::create(
            rewriter, loc, castInit.getType(), ValueRange{maxVal},
            ValueRange{castInit}, SmallVector<AffineMap>(2, identityMap3D),
            SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value casted =
                  convertScalarToDtype(b, loc, args[0], args[1].getType(),
                                       /*isUnsignedCast=*/false);
              linalg::YieldOp::create(b, loc, casted);
            });
        maxVal = castOp.getResult(0);
      }
      maxScoresResult = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, maxType, maxVal);
    } else {
      maxScoresResult = torch::Torch::ConstantNoneOp::create(rewriter, loc);
    }

    // Wrap with DerefineOp if the result type is Optional.
    if (logsumexpResult.getType() != op.getLogsumexp().getType()) {
      logsumexpResult = torch::Torch::DerefineOp::create(
          rewriter, loc, op.getLogsumexp().getType(), logsumexpResult);
    }
    if (maxScoresResult.getType() != op.getMaxScores().getType()) {
      maxScoresResult = torch::Torch::DerefineOp::create(
          rewriter, loc, op.getMaxScores().getType(), maxScoresResult);
    }

    rewriter.replaceOp(op, {torchOutput, logsumexpResult, maxScoresResult});
    return success();
  }
};

class ConvertTorchUnstructuredToLinalgExtPass final
    : public impl::ConvertTorchUnstructuredToLinalgExtPassBase<
          ConvertTorchUnstructuredToLinalgExtPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    torch::Torch::TorchDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect, func::FuncDialect,
                    torch::TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FftRfftOpConversion>(context);
    patterns.add<FlexAttentionOpConversion>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
