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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
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

struct FftRfftOpConversion
    : public OpRewritePattern<torch::Torch::AtenFftRfftOp> {
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
        IREE::LinalgExt::rewriteFft(op, builtinCast, fftLength, rewriter);
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

// Utility to add a score modification region to the attention op.
void createScoreModificationRegion(
    PatternRewriter &rewriter, Location loc,
    IREE::LinalgExt::AttentionOp attentionOp,
    std::optional<llvm::StringRef> scoreModSymbol, FloatType floatType,
    const int kAttentionRank) {
  OpBuilder::InsertionGuard g(rewriter);
  Block *block = rewriter.createBlock(&attentionOp.getRegion());

  block->addArgument(floatType, loc);
  rewriter.setInsertionPointToStart(block);

  Value score = block->getArgument(0);
  Value modifiedScore = score;

  if (scoreModSymbol) {
    Type i32Type = rewriter.getI32Type();
    Type si32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
    RankedTensorType scalarTensorType = RankedTensorType::get({}, floatType);
    torch::Torch::ValueTensorType torchScalarType =
        rewriter.getType<torch::Torch::ValueTensorType>(ArrayRef<int64_t>{},
                                                        floatType);
    RankedTensorType i32ScalarTensorType = RankedTensorType::get({}, i32Type);
    torch::Torch::ValueTensorType torchI32ScalarType =
        rewriter.getType<torch::Torch::ValueTensorType>(ArrayRef<int64_t>{},
                                                        si32Type);

    Value scoreTensor = tensor::FromElementsOp::create(
        rewriter, loc, scalarTensorType, ValueRange{score});
    Value torchScore = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, torchScalarType, scoreTensor);

    SmallVector<Value> callArgs;
    callArgs.push_back(torchScore);

    for (unsigned i = 0; i < kAttentionRank; ++i) {
      Value idx = IREE::LinalgExt::IndexOp::create(rewriter, loc, i);
      Value idxI32 = arith::IndexCastOp::create(rewriter, loc, i32Type, idx);
      Value idxTensor = tensor::FromElementsOp::create(
          rewriter, loc, i32ScalarTensorType, ValueRange{idxI32});
      Value torchIdx = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, torchI32ScalarType, idxTensor);
      callArgs.push_back(torchIdx);
    }

    auto callOp =
        func::CallOp::create(rewriter, loc, TypeRange{torchScalarType},
                             scoreModSymbol.value(), ValueRange(callArgs));
    Value torchResult = callOp.getResult(0);

    Value resultTensor = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, scalarTensorType, torchResult);

    modifiedScore =
        tensor::ExtractOp::create(rewriter, loc, resultTensor, ValueRange{});
  }

  IREE::LinalgExt::YieldOp::create(rewriter, loc, modifiedScore);
}

// Utility to create a modified mask tensor.
Value createModifiedMask(PatternRewriter &rewriter, Location loc,
                         MLIRContext *ctx, FlatSymbolRefAttr maskModRef,
                         SmallVector<int64_t> maskShape, FloatType floatType,
                         Value builtinQuery, Value builtInValue, Value zero,
                         const int kAttentionRank) {
  // Create mask tensor [B, H, M, N] with values 0.0 (attend) or -inf
  // (mask).
  RankedTensorType boolScalarTensorType =
      RankedTensorType::get({}, rewriter.getI1Type());
  torch::Torch::ValueTensorType torchBoolScalarType =
      rewriter.getType<torch::Torch::ValueTensorType>(ArrayRef<int64_t>{},
                                                      rewriter.getI1Type());
  Type i32Type = rewriter.getI32Type();
  RankedTensorType i32ScalarTensorType = RankedTensorType::get({}, i32Type);
  Type si32Type =
      IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
  torch::Torch::ValueTensorType torchI32ScalarType =
      rewriter.getType<torch::Torch::ValueTensorType>(ArrayRef<int64_t>{},
                                                      si32Type);

  SmallVector<Value> dynSizes;
  for (int i = 0; i < kAttentionRank-1; ++i) {
    if (maskShape[i] == torch::Torch::kUnknownSize) {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, builtinQuery, i));
    }
  }

  if(maskShape[kAttentionRank-1] == torch::Torch::kUnknownSize) {
    dynSizes.push_back(tensor::DimOp::create(rewriter, loc, builtInValue, kAttentionRank-2));
  }

  Value maskTensor =
      tensor::EmptyOp::create(rewriter, loc, maskShape, floatType, dynSizes);
  // Create linalg.generic to materialize mask.
  SmallVector<AffineMap> maskMaps;
  maskMaps.push_back(AffineMap::getMultiDimIdentityMap(kAttentionRank, ctx));

  SmallVector<utils::IteratorType> iteratorTypes(kAttentionRank,
                                                 utils::IteratorType::parallel);

  Value negInf = arith::ConstantFloatOp::create(
      rewriter, loc, floatType,
      llvm::APFloat::getInf(floatType.getFloatSemantics(),
                            /*Negative=*/true));

  auto maskGeneric = linalg::GenericOp::create(
      rewriter, loc, TypeRange{maskTensor.getType()}, ValueRange{},
      ValueRange{maskTensor}, maskMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        // Get indices and convert to torch tensors.
        SmallVector<Value> torchIndices;
        for (unsigned i = 0; i < kAttentionRank; ++i) {
          Value idx = linalg::IndexOp::create(b, loc, i);
          Value idxI32 =
              arith::IndexCastOp::create(b, loc, rewriter.getI32Type(), idx);
          Value idxTensor = tensor::FromElementsOp::create(
              b, loc, i32ScalarTensorType, ValueRange{idxI32});
          Value torchIdx = torch::TorchConversion::FromBuiltinTensorOp::create(
              b, loc, torchI32ScalarType, idxTensor);
          torchIndices.push_back(torchIdx);
        }

        // Call mask_mod_fn(b, h, q_idx, kv_idx).
        auto callOp =
            func::CallOp::create(b, loc, TypeRange{torchBoolScalarType},
                                 maskModRef, ValueRange(torchIndices));
        Value torchMaskResult = callOp.getResult(0);

        Value maskResult = torch::TorchConversion::ToBuiltinTensorOp::create(
            b, loc, boolScalarTensorType, torchMaskResult);

        Value maskBool =
            tensor::ExtractOp::create(b, loc, maskResult, ValueRange{});

        Value maskValue =
            arith::SelectOp::create(b, loc, maskBool, zero, negInf);

        linalg::YieldOp::create(b, loc, maskValue);
      });

  return maskGeneric.getResult(0);
}

Value convertToBuiltinTensor(PatternRewriter &rewriter, Location loc,
                             Value torchTensor) {
  auto torchType = cast<torch::Torch::ValueTensorType>(torchTensor.getType());
  return torch::TorchConversion::ToBuiltinTensorOp::create(
      rewriter, loc, torchType.toBuiltinTensor(), torchTensor);
}

struct FlexAttentionOpConversion
    : public OpRewritePattern<torch::Torch::HigherOrderFlexAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  // Attention tensors are 4D: [batch, head, query_seq, key_seq].
  static const int kAttentionRank = 4;

  LogicalResult matchAndRewrite(torch::Torch::HigherOrderFlexAttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    Value scaleValue = op.getScale();
    auto scoreModSymbol = op.getScoreModFn();
    auto maskModSymbol = op.getMaskModFn();

    bool returnLseValue;
    if (!matchPattern(op.getReturnLse(),
                      torch::Torch::m_TorchConstantBool(&returnLseValue))) {
      return rewriter.notifyMatchFailure(
          op, "expected return_lse to be a constant bool");
    }

    bool returnMaxScoresValue;
    if (!matchPattern(
            op.getReturnMaxScores(),
            torch::Torch::m_TorchConstantBool(&returnMaxScoresValue))) {
      return rewriter.notifyMatchFailure(
          op, "expected return_max_scores to be a constant bool");
    }

    auto queryType = cast<torch::Torch::ValueTensorType>(query.getType());
    auto keyType = cast<torch::Torch::ValueTensorType>(key.getType());
    auto valueType = cast<torch::Torch::ValueTensorType>(value.getType());

    ArrayRef<int64_t> queryShape = queryType.getSizes();
    ArrayRef<int64_t> valueShape = valueType.getSizes();

    int64_t batch = queryShape[0];
    int64_t numHeads = queryShape[1];
    int64_t seqLenQ = queryShape[2];
    int64_t headDim = queryShape[3];
    int64_t seqLenKV = keyType.getSizes()[2];
    int64_t valueDim = valueShape[3];

    // Dynamic head dim is not supported.
    if (headDim == torch::Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(op, "NYI: dynamic head dimension");
    }

    auto floatType = dyn_cast<FloatType>(queryType.getOptionalDtype());
    // Default scale: 1.0 / sqrt(head_dim).
    double scaleVal;
    if (!matchPattern(scaleValue,
                      torch::Torch::m_TorchConstantFloat(&scaleVal))) {
      scaleVal = 1.0 / std::sqrt(static_cast<double>(headDim));
    }

    Value scale = arith::ConstantOp::create(
        rewriter, loc, floatType, rewriter.getFloatAttr(floatType, scaleVal));

    Value builtinQuery = convertToBuiltinTensor(rewriter, loc, query);
    Value builtinKey = convertToBuiltinTensor(rewriter, loc, key);
    Value builtInValue = convertToBuiltinTensor(rewriter, loc, value);

    // Declare common types for mask and score modification regions.
    Value zero = arith::ConstantFloatOp::create(
        rewriter, loc, floatType,
        llvm::APFloat::getZero(floatType.getFloatSemantics()));
    Value mask;
    if (maskModSymbol) {
      FlatSymbolRefAttr maskModRef =
          FlatSymbolRefAttr::get(ctx, *maskModSymbol);
      SmallVector<int64_t> maskShape = {batch, numHeads, seqLenQ, seqLenKV};
      mask = createModifiedMask(rewriter, loc, ctx, maskModRef, maskShape,
                                floatType, builtinQuery, builtInValue, zero,
                                kAttentionRank);
    }

    // Create output tensor for attention.
    SmallVector<Value> outputDynSizes;
    SmallVector<int64_t> outputShape = {batch, numHeads, seqLenQ, valueDim};
    for (int i = 0; i < kAttentionRank-1; ++i) {
      if (outputShape[i] == torch::Torch::kUnknownSize) {
        outputDynSizes.push_back(
            tensor::DimOp::create(rewriter, loc, builtinQuery, i));
      }
    }
    if (outputShape[kAttentionRank-1] == torch::Torch::kUnknownSize) {
      outputDynSizes.push_back(
          tensor::DimOp::create(rewriter, loc, builtInValue, kAttentionRank-1));
    }

    Value outputTensor =
        tensor::EmptyOp::create(rewriter, loc, outputShape, floatType, outputDynSizes);

    // Build indexing maps for attention.
    // Standard maps: Q, K, V, scale, [mask], output.
    AffineExpr b,
      h, m, n, k1, k2;
    bindDims(ctx, b, h, m, n, k1, k2);

    auto qMap = AffineMap::get(6, 0, {b, h, m, k1}, ctx);
    auto kMap = AffineMap::get(6, 0, {b, h, n, k1}, ctx);
    auto vMap = AffineMap::get(6, 0, {b, h, n, k2}, ctx);
    auto sMap = AffineMap::get(6, 0, {}, ctx);
    auto oMap = AffineMap::get(6, 0, {b, h, m, k2}, ctx);

    SmallVector<AffineMap> indexingMaps = {qMap, kMap, vMap, sMap};
    if (mask) {
      indexingMaps.push_back(AffineMap::get(6, 0, {b, h, m, n}, ctx));
    }

    indexingMaps.push_back(oMap);

    // Create attention op.
    auto attentionOp = IREE::LinalgExt::AttentionOp::create(
        rewriter, loc, outputTensor.getType(), builtinQuery, builtinKey, builtInValue,
        scale, outputTensor, rewriter.getAffineMapArrayAttr(indexingMaps),
        mask);

    createScoreModificationRegion(rewriter, loc, attentionOp, scoreModSymbol,
                                  floatType, kAttentionRank);

    rewriter.setInsertionPointAfter(attentionOp);

    Value normalizedOutput = attentionOp.getResult(0);

    auto outputTorchType =
        queryType.getWithSizesAndDtype(outputShape, floatType);
    Value torchOutput = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, outputTorchType, normalizedOutput);

    // Handle logsumexp.
    // Note: AttentionOp doesn't expose intermediate max/sum
    // values needed for LSE calculation. Return a dummy tensor - logsumexp
    // shape is output_shape[:-1] (remove last dim).
    if (returnLseValue) {
      op.emitWarning("FlexAttention: logsumexp output is a dummy (zeros), "
                     "actual values are not available from AttentionOp");
    }
    // Same goes for max_scores computation from AttentionOp.
    if (returnMaxScoresValue) {
      op.emitWarning("FlexAttention: max_scores output is a dummy (zeros), "
                     "actual values are not available from AttentionOp");
    }
    SmallVector<int64_t> lseShape = outputShape;
    lseShape.pop_back();

    SmallVector<Value> lseDynSizes = outputDynSizes;
    if (ShapedType::isDynamic(outputShape.back())) {
      lseDynSizes.pop_back();
    }

    Value lseTensor =
        tensor::SplatOp::create(rewriter, loc, zero, lseShape, lseDynSizes);

    auto lseTorchType = queryType.getWithSizesAndDtype(lseShape, floatType);
    Value torchLogsumexp = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, lseTorchType, lseTensor);

    rewriter.replaceOp(
        op, {torchOutput, torchLogsumexp, /*max_scores=*/torchLogsumexp});
    return success();
  }
};

class ConvertTorchUnstructuredToLinalgExtPass final
    : public impl::ConvertTorchUnstructuredToLinalgExtPassBase<
          ConvertTorchUnstructuredToLinalgExtPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        IREE::LinalgExt::IREELinalgExtDialect, torch::Torch::TorchDialect,
        tensor::TensorDialect, linalg::LinalgDialect, arith::ArithDialect,
        func::FuncDialect, torch::TorchConversion::TorchConversionDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FftRfftOpConversion, FlexAttentionOpConversion>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
