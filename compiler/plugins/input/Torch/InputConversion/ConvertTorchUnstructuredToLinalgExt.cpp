// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

struct FlexAttentionOpConversion
    : public OpRewritePattern<torch::Torch::AtenFlexAttentionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(torch::Torch::AtenFlexAttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    // Extract operands (new order: query, key, value, 8 block_mask tensors,
    // scale, return_lse)
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    Value scaleValue = op.getScale();
    Value returnLseValue = op.getReturnLse();

    // Get tensor types
    auto queryType = cast<torch::Torch::ValueTensorType>(query.getType());
    auto keyType = cast<torch::Torch::ValueTensorType>(key.getType());
    auto valueType = cast<torch::Torch::ValueTensorType>(value.getType());

    // TODO: See if this check is necessary (Op assertations)
    if (!queryType.hasSizes() || !keyType.hasSizes() || !valueType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected tensors with known sizes");
    }

    ArrayRef<int64_t> queryShape = queryType.getSizes();
    ArrayRef<int64_t> valueShape = valueType.getSizes();

    // TODO: See if this check is necessary (Op assertations)
    // Query shape: [B, H, M, E]
    if (queryShape.size() != 4) {
      return rewriter.notifyMatchFailure(op, "expected 4D query tensor");
    }

    int64_t batch = queryShape[0];
    int64_t numHeads = queryShape[1];
    int64_t seqLenQ = queryShape[2];
    int64_t headDim = queryShape[3];
    int64_t seqLenKV = keyType.getSizes()[2];
    int64_t valueDim = valueShape[3];

    if (headDim == torch::Torch::kUnknownSize) {
      return rewriter.notifyMatchFailure(op, "NYI: dynamic head dimension");
    }

    // Get element type
    Type elementType = queryType.getOptionalDtype();
    if (!elementType || !isa<FloatType>(elementType)) {
      return rewriter.notifyMatchFailure(op, "expected float element type");
    }
    FloatType floatType = cast<FloatType>(elementType);

    // Handle scale parameter
    // Default scale: 1.0 / sqrt(head_dim)
    double scaleVal;
    if (!matchPattern(scaleValue,
                      torch::Torch::m_TorchConstantFloat(&scaleVal)))
      // Fallback to default scale
      scaleVal = 1.0 / std::sqrt(static_cast<double>(headDim));

    Value scale = arith::ConstantOp::create(
        rewriter, loc, floatType, rewriter.getFloatAttr(floatType, scaleVal));

    // TODO: See if this check is necessary (Op assertations)
    // Check return_lse flag
    bool returnLse = false;
    if (!matchPattern(returnLseValue,
                      torch::Torch::m_TorchConstantBool(&returnLse)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant return_lse value");

    // Get function symbols from op attributes (now properly declared in
    // TableGen)
    auto scoreModeSymbol = op.getScoreModFn();
    auto maskModSymbol = op.getMaskModFn();

    // Convert torch tensors to builtin tensors
    auto toBuiltinTensor = [&](Value torchTensor) -> Value {
      auto torchType =
          cast<torch::Torch::ValueTensorType>(torchTensor.getType());
      return torch::TorchConversion::ToBuiltinTensorOp::create(
          rewriter, loc, torchType.toBuiltinTensor(), torchTensor);
    };

    Value builtinQuery = toBuiltinTensor(query);
    Value builtinKey = toBuiltinTensor(key);
    Value builtinValue = toBuiltinTensor(value);

    // Declare common types for mask and score modification regions
    Type i32Type = rewriter.getI32Type();
    Type si32Type =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
    auto scalarTensorType = RankedTensorType::get({}, floatType);
    auto i32ScalarTensorType = RankedTensorType::get({}, i32Type);
    auto boolScalarTensorType = RankedTensorType::get({}, rewriter.getI1Type());
    auto torchScalarType = rewriter.getType<torch::Torch::ValueTensorType>(
        ArrayRef<int64_t>{}, floatType);
    auto torchI32ScalarType = rewriter.getType<torch::Torch::ValueTensorType>(
        ArrayRef<int64_t>{}, si32Type);
    auto torchBoolScalarType = rewriter.getType<torch::Torch::ValueTensorType>(
        ArrayRef<int64_t>{}, rewriter.getI1Type());

    // Create mask if mask_mod_fn is provided
    Value mask;
    if (maskModSymbol) {
      // Create mask tensor [B, H, M, N] with values 0.0 (attend) or -inf (mask)
      SmallVector<int64_t> maskShape = {batch, numHeads, seqLenQ, seqLenKV};
      SmallVector<Value> maskDynSizes;

      // Handling dynamic dimensions
      for (int i = 0; i < 4; ++i) {
        if (maskShape[i] == torch::Torch::kUnknownSize) {
          Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
          Value dim =
              (i < 2) ? tensor::DimOp::create(rewriter, loc, builtinQuery, idx)
              : (i == 2)
                  ? tensor::DimOp::create(rewriter, loc, builtinQuery, idx)
                  : tensor::DimOp::create(
                        rewriter, loc, builtinKey,
                        arith::ConstantIndexOp::create(rewriter, loc, 2));
          maskDynSizes.push_back(dim);
        }
      }

      Value maskTensor = tensor::EmptyOp::create(rewriter, loc, maskShape,
                                                 floatType, maskDynSizes);
      // Create linalg.generic to materialize mask
      SmallVector<AffineMap> maskMaps;
      maskMaps.push_back(
          AffineMap::getMultiDimIdentityMap(4, ctx)); // output map

      SmallVector<utils::IteratorType> iteratorTypes(
          4, utils::IteratorType::parallel);

      Value zero = arith::ConstantOp::create(
          rewriter, loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
      Value negInf = arith::ConstantOp::create(
          rewriter, loc, floatType,
          rewriter.getFloatAttr(floatType,
                                -std::numeric_limits<double>::infinity()));

      auto maskGeneric = linalg::GenericOp::create(
          rewriter, loc, TypeRange{maskTensor.getType()}, ValueRange{},
          ValueRange{maskTensor}, maskMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // Get indices and convert to torch tensors

            SmallVector<Value> torchIndices;
            for (unsigned i = 0; i < 4; ++i) {
              Value idx = linalg::IndexOp::create(b, loc, i);
              Value idxI32 = arith::IndexCastOp::create(b, loc, i32Type, idx);
              Value idxTensor = tensor::FromElementsOp::create(
                  b, loc, i32ScalarTensorType, ValueRange{idxI32});
              Value torchIdx =
                  torch::TorchConversion::FromBuiltinTensorOp::create(
                      b, loc, torchI32ScalarType, idxTensor);
              torchIndices.push_back(torchIdx);
            }

            // Call mask_mod_fn(b, h, q_idx, kv_idx)
            auto callOp =
                func::CallOp::create(b, loc, TypeRange{torchBoolScalarType},
                                     *maskModSymbol, ValueRange(torchIndices));
            Value torchMaskResult = callOp.getResult(0);

            // Convert result back to builtin tensor
            Value maskResult =
                torch::TorchConversion::ToBuiltinTensorOp::create(
                    b, loc, boolScalarTensorType, torchMaskResult);

            // Extract scalar from 0-d tensor
            Value maskBool =
                tensor::ExtractOp::create(b, loc, maskResult, ValueRange{});

            // Select: mask ? 0.0 : -inf
            Value maskValue =
                arith::SelectOp::create(b, loc, maskBool, zero, negInf);

            linalg::YieldOp::create(b, loc, maskValue);
          });

      mask = maskGeneric.getResult(0);
    }

    // Create output tensor for attention
    SmallVector<Value> outputDynSizes;
    SmallVector<int64_t> outputShape = {batch, numHeads, seqLenQ, valueDim};
    // Handle dynamic dimensions
    for (int i = 0; i < 4; ++i) {
      if (outputShape[i] == torch::Torch::kUnknownSize) {
        Value idx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value dim =
            (i == 3) ? tensor::DimOp::create(rewriter, loc, builtinValue, idx)
                     : tensor::DimOp::create(rewriter, loc, builtinQuery, idx);
        outputDynSizes.push_back(dim);
      }
    }

    // Initialize output tensor with identity value (0.0 for addition)
    Value outputInit = arith::getIdentityValue(arith::AtomicRMWKind::addf,
                                               floatType, rewriter, loc,
                                               /*useOnlyFiniteValue=*/true);
    Value outputTensor = tensor::SplatOp::create(rewriter, loc, outputInit,
                                                 outputShape, outputDynSizes);

    // Build indexing maps for attention
    // Standard maps: Q, K, V, scale, [mask], output
    AffineExpr b, h, m, n, k1, k2;
    bindDims(ctx, b, h, m, n, k1, k2);

    auto qMap = AffineMap::get(6, 0, {b, h, m, k1}, ctx);
    auto kMap = AffineMap::get(6, 0, {b, h, n, k1}, ctx);
    auto vMap = AffineMap::get(6, 0, {b, h, n, k2}, ctx);
    auto sMap = AffineMap::get(6, 0, {}, ctx);
    auto oMap = AffineMap::get(6, 0, {b, h, m, k2}, ctx);

    SmallVector<AffineMap> indexingMaps = {qMap, kMap, vMap, sMap};
    // Mask map is optional
    if (mask)
      indexingMaps.push_back(AffineMap::get(6, 0, {b, h, m, n}, ctx));

    indexingMaps.push_back(oMap);

    // Create decomposition config with use_exp2 flag
    // PyTorch's compiled kernels use exp2 for performance, so we match that
    SmallVector<NamedAttribute> decompositionConfigAttrs;
    decompositionConfigAttrs.push_back(
        rewriter.getNamedAttr("use_exp2", rewriter.getBoolAttr(false)));
    DictionaryAttr decompositionConfig =
        rewriter.getDictionaryAttr(decompositionConfigAttrs);

    // Create attention op
    auto attentionOp = IREE::LinalgExt::AttentionOp::create(
        rewriter, loc, outputTensor.getType(), builtinQuery, builtinKey,
        builtinValue, scale, outputTensor,
        rewriter.getAffineMapArrayAttr(indexingMaps), mask);

    // Set decomposition config
    attentionOp.setDecompositionConfigAttr(decompositionConfig);

    {
      OpBuilder::InsertionGuard g(rewriter);
      Block *block = rewriter.createBlock(&attentionOp.getRegion());

      // Add block arguments: score (floatType), b, h, m, n (all index type)
      Type indexType = rewriter.getIndexType();
      block->addArgument(floatType, loc); // score
      for (int i = 0; i < 4; ++i) {
        block->addArgument(indexType, loc); // b (batch index)
      }
      rewriter.setInsertionPointToStart(block);

      Value score = block->getArgument(0);
      SmallVector<Value> indices;
      for (int i = 0; i < 4; ++i) {
        indices.push_back(block->getArgument(i + 1));
      }
      Value modifiedScore = score;

      // If score_mod_fn is provided, call it with indices
      if (scoreModeSymbol) {
        // The score_mod_fn takes (score, b, h, m, n) where m=q_idx, n=kv_idx

        // Convert score to torch tensor
        Value scoreTensor = tensor::FromElementsOp::create(
            rewriter, loc, scalarTensorType, ValueRange{score});
        Value torchScore = torch::TorchConversion::FromBuiltinTensorOp::create(
            rewriter, loc, torchScalarType, scoreTensor);

        // Build arguments: score first, then indices (b, h, m, n)
        SmallVector<Value> callArgs;
        callArgs.push_back(torchScore);

        // Convert index arguments to i32 tensors for torch
        for (Value idx : indices) {
          Value idxI32 =
              arith::IndexCastOp::create(rewriter, loc, i32Type, idx);
          Value idxTensor = tensor::FromElementsOp::create(
              rewriter, loc, i32ScalarTensorType, ValueRange{idxI32});
          Value torchIdx = torch::TorchConversion::FromBuiltinTensorOp::create(
              rewriter, loc, torchI32ScalarType, idxTensor);
          callArgs.push_back(torchIdx);
        }

        // Call score_mod_fn(score, b, h, q_idx, kv_idx)
        auto callOp =
            func::CallOp::create(rewriter, loc, TypeRange{torchScalarType},
                                 *scoreModeSymbol, ValueRange(callArgs));
        Value torchResult = callOp.getResult(0);

        // Convert result back to builtin tensor
        Value resultTensor = torch::TorchConversion::ToBuiltinTensorOp::create(
            rewriter, loc, scalarTensorType, torchResult);

        // Extract scalar from 0-d tensor
        modifiedScore = tensor::ExtractOp::create(rewriter, loc, resultTensor,
                                                  ValueRange{});
      }

      // Yield modified score
      IREE::LinalgExt::YieldOp::create(rewriter, loc, modifiedScore);
    }

    // Set insertion point after the attention op
    rewriter.setInsertionPointAfter(attentionOp);

    // Extract result from attention (already normalized)
    Value normalizedOutput = attentionOp.getResult(0);

    // Convert back to torch tensors
    auto outputTorchType =
        queryType.getWithSizesAndDtype(outputShape, elementType);
    Value torchOutput = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, outputTorchType, normalizedOutput);

    // Handle logsumexp
    // Note: AttentionOp doesn't expose intermediate max/sum values needed for
    // LSE calculation Return a dummy tensor - logsumexp shape is
    // output_shape[:-1] (remove last dim)
    SmallVector<int64_t> lseShape = outputShape;
    lseShape.pop_back(); // Remove last dimension (valueDim)

    SmallVector<Value> lseDynSizes = outputDynSizes;
    if (!outputDynSizes.empty()) {
      lseDynSizes.pop_back(); // Remove last dynamic size if it exists
    }

    // Initialize logsumexp tensor with zeros (dummy values)
    Value zeroInit = arith::ConstantOp::create(
        rewriter, loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
    Value lseTensor =
        tensor::SplatOp::create(rewriter, loc, zeroInit, lseShape, lseDynSizes);

    auto lseTorchType = queryType.getWithSizesAndDtype(lseShape, elementType);
    Value torchLogsumexp = torch::TorchConversion::FromBuiltinTensorOp::create(
        rewriter, loc, lseTorchType, lseTensor);

    rewriter.replaceOp(op, {torchOutput, torchLogsumexp});
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
