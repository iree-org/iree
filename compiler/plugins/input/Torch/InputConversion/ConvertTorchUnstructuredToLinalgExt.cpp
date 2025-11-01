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
    
    // Extract operands
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    Value scaleValue = op.getScale();
    Value blockMask = op.getBlockMask();
    Value returnLseValue = op.getReturnLse();
    
    // Get tensor types
    auto queryType = cast<torch::Torch::ValueTensorType>(query.getType());
    auto keyType = cast<torch::Torch::ValueTensorType>(key.getType());
    auto valueType = cast<torch::Torch::ValueTensorType>(value.getType());
    
    // TODO: See if this check is necessary (Op assertations)
    if (!queryType.hasSizes() || !keyType.hasSizes() || !valueType.hasSizes()) {
      return rewriter.notifyMatchFailure(op, "expected tensors with known sizes");
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
    if (!matchPattern(scaleValue, torch::Torch::m_TorchConstantFloat(&scaleVal)))
      scaleVal = 1.0 / std::sqrt(static_cast<double>(headDim));
    
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, scaleVal));

    // TODO: See if this check is necessary (Op assertations)
    // Check return_lse flag
    bool returnLse = false;
    if (!matchPattern(returnLseValue, torch::Torch::m_TorchConstantBool(&returnLse)))
      return rewriter.notifyMatchFailure(op, "expected constant return_lse value");
    
    // Get function symbols from op attributes
    auto scoreModeSymbol = op->getAttrOfType<FlatSymbolRefAttr>("score_mod_fn");
    auto maskModSymbol = op->getAttrOfType<FlatSymbolRefAttr>("mask_mod_fn");
    
    // Convert torch tensors to builtin tensors
    auto toBuiltinTensor = [&](Value torchTensor) -> Value {
      auto torchType = cast<torch::Torch::ValueTensorType>(torchTensor.getType());
      return rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(
          loc, torchType.toBuiltinTensor(), torchTensor);
    };
    
    Value builtinQuery = toBuiltinTensor(query);
    Value builtinKey = toBuiltinTensor(key);
    Value builtinValue = toBuiltinTensor(value);
    
    // Declare common types for mask and score modification regions
    Type i32Type = rewriter.getI32Type();
    Type si32Type = IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
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
      
      for (int i = 0; i < 4; ++i) {
        if (maskShape[i] == torch::Torch::kUnknownSize) {
          Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
          Value dim = (i < 2) ? 
              rewriter.create<tensor::DimOp>(loc, builtinQuery, idx) :
              (i == 2) ? rewriter.create<tensor::DimOp>(loc, builtinQuery, idx) :
              rewriter.create<tensor::DimOp>(loc, builtinKey, rewriter.create<arith::ConstantIndexOp>(loc, 2));
          maskDynSizes.push_back(dim);
        }
      }
      
      Value maskTensor = rewriter.create<tensor::EmptyOp>(
          loc, maskShape, floatType, maskDynSizes);
      
      // Create linalg.generic to materialize mask
      SmallVector<AffineMap> maskMaps;
      maskMaps.push_back(AffineMap::getMultiDimIdentityMap(4, ctx)); // output map
      
      SmallVector<utils::IteratorType> iteratorTypes(4, utils::IteratorType::parallel);

      Value zero = rewriter.create<arith::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
      Value negInf = rewriter.create<arith::ConstantOp>(
        loc, floatType, 
        rewriter.getFloatAttr(floatType, -std::numeric_limits<double>::infinity()));

      auto maskGeneric = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{maskTensor.getType()}, ValueRange{}, ValueRange{maskTensor},
        maskMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Get indices and convert to torch tensors
          
          SmallVector<Value> torchIndices;
          for (unsigned i = 0; i < 4; ++i) {
            Value idx = b.create<linalg::IndexOp>(loc, i);
            Value idxI32 = b.create<arith::IndexCastOp>(loc, i32Type, idx);
            Value idxTensor = b.create<tensor::FromElementsOp>(
                loc, i32ScalarTensorType, ValueRange{idxI32});
            Value torchIdx = b.create<torch::TorchConversion::FromBuiltinTensorOp>(
                loc, torchI32ScalarType, idxTensor);
            torchIndices.push_back(torchIdx);
          }
          
          // Call mask_mod_fn(b, h, q_idx, kv_idx)
          auto callOp = b.create<func::CallOp>(
              loc, maskModSymbol, TypeRange{torchBoolScalarType},
              ValueRange(torchIndices));
          Value torchMaskResult = callOp.getResult(0);
          
          // Convert result back to builtin tensor
          Value maskResult = b.create<torch::TorchConversion::ToBuiltinTensorOp>(
              loc, boolScalarTensorType, torchMaskResult);
          
          // Extract scalar from 0-d tensor
          Value maskBool = b.create<tensor::ExtractOp>(loc, maskResult, ValueRange{});
          
          // Select: mask ? 0.0 : -inf
          Value maskValue = b.create<arith::SelectOp>(loc, maskBool, zero, negInf);
          
          b.create<linalg::YieldOp>(loc, maskValue);
      });
    
      mask = maskGeneric.getResult(0);
    }
    
    // Create output, max, and sum tensors for online_attention
    SmallVector<Value> outputDynSizes;
    SmallVector<int64_t> outputShape = {batch, numHeads, seqLenQ, valueDim};
    for (int i = 0; i < 4; ++i) {
      if (outputShape[i] == torch::Torch::kUnknownSize) {
        Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value dim = (i == 3) ?
            rewriter.create<tensor::DimOp>(loc, builtinValue, idx) :
            rewriter.create<tensor::DimOp>(loc, builtinQuery, idx);
        outputDynSizes.push_back(dim);
      }
    }
    
    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputShape, floatType, outputDynSizes);
    
    // Create max and sum tensors [B, H, M]
    SmallVector<int64_t> maxSumShape = {batch, numHeads, seqLenQ};
    SmallVector<Value> maxSumDynSizes;
    for (int i = 0; i < 3; ++i) {
      if (maxSumShape[i] == torch::Torch::kUnknownSize) {
        Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
        Value dim = rewriter.create<tensor::DimOp>(loc, builtinQuery, idx);
        maxSumDynSizes.push_back(dim);
      }
    }
    
    Value maxTensor = rewriter.create<tensor::EmptyOp>(
        loc, maxSumShape, floatType, maxSumDynSizes);
    Value sumTensor = rewriter.create<tensor::EmptyOp>(
        loc, maxSumShape, floatType, maxSumDynSizes);
    
    // Build indexing maps for online_attention
    // Standard maps: Q, K, V, scale, [mask], output, max, sum
    AffineExpr b, h, m, n, k1, k2;
    bindDims(ctx, b, h, m, n, k1, k2);
    
    auto qMap = AffineMap::get(6, 0, {b, h, m, k1}, ctx);
    auto kMap = AffineMap::get(6, 0, {b, h, n, k1}, ctx);
    auto vMap = AffineMap::get(6, 0, {b, h, n, k2}, ctx);
    auto sMap = AffineMap::get(6, 0, {}, ctx);
    auto oMap = AffineMap::get(6, 0, {b, h, m, k2}, ctx);
    auto maxMap = AffineMap::get(6, 0, {b, h, m}, ctx);
    auto sumMap = AffineMap::get(6, 0, {b, h, m}, ctx);
    
    SmallVector<AffineMap> indexingMaps = {qMap, kMap, vMap, sMap};
    // Mask map is optional
    if (mask)
      indexingMaps.push_back(AffineMap::get(6, 0, {b, h, m, n}, ctx));
    
    indexingMaps.push_back(oMap);
    indexingMaps.push_back(maxMap);
    indexingMaps.push_back(sumMap);
    
    // Create online_attention op
    auto onlineAttnType = TypeRange{outputTensor.getType(), maxTensor.getType(), sumTensor.getType()};
    auto onlineAttn = IREE::LinalgExt::OnlineAttentionOp::create(
        rewriter, loc, onlineAttnType,
        builtinQuery, builtinKey, builtinValue, scale,
        mask, outputTensor, maxTensor, sumTensor,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        nullptr);
    
    // Create region for score_mod
    {
      OpBuilder::InsertionGuard g(rewriter);
      auto *block = rewriter.createBlock(&onlineAttn.getRegion());
      block->addArgument(floatType, loc);
      rewriter.setInsertionPointToStart(block);
      
      Value score = block->getArgument(0);
      Value modifiedScore = score;
      
      // If score_mod_fn is provided, call it with indices
      if (scoreModeSymbol) {
        // Get iteration indices using iree_linalg_ext.index
        // The online_attention iteration domain is (b, h, m, n, k1, k2)
        // but score_mod_fn takes (score, b, h, m, n) where m=q_idx, n=kv_idx
        
        // Convert score to torch tensor
        Value scoreTensor = rewriter.create<tensor::FromElementsOp>(
            loc, scalarTensorType, ValueRange{score});
        Value torchScore = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, torchScalarType, scoreTensor);
        
        // Build arguments: score first, then indices (b, h, m, n)
        SmallVector<Value> callArgs;
        callArgs.push_back(torchScore);
        for (unsigned i = 0; i < 4; ++i) {
          Value idx = rewriter.create<IREE::LinalgExt::IndexOp>(loc, i);
          Value idxI32 = rewriter.create<arith::IndexCastOp>(loc, i32Type, idx);
          Value idxTensor = rewriter.create<tensor::FromElementsOp>(
              loc, i32ScalarTensorType, ValueRange{idxI32});
          Value torchIdx = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
              loc, torchI32ScalarType, idxTensor);
          callArgs.push_back(torchIdx);
        }
        
        // Call score_mod_fn(score, b, h, q_idx, kv_idx)
        auto callOp = rewriter.create<func::CallOp>(
            loc, scoreModeSymbol, TypeRange{torchScalarType},
            ValueRange(callArgs));
        Value torchResult = callOp.getResult(0);
        
        // Convert result back to builtin tensor
        Value resultTensor = rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(
            loc, scalarTensorType, torchResult);
        
        // Extract scalar from 0-d tensor
        modifiedScore = rewriter.create<tensor::ExtractOp>(loc, resultTensor, ValueRange{});
      }
      
      // Yield modified score
      rewriter.create<IREE::LinalgExt::YieldOp>(loc, modifiedScore);
    }
    
    // Set insertion point after the online_attention op
    rewriter.setInsertionPointAfter(onlineAttn);
    
    // Extract results from online_attention
    Value unnormalizedOutput = onlineAttn.getResult(0);
    Value max = onlineAttn.getResult(1);
    Value sum = onlineAttn.getResult(2);
    
    // Normalize output: output = unnormalizedOutput / sum
    // Use linalg.generic to broadcast sum over output
    auto identityMap4D = AffineMap::getMultiDimIdentityMap(4, ctx);
    SmallVector<AffineMap> normMaps;
    normMaps.push_back(identityMap4D);                          // unnormalized output
    normMaps.push_back(AffineMap::get(4, 0, {b, h, m}, ctx));  // sum (broadcast over last dim)
    normMaps.push_back(identityMap4D);                          // normalized output
    
    Value normalizedOutputTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputShape, floatType, outputDynSizes);
    
    auto normGeneric = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{normalizedOutputTensor.getType()},
        ValueRange{unnormalizedOutput, sum}, ValueRange{normalizedOutputTensor},
        normMaps, SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value unnorm = args[0];
          Value sumVal = args[1];
          Value norm = b.create<arith::DivFOp>(loc, unnorm, sumVal);
          b.create<linalg::YieldOp>(loc, norm);
        });
    
    Value normalizedOutput = normGeneric.getResult(0);
    
    // Convert back to torch tensors
    auto outputTorchType = queryType.getWithSizesAndDtype(outputShape, elementType);
    Value torchOutput = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, outputTorchType, normalizedOutput);
    
    // Handle logsumexp if return_lse is true
    Value torchLogsumexp;
    if (returnLse) {
      // logsumexp = max + log(sum)
      auto identityMap3D = AffineMap::getMultiDimIdentityMap(3, ctx);
      auto logsumexpGeneric = rewriter.create<linalg::GenericOp>(
          loc, TypeRange{maxTensor.getType()},
          ValueRange{max, sum}, ValueRange{maxTensor},
          SmallVector<AffineMap>{identityMap3D, identityMap3D, identityMap3D},
          SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value logSum = b.create<math::LogOp>(loc, args[1]);
            Value lse = b.create<arith::AddFOp>(loc, args[0], logSum);
            b.create<linalg::YieldOp>(loc, lse);
          });
      
      Value logsumexp = logsumexpGeneric.getResult(0);
      auto lseTorchType = queryType.getWithSizesAndDtype(maxSumShape, elementType);
      torchLogsumexp = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
          loc, lseTorchType, logsumexp);
    } else {
      // Return None for logsumexp
      torchLogsumexp = rewriter.create<torch::Torch::ConstantNoneOp>(loc);
    }
    
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
