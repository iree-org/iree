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
/// Falls back to func.call for multi-block or external functions.
static SmallVector<Value> inlineTorchFunction(PatternRewriter &rewriter,
                                              Location loc,
                                              FlatSymbolRefAttr funcSymbol,
                                              ValueRange args,
                                              Operation *contextOp) {
  auto module = contextOp->getParentOfType<ModuleOp>();
  auto funcOp = module.lookupSymbol<func::FuncOp>(funcSymbol);
  if (!funcOp || funcOp.isExternal() || !funcOp.getBody().hasOneBlock()) {
    auto callOp = func::CallOp::create(rewriter, loc, funcSymbol,
                                       funcOp.getResultTypes(), args);
    return SmallVector<Value>(callOp->getResults());
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
  SmallVector<Value> results;
  for (Value operand : returnOp.getOperands()) {
    results.push_back(mapper.lookupOrDefault(operand));
  }
  return results;
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
                                           torchIndices, contextOp)[0];

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
                                            scoreArgs, contextOp)[0];

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
    int64_t valueDim = valueShape[3];

    auto floatType = Float32Type::get(rewriter.getContext());

    Value builtinQ = convertToBuiltinTensor(rewriter, loc, query);
    Value builtinK = convertToBuiltinTensor(rewriter, loc, key);
    Value builtinV = convertToBuiltinTensor(rewriter, loc, value);

    // Resolve scale: try constant float, else compute rsqrt(headDim).
    Value scale;
    double scaleDouble;
    if (matchPattern(scaleVal,
                     torch::Torch::m_TorchConstantFloat(&scaleDouble))) {
      scale = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(floatType, scaleDouble));
    } else {
      int64_t queryRank = queryShape.size();
      Value dimIdx =
          tensor::DimOp::create(rewriter, loc, builtinQ, queryRank - 1);
      Value dimI64 = arith::IndexCastOp::create(rewriter, loc,
                                                rewriter.getI64Type(), dimIdx);
      Value dimF32 = arith::SIToFPOp::create(rewriter, loc, floatType, dimI64);
      scale = math::RsqrtOp::create(rewriter, loc, dimF32);
    }

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

    Value zeroInit = arith::getIdentityValue(arith::AtomicRMWKind::addf,
                                             floatType, rewriter, loc);
    Value maxInit = arith::getIdentityValue(arith::AtomicRMWKind::maximumf,
                                            floatType, rewriter, loc,
                                            /*useOnlyFiniteValue=*/true);

    Value accFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{zeroInit}, outputEmpty)
            .getResult(0);
    Value maxFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{maxInit}, rowRedEmpty)
            .getResult(0);
    Value sumFill =
        linalg::FillOp::create(rewriter, loc, ValueRange{zeroInit}, rowRedEmpty)
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

    // Handle logsumexp: log(sum) + max, shape [B, H, M]
    Value logsumexpResult;
    if (returnLse) {
      auto lseType =
          cast<torch::Torch::ValueTensorType>(op.getLogsumexp().getType());
      Value lseInit = tensor::EmptyOp::create(rewriter, loc, rowRedShape,
                                              lseType.getDtype());

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
            linalg::YieldOp::create(b, loc, lse);
          });
      logsumexpResult = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, lseType, lseOp.getResult(0));
    } else {
      logsumexpResult = torch::Torch::ConstantNoneOp::create(rewriter, loc);
    }

    // Handle max_scores: directly from max result.
    Value maxScoresResult;
    if (returnMaxScores) {
      auto maxType =
          cast<torch::Torch::ValueTensorType>(op.getMaxScores().getType());
      maxScoresResult = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, maxType, maxResult);
    } else {
      maxScoresResult = torch::Torch::ConstantNoneOp::create(rewriter, loc);
    }

    rewriter.replaceOp(op, {torchOutput, logsumexpResult, maxScoresResult});
    return success();
  }
};

// Result of recognizing the canonical torch-mlir chain emitted for
// `onnx.ArgMax/ArgMin{select_last_index=1}`:
//   %f = aten.flip %x, [dim]
//   %a = aten.argmax/argmin %f, dim, keepdim
//   %s = aten.sub.Scalar %a, dim_size-1, 1
//   %r = aten.abs %s
struct SelectLastIndexChain {
  torch::Torch::AtenFlipOp flip;
  torch::Torch::AtenSubScalarOp sub;
  torch::Torch::AtenAbsOp abs;
  Value flipInput;
};

// Snapshot of the input state seen by the chain matcher after any dim=None
// flatten has already been applied. Grouped so the matcher's signature stays
// short and the call site reads as one logical "post-flatten state" argument.
struct ArgInputState {
  Value self;
  ArrayRef<int64_t> shape;
  int64_t rank;
};

// Returns the matched chain when `argOp` participates in the
// flip -> argmax/argmin -> sub.Scalar -> abs sequence on `dim`. Returns
// `std::nullopt` for any precondition miss; the caller falls back to a
// strict-comparator lowering.
template <typename OpTy>
static std::optional<SelectLastIndexChain>
tryMatchSelectLastIndexChain(OpTy argOp, int64_t dim, bool dimIsNone,
                             const ArgInputState &state) {
  if (dimIsNone) {
    return std::nullopt;
  }
  auto flipOp = state.self.getDefiningOp<torch::Torch::AtenFlipOp>();
  if (!flipOp || !flipOp->hasOneUse() || !argOp->hasOneUse()) {
    return std::nullopt;
  }
  SmallVector<int64_t> flipDims;
  if (!matchPattern(flipOp.getDims(),
                    torch::Torch::m_TorchListOfConstantInts(flipDims)) ||
      flipDims.size() != 1) {
    return std::nullopt;
  }
  int64_t flipDim = flipDims[0];
  if (flipDim < 0) {
    flipDim += state.rank;
  }
  if (flipDim != dim || state.shape[dim] == torch::Torch::kUnknownSize) {
    return std::nullopt;
  }
  auto subOp = dyn_cast<torch::Torch::AtenSubScalarOp>(*argOp->user_begin());
  if (!subOp || !subOp->hasOneUse()) {
    return std::nullopt;
  }
  int64_t subVal, subAlpha;
  if (!matchPattern(subOp.getOther(),
                    torch::Torch::m_TorchConstantInt(&subVal)) ||
      !matchPattern(subOp.getAlpha(),
                    torch::Torch::m_TorchConstantInt(&subAlpha)) ||
      subAlpha != 1 || subVal != state.shape[dim] - 1) {
    return std::nullopt;
  }
  auto absOp = dyn_cast<torch::Torch::AtenAbsOp>(*subOp->user_begin());
  if (!absOp) {
    return std::nullopt;
  }
  return SelectLastIndexChain{flipOp, subOp, absOp, flipOp.getSelf()};
}

template <typename OpTy, bool isMax>
struct ArgCompareOpConversion : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value self = op.getSelf();

    auto inputTensorType = cast<torch::Torch::ValueTensorType>(self.getType());
    if (!inputTensorType.hasSizes()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected input type having sizes");
    }
    ArrayRef<int64_t> inputShape = inputTensorType.getSizes();
    int64_t inputRank = inputShape.size();
    if (inputRank == 0) {
      return rewriter.notifyMatchFailure(op, "expected input with rank > 0");
    }

    int64_t dim;
    bool dimIsNone = isa<torch::Torch::NoneType>(op.getDim().getType());
    if (!dimIsNone &&
        !matchPattern(op.getDim(), torch::Torch::m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(op, "requires dim to be constant");
    }

    bool keepdim;
    if (!matchPattern(op.getKeepdim(),
                      torch::Torch::m_TorchConstantBool(&keepdim))) {
      return rewriter.notifyMatchFailure(op, "requires keepdim to be constant");
    }

    // Captured before flatten collapses inputRank to 1 — needed to rebuild
    // the all-ones keepdim shape in the dim=None case.
    int64_t preFlattenRank = inputRank;

    // Handle dim=None: flatten to 1D, reduce on dim 0.
    if (dimIsNone) {
      Value cstZero = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(0));
      Value cstLastDim = torch::Torch::ConstantIntOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(inputRank - 1));

      int64_t flattenedSize = torch::Torch::kUnknownSize;
      if (llvm::none_of(inputShape, [](int64_t s) {
            return s == torch::Torch::kUnknownSize;
          })) {
        flattenedSize = 1;
        for (int64_t s : inputShape) {
          flattenedSize *= s;
        }
      }

      auto flattenedType = inputTensorType.getWithSizesAndDtype(
          {flattenedSize}, inputTensorType.getDtype());
      self = torch::Torch::AtenFlattenUsingIntsOp::create(
          rewriter, loc, flattenedType, self, cstZero, cstLastDim);
      inputTensorType = cast<torch::Torch::ValueTensorType>(self.getType());
      inputShape = inputTensorType.getSizes();
      inputRank = 1;
      dim = 0;
    }

    if (dim < 0) {
      dim += inputRank;
    }

    // arg_compare with a non-strict comparator (oge/ole) computes the same
    // last-index result directly from %x, eliminating the flip and the
    // post-process. Fusing the flip into a non-affine `tensor.extract` later
    // produces an `llvm.intr.masked.gather` that the AMDGPU backend cannot
    // codegen efficiently, so this rewrite is a soundness *and* a compile-time
    // win.
    std::optional<SelectLastIndexChain> chain = tryMatchSelectLastIndexChain(
        op, dim, dimIsNone, ArgInputState{self, inputShape, inputRank});
    bool useNonStrictPredicate = chain.has_value();
    if (useNonStrictPredicate) {
      self = chain->flipInput;
      inputTensorType = cast<torch::Torch::ValueTensorType>(self.getType());
      inputShape = inputTensorType.getSizes();
    }

    // Capture original signedness from the Torch dtype before lowering to a
    // signless builtin type. arith comparison and min/max ops need the
    // matching signed/unsigned variant for unsigned integer inputs (e.g.
    // ui8/ui16) — picking signed `sgt`/`maxs` on a `ui8` tensor would read
    // 0xFF as -1 and silently return the wrong argmax/argmin index.
    bool isUnsigned = false;
    if (auto torchIntTy = dyn_cast<IntegerType>(inputTensorType.getDtype())) {
      isUnsigned = torchIntTy.isUnsigned();
    }

    auto builtinTensorType =
        cast<RankedTensorType>(inputTensorType.toBuiltinTensor());
    Type elemType = builtinTensorType.getElementType();
    // Ensure signless integer types for compatibility with arith ops.
    auto intTy = dyn_cast<IntegerType>(elemType);
    if (intTy && !intTy.isSignless()) {
      elemType = IntegerType::get(rewriter.getContext(), intTy.getWidth());
      builtinTensorType =
          RankedTensorType::get(builtinTensorType.getShape(), elemType);
    }
    Value builtinInput = torch::TorchConversion::ToBuiltinTensorOp::create(
        rewriter, loc, builtinTensorType, self);

    ArrayRef<int64_t> builtinShape = builtinTensorType.getShape();
    SmallVector<int64_t> outputShape;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i != dim) {
        outputShape.push_back(builtinShape[i]);
      }
    }

    auto outputValueType = RankedTensorType::get(outputShape, elemType);
    auto i32Type = rewriter.getI32Type();
    auto outputIndexType = RankedTensorType::get(outputShape, i32Type);

    SmallVector<Value> dynamicDims;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i != dim && builtinShape[i] == ShapedType::kDynamic) {
        Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        dynamicDims.push_back(
            tensor::DimOp::create(rewriter, loc, builtinInput, dimIdx));
      }
    }

    Value outputValueEmpty = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                                     elemType, dynamicDims);
    Value outputIndexEmpty = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                                     i32Type, dynamicDims);

    bool isFloat = isa<FloatType>(elemType);
    arith::AtomicRMWKind initKind;
    if (isFloat) {
      initKind = isMax ? arith::AtomicRMWKind::maximumf
                       : arith::AtomicRMWKind::minimumf;
    } else if (isUnsigned) {
      initKind =
          isMax ? arith::AtomicRMWKind::maxu : arith::AtomicRMWKind::minu;
    } else {
      initKind =
          isMax ? arith::AtomicRMWKind::maxs : arith::AtomicRMWKind::mins;
    }
    // Seed with the true identity (+/-inf for floats), not a finite extremum.
    // PyTorch's argmax/argmin treats -FLT_MAX as strictly greater than -inf,
    // so seeding with -FLT_MAX would silently return index 0 on inputs like
    // [-inf, -FLT_MAX] instead of 1. Strict `ogt`/`olt` already filter NaN
    // operands, so the inf seed does not change NaN-handling behavior.
    Value valueInit = arith::getIdentityValue(initKind, elemType, rewriter, loc,
                                              /*useOnlyFiniteValue=*/false);
    Value indexInit =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));

    Value filledValue =
        linalg::FillOp::create(rewriter, loc, ValueRange{valueInit},
                               outputValueEmpty)
            .getResult(0);
    Value filledIndex =
        linalg::FillOp::create(rewriter, loc, ValueRange{indexInit},
                               outputIndexEmpty)
            .getResult(0);

    // Null `input_index` and `index_base` mean "indices start at 0", which
    // matches the i32(0) `linalg.fill` seed for `filledIndex` above.
    auto argCompareOp = IREE::LinalgExt::ArgCompareOp::create(
        rewriter, loc, TypeRange{outputValueType, outputIndexType},
        builtinInput, /*input_index=*/Value(), filledValue, filledIndex,
        /*index_base=*/Value(), rewriter.getI64IntegerAttr(dim));

    {
      Region &region = argCompareOp.getRegion();
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&region, region.end(),
                                          {elemType, elemType}, {loc, loc});
      Value lhs = block->getArgument(0);
      Value rhs = block->getArgument(1);
      Value cmp;
      if (isFloat) {
        arith::CmpFPredicate pred =
            isMax ? (useNonStrictPredicate ? arith::CmpFPredicate::OGE
                                           : arith::CmpFPredicate::OGT)
                  : (useNonStrictPredicate ? arith::CmpFPredicate::OLE
                                           : arith::CmpFPredicate::OLT);
        cmp = arith::CmpFOp::create(rewriter, loc, pred, lhs, rhs);
      } else {
        arith::CmpIPredicate pred;
        if (isMax) {
          if (useNonStrictPredicate) {
            pred = isUnsigned ? arith::CmpIPredicate::uge
                              : arith::CmpIPredicate::sge;
          } else {
            pred = isUnsigned ? arith::CmpIPredicate::ugt
                              : arith::CmpIPredicate::sgt;
          }
        } else {
          if (useNonStrictPredicate) {
            pred = isUnsigned ? arith::CmpIPredicate::ule
                              : arith::CmpIPredicate::sle;
          } else {
            pred = isUnsigned ? arith::CmpIPredicate::ult
                              : arith::CmpIPredicate::slt;
          }
        }
        cmp = arith::CmpIOp::create(rewriter, loc, pred, lhs, rhs);
      }
      IREE::LinalgExt::YieldOp::create(rewriter, loc, cmp);
    }

    Value indexResult = argCompareOp.getResult(1);
    auto torchResultType =
        cast<torch::Torch::ValueTensorType>(op.getResult().getType());

    // The arg_compare op produces i32 indices, but PyTorch expects i64.
    auto resultDtype = torchResultType.getDtype();
    Type builtinIndexType = resultDtype;
    if (auto resultIntTy = dyn_cast<IntegerType>(resultDtype)) {
      builtinIndexType =
          IntegerType::get(rewriter.getContext(), resultIntTy.getWidth());
    }

    if (builtinIndexType != i32Type) {
      auto i64OutputType = RankedTensorType::get(outputShape, builtinIndexType);
      Value emptyI64 = tensor::EmptyOp::create(rewriter, loc, outputShape,
                                               builtinIndexType, dynamicDims);
      auto identityMap = AffineMap::getMultiDimIdentityMap(
          outputShape.size(), rewriter.getContext());
      SmallVector<utils::IteratorType> iterTypes(outputShape.size(),
                                                 utils::IteratorType::parallel);
      indexResult =
          linalg::GenericOp::create(
              rewriter, loc, i64OutputType, ValueRange{indexResult},
              ValueRange{emptyI64},
              SmallVector<AffineMap>{identityMap, identityMap}, iterTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value ext =
                    arith::ExtSIOp::create(b, loc, builtinIndexType, args[0]);
                linalg::YieldOp::create(b, loc, ext);
              })
              .getResult(0);
    }

    // In the chain-fold case the terminal `abs` is the user-visible op and
    // the argmax/sub/flip behind it become dead; otherwise we replace the
    // original argmax/argmin directly.
    Operation *replaceTarget = op.getOperation();
    if (useNonStrictPredicate) {
      replaceTarget = chain->abs.getOperation();
    }

    Value finalReplacement;
    if (keepdim) {
      // Re-insert the reduced dimension as size 1 via torch ops that will be
      // lowered by the subsequent ConvertTorchToLinalgPass.
      SmallVector<int64_t> torchOutputShape(outputShape.begin(),
                                            outputShape.end());
      auto intermediateTorchType = torch::Torch::ValueTensorType::get(
          rewriter.getContext(), torchOutputShape, resultDtype);
      Value torchResult = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, intermediateTorchType, indexResult);

      if (dimIsNone) {
        // For dim=None, the result is scalar; reshape to all-ones shape.
        SmallVector<Value> shapeValues;
        Value cstOne = torch::Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        for (int64_t i = 0; i < preFlattenRank; ++i) {
          shapeValues.push_back(cstOne);
        }
        auto listType = torch::Torch::ListType::get(
            torch::Torch::IntType::get(rewriter.getContext()));
        Value shapeList = torch::Torch::PrimListConstructOp::create(
            rewriter, loc, listType, shapeValues);
        finalReplacement = torch::Torch::AtenViewOp::create(
            rewriter, loc, torchResultType, torchResult, shapeList);
      } else {
        Value cstDim = torch::Torch::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(dim));
        finalReplacement = torch::Torch::AtenUnsqueezeOp::create(
            rewriter, loc, torchResultType, torchResult, cstDim);
      }
    } else {
      // Without keepdim, the reduced shape matches the torch result directly.
      finalReplacement = torch::TorchConversion::FromBuiltinTensorOp::create(
          rewriter, loc, torchResultType, indexResult);
    }

    rewriter.replaceOp(replaceTarget, finalReplacement);
    if (useNonStrictPredicate) {
      // Erase the now-dead intermediate ops in chain order: sub, argmax, flip.
      rewriter.eraseOp(chain->sub);
      rewriter.eraseOp(op);
      rewriter.eraseOp(chain->flip);
    }
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
    patterns.add<ArgCompareOpConversion<torch::Torch::AtenArgmaxOp,
                                        /*isMax=*/true>>(context);
    patterns.add<ArgCompareOpConversion<torch::Torch::AtenArgminOp,
                                        /*isMax=*/false>>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
