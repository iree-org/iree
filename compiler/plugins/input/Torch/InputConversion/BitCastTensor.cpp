// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_BITCASTTENSORPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

class BitCastViewDtype
    : public OpRewritePattern<torch::Torch::AtenViewDtypeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(torch::Torch::AtenViewDtypeOp op,
                                PatternRewriter &rewriter) const override {

    Value in = op.getSelf();
    auto loc = op.getLoc();
    auto inType = cast<torch::Torch::ValueTensorType>(in.getType());
    auto resultType = cast<torch::Torch::ValueTensorType>(op.getType());

    auto bType = inType.toBuiltinTensor();

    if (auto dtype = dyn_cast<IntegerType>(bType.getElementType())) {
      bType = bType.clone(
          rewriter.getType<IntegerType>(dtype.getIntOrFloatBitWidth()));
    }

    // Cast to the builtin tensor type.
    Value builtinCast =
        rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, bType,
                                                                   in);

    auto rType = resultType.toBuiltinTensor();
    if (auto dtype = dyn_cast<IntegerType>(rType.getElementType())) {
      rType = rType.clone(
          rewriter.getType<IntegerType>(dtype.getIntOrFloatBitWidth()));
    }

    SmallVector<Value> inDynamicDims =
        IREE::Util::buildDynamicDimsForValue(loc, builtinCast, rewriter);
    SmallVector<Value> outDynamicDims(inDynamicDims);
    if (bType.isDynamicDim(bType.getRank() - 1)) {
      auto inElemWidth = IREE::Util::getTypeBitWidth(bType.getElementType());
      auto outElemWidth = IREE::Util::getTypeBitWidth(rType.getElementType());
      if (inElemWidth > outElemWidth) {
        auto scale = rewriter.create<arith::ConstantIndexOp>(
            loc, inElemWidth / outElemWidth);
        outDynamicDims.back() = rewriter.create<arith::DivSIOp>(
            loc, inDynamicDims.back(), scale.getResult());

      } else if (inElemWidth < outElemWidth) {
        auto scale = rewriter.create<arith::ConstantIndexOp>(
            loc, outElemWidth / inElemWidth);
        outDynamicDims.back() = rewriter.create<arith::MulIOp>(
            loc, outDynamicDims.back(), scale.getResult(),
            arith::IntegerOverflowFlags::nsw);
      }
    }

    Value flowBitcast = rewriter.create<IREE::TensorExt::BitCastOp>(
        loc, rType, builtinCast, inDynamicDims, outDynamicDims);

    auto torchCast =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, resultType, flowBitcast);
    rewriter.replaceOp(op, torchCast);
    return success();
  }
};

template <typename BitcastOp>
class BitCastViewComplex : public OpRewritePattern<BitcastOp> {
public:
  using OpRewritePattern<BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BitcastOp op,
                                PatternRewriter &rewriter) const override {

    Value in = op.getSelf();
    auto loc = op.getLoc();
    auto inType = cast<torch::Torch::ValueTensorType>(in.getType());
    auto resultType = cast<torch::Torch::ValueTensorType>(op.getType());
    auto bType = inType.toBuiltinTensor();

    Value builtinCast =
        rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, bType,
                                                                   in);
    auto rType = resultType.toBuiltinTensor();

    // Cast to the builtin tensor type.
    llvm::SmallVector<Value> dynDims;
    for (int i = 0, s = bType.getRank(); i < s; ++i) {
      if (bType.isDynamicDim(i)) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, builtinCast, i));
      }
    }

    Value flowBitcast = rewriter.create<IREE::TensorExt::BitCastOp>(
        loc, rType, builtinCast, dynDims, dynDims);

    auto torchCast =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, resultType, flowBitcast);
    rewriter.replaceOp(op, torchCast);
    return success();
  }
};

class BitCastMatmul : public OpRewritePattern<torch::Torch::OperatorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(torch::Torch::OperatorOp op,
                                PatternRewriter &rewriter) const override {
    // Check for group quantized matrix multiplications.
    if (op.getName().str() != "quant.matmul_rhs_group_quant") {
      return failure();
    }

    Value rhs = op.getOperand(1);
    Value bitWidth = op.getOperand(4);

    // Extract the target bitwidth from the constant on the matmul.
    auto getConstantIntegerFromDefiningOp = [](Value operand,
                                               int &extractedInt) {
      auto constOp =
          dyn_cast<torch::Torch::ConstantIntOp>(operand.getDefiningOp());
      if (!constOp) {
        return failure();
      }
      extractedInt = constOp.getValue();
      return success();
    };
    int unpackedBitWidth;
    if (failed(getConstantIntegerFromDefiningOp(bitWidth, unpackedBitWidth)))
      return failure();

    auto rhsType = dyn_cast<torch::Torch::ValueTensorType>(rhs.getType());
    if (!rhsType)
      return failure();

    if (!rhsType.hasDtype())
      return failure();

    Type dType = rhsType.getDtype();
    int dTypeWidth = dType.getIntOrFloatBitWidth();
    // If the dtype width already matches the target width, nothing to do.
    if (dTypeWidth == unpackedBitWidth)
      return failure();

    if (!rhsType.hasSizes())
      return failure();

    SmallVector<int64_t> tensorShape(rhsType.getSizes());
    // Constants should have constant shape.
    if (llvm::any_of(tensorShape,
                     [](int64_t s) { return s == torch::Torch::kUnknownSize; }))
      return failure();
    int packRatio = dTypeWidth / unpackedBitWidth;

    tensorShape[tensorShape.size() - 1] *= packRatio;

    Location loc = op.getLoc();
    auto bitCastTargetType = RankedTensorType::get(
        tensorShape, rewriter.getIntegerType(unpackedBitWidth));

    // Cast to the builtin tensor type.
    auto builtinCast =
        rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(
            loc, rhsType.toBuiltinTensor(), rhs);

    // No dynamic dims because we are bitcasting a constant.
    auto flowBitcast = rewriter.create<IREE::TensorExt::BitCastOp>(
        loc, bitCastTargetType, builtinCast, ValueRange(), ValueRange());

    // Cast back to the (un)signed torch tensor type to inform later lowerings.
    Type unpackedElementType;
    if (dType.isSignedInteger())
      unpackedElementType = rewriter.getIntegerType(unpackedBitWidth, true);
    else
      unpackedElementType = rewriter.getIntegerType(unpackedBitWidth, false);
    torch::Torch::ValueTensorType newRhsType =
        torch::Torch::ValueTensorType::get(rewriter.getContext(), tensorShape,
                                           unpackedElementType);
    auto torchCast =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, newRhsType, flowBitcast);
    op->replaceUsesOfWith(rhs, torchCast);
    return success();
  }
};
} // namespace

namespace {
class BitCastTensorPass final
    : public impl::BitCastTensorPassBase<BitCastTensorPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::TensorExt::IREETensorExtDialect>();
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BitCastMatmul, BitCastViewDtype>(context);
    patterns.add<BitCastViewComplex<torch::Torch::AtenViewAsComplexOp>,
                 BitCastViewComplex<torch::Torch::AtenViewAsRealOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
