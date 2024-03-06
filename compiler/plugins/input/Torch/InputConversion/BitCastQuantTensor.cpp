// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/PassDetail.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler::TorchInput {

namespace {

class BitCastQuantizedMatmul
    : public OpRewritePattern<torch::Torch::OperatorOp> {
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

    auto rhsType = rhs.getType().dyn_cast<torch::Torch::ValueTensorType>();
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
        rewriter.create<torch::TorchConversion::ToBuiltinTensorOp>(loc, rhs);

    // No dynamic dims because we are bitcasting a constant.
    auto flowBitcast = rewriter.create<IREE::Flow::TensorBitCastOp>(
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
class BitCastQuantTensorPass
    : public BitCastQuantTensorPassBase<BitCastQuantTensorPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<torch::Torch::TorchDialect>();
    registry.insert<torch::TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BitCastQuantizedMatmul>(context);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createBitCastQuantTensorPass() {
  return std::make_unique<BitCastQuantTensorPass>();
}

} // namespace mlir::iree_compiler::TorchInput
